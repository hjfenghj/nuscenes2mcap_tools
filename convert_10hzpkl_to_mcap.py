import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from mcap.writer import Writer, CompressionType
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from tqdm import tqdm
import random
import pickle

from foxglove.CameraCalibration_pb2 import CameraCalibration
from foxglove.CompressedImage_pb2 import CompressedImage
from foxglove.FrameTransform_pb2 import FrameTransform
from foxglove.Grid_pb2 import Grid
from foxglove.ImageAnnotations_pb2 import ImageAnnotations
from foxglove.LinePrimitive_pb2 import LinePrimitive
from foxglove.LocationFix_pb2 import LocationFix
from foxglove.PackedElementField_pb2 import PackedElementField
from foxglove.PointCloud_pb2 import PointCloud
from foxglove.PoseInFrame_pb2 import PoseInFrame
from foxglove.PointsAnnotation_pb2 import PointsAnnotation
from foxglove.Quaternion_pb2 import Quaternion as foxglove_Quaternion
from foxglove.SceneUpdate_pb2 import SceneUpdate
from foxglove.Vector3_pb2 import Vector3
from ProtobufWriter import ProtobufWriter
from RosmsgWriter import RosmsgWriter

from foxglove.SceneEntityDeletion_pb2 import SceneEntityDeletion
from foxglove.TriangleListPrimitive_pb2 import TriangleListPrimitive
from nuscenes.utils import splits

def get_translation(data):
    return Vector3(x = data["translation"][0], y = data["translation"][1], z = data["translation"][2])

def get_rotation(data):
    return foxglove_Quaternion(x = data["rotation"][1], y = data["rotation"][2], z = data["rotation"][3], w = data["rotation"][0])


def get_time(data):
    t = rospy.Time()

    t.secs, msecs = divmod(data, 1_000_000)
    t.nsecs       = msecs * 1000

    return t

PCD_TO_PACKED_ELEMENT_TYPE_MAP = {  ("I", 1): PackedElementField.INT8,
                                    ("U", 1): PackedElementField.UINT8,
                                    ("I", 2): PackedElementField.INT16,
                                    ("U", 2): PackedElementField.UINT16,
                                    ("I", 4): PackedElementField.INT32,
                                    ("U", 4): PackedElementField.UINT32,
                                    ("F", 4): PackedElementField.FLOAT32,
                                    ("F", 8): PackedElementField.FLOAT64,}


def get_radar(data_path, sample_data, frame_id) -> PointCloud:
    pc  = pypcd.PointCloud.from_path(data_path / sample_data["filename"])
    msg = PointCloud()

    msg.frame_id = frame_id
    msg.timestamp.FromMicroseconds(sample_data["timestamp"])

    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        msg.fields.add(name = name, offset = offset, type = PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)])
        offset += size

    msg.point_stride = offset

    msg.data = pc.pc_data.tobytes()
    return msg


def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data["filename"]
    
    msg = CompressedImage()
    msg.timestamp.FromMicroseconds(sample_data["timestamp"])
    msg.frame_id = frame_id
    msg.format   = "jpeg"
    with open(jpg_filename, "rb") as jpg_file:
        msg.data = jpg_file.read()
    return msg

def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    msg_info = CameraCalibration()
    msg_info.timestamp.FromMicroseconds(sample_data["timestamp"])

    msg_info.frame_id = frame_id

    msg_info.height = sample_data["height"]
    
    msg_info.width = sample_data["width"]

    msg_info.K[:] = (calib["camera_intrinsic"][r][c] for r in range(3) for c in range(3))
    msg_info.R[:] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg_info.P[:] = [msg_info.K[0], msg_info.K[1], msg_info.K[2], 0, msg_info.K[3], msg_info.K[4], msg_info.K[5], 0, 0, 0, 1, 0]
    return msg_info


def get_lidar(data_path, timestamp, frame_id) -> PointCloud:
    pc_filename = data_path 

    with open(pc_filename, "rb") as pc_file:
        msg = PointCloud()

        msg.frame_id = frame_id
        msg.timestamp.FromMicroseconds(timestamp)
        msg.fields.add(name = "x", offset = 0, type = PackedElementField.FLOAT32),
        msg.fields.add(name = "y", offset = 4, type = PackedElementField.FLOAT32),
        msg.fields.add(name = "z", offset = 8, type = PackedElementField.FLOAT32),
        msg.fields.add(name = "intensity", offset = 12, type = PackedElementField.FLOAT32),
        msg.fields.add(name = "ring", offset = 16, type = PackedElementField.FLOAT32),
        
        msg.point_stride = len(msg.fields) * 4  # 4 bytes per field

        msg.data = pc_file.read()
        return msg


def get_ego_tf(timestamp,translation,rotation):
    ego_tf = FrameTransform()

    ego_tf.parent_frame_id = "map"
    ego_tf.timestamp.FromMicroseconds(timestamp)
    ego_tf.child_frame_id = "base_link"   
    ego_tf.translation.CopyFrom(Vector3(x = translation[0], y = translation[1], z = translation[2]))  #自车在世界坐标系中的位置
    ego_tf.rotation.CopyFrom(foxglove_Quaternion(x = rotation[1], y = rotation[2], z = rotation[3], w = rotation[0]))
    return ego_tf

def get_lidar_tf(sensor_id, timestamp,translation,rotation):
    sensor_tf = FrameTransform()

    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(timestamp)
    sensor_tf.child_frame_id  = sensor_id

    sensor_tf.translation.CopyFrom(Vector3(x = translation[0], y = translation[1], z = translation[2]))
    sensor_tf.rotation.CopyFrom(foxglove_Quaternion(x = rotation[1], y = rotation[2], z = rotation[3], w = rotation[0])) 
    return sensor_tf

def get_sensor_tf(nusc, sensor_id, sample_data):
    sensor_tf = FrameTransform()

    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(sample_data["timestamp"])
    sensor_tf.child_frame_id = sensor_id
     
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"]) # lidar2ego
    sensor_tf.translation.CopyFrom(get_translation(calibrated_sensor))
    sensor_tf.rotation.CopyFrom(get_rotation(calibrated_sensor))
    return sensor_tf

def get_car_scene_update(stamp) -> SceneUpdate:
    scene_update = SceneUpdate()

    entity = scene_update.entities.add()

    entity.frame_id = "base_link"

    entity.frame_locked = True

    entity.timestamp.FromNanoseconds(stamp)
    entity.id = "car"

    model = entity.models.add()

    model.pose.position.x = 1
    model.pose.orientation.w = 1

    model.scale.x = 1
    model.scale.y = 1
    model.scale.z = 1

    model.url = "https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    return scene_update

def write_scene_to_mcap(nusc: NuScenes, pkl_file, filepath, scene_token):

    filepath.parent.mkdir(parents = True, exist_ok = True)
    data_root = Path(nusc.dataroot)
    
    F = open(pkl_file, "rb")

    all_samples = pickle.load(F)

    pbar = tqdm(total = len(all_samples["infos"]), leave = False)

    with open(filepath, "wb") as fp:
        print(f"writing to {filepath}")

        writer = Writer(fp, compression = CompressionType.LZ4)

        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile = "", library = "nuscenes2mcap")

        for cur_info in all_samples["infos"]:
            if cur_info["scene_token"] != scene_token:
                continue
            pbar.update(1)
            stamp = get_time(cur_info["timestamp"])
            
            ############### 发布自车系到世界系的tf变换 ###############
            protobuf_writer.write_message("/tf", get_ego_tf(cur_info["timestamp"], cur_info["ego2global_translation"], cur_info["ego2global_rotation"]), stamp.to_nsec())
            
            ############### 发布原始点云 ##############
            data_path = os.path.join(data_root, *cur_info["lidar_path"].split(os.path.sep)[-3:])

            msg = get_lidar(data_path, cur_info["timestamp"], "LIDAR_TOP")
            protobuf_writer.write_message("/LIDAR_TOP", msg, stamp.to_nsec())

            ############### 发布激光雷达到自车系的tf变换 ####################
            protobuf_writer.write_message("/tf", get_lidar_tf("LIDAR_TOP", cur_info["timestamp"], cur_info["lidar2ego_translation"], cur_info["lidar2ego_rotation"]), stamp.to_nsec())

            ############### 发布6路相机的信息 #################
            for sensor_name in cur_info["cams"].keys():
                sample_data = nusc.get("sample_data", cur_info["cams"][sensor_name]["sample_data_token"])

                ################  发布相机到自车的tf  ##################
                protobuf_writer.write_message("/tf", get_sensor_tf(nusc, sensor_name, sample_data), stamp.to_nsec())
              
                ############### 发布图片信息 ##################
                msg = get_camera(data_root, sample_data, sensor_name)
                protobuf_writer.write_message("/" + sensor_name + "/image_rect_compressed", msg, stamp.to_nsec())

                ############### 发布相机内参 ##################
                msg = get_camera_info(nusc, sample_data, sensor_name)
                protobuf_writer.write_message("/" + sensor_name + "/camera_info", msg, stamp.to_nsec())

            ############## 发布标定框 #####################
            annsSceneUpdate = SceneUpdate()

            if not cur_info["is_key_frame"]:
                protobuf_writer.write_message("/markers/annotations", annsSceneUpdate, stamp.to_nsec())
                continue

            allAnnsToken = nusc.get("sample", cur_info["token"])["anns"]

            for annToken in allAnnsToken:
                ann = nusc.get("sample_annotation", annToken)

                marker_id = ann["instance_token"][:4]
                annColor  = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

                annsDeleteEntity = annsSceneUpdate.deletions.add()

                annsDeleteEntity.type = 1

                annsDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                annsEntity = annsSceneUpdate.entities.add()

                annsEntity.id = marker_id

                annsEntity.frame_id     = "map"
                annsEntity.frame_locked = True

                annsEntity.timestamp.FromNanoseconds(stamp.to_nsec())

                annscube = annsEntity.cubes.add()

                annscube.pose.position.x = ann["translation"][0]
                annscube.pose.position.y = ann["translation"][1]
                annscube.pose.position.z = ann["translation"][2]

                annscube.pose.orientation.w = ann["rotation"][0]
                annscube.pose.orientation.x = ann["rotation"][1]
                annscube.pose.orientation.y = ann["rotation"][2]
                annscube.pose.orientation.z = ann["rotation"][3]

                annscube.size.x  = ann["size"][1]
                annscube.size.y  = ann["size"][0]
                annscube.size.z  = ann["size"][2]
                annscube.color.r = annColor[0]
                annscube.color.g = annColor[1]
                annscube.color.b = annColor[2]
                annscube.color.a = 0.5

            protobuf_writer.write_message("/markers/annotations", annsSceneUpdate, stamp.to_nsec())

            # publish /markers/car
            protobuf_writer.write_message("/markers/car", get_car_scene_update(stamp.to_nsec()), stamp.to_nsec())

        pbar.close()
        writer.finish()
        print(f"Finished writing {filepath}")


def convert_all(output_dir, pkl_file, nusc, scene_name,):
    scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    write_scene_to_mcap(nusc, pkl_file, output_dir / f"{scene_name}-pkl_2hz.mcap", scene_token)

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument("--data-dir", "-d", default = script_dir / "data")
    parser.add_argument("--dataset-name", "-n", default = "v1.0-mini", nargs = "+")
    parser.add_argument("--output-dir", "-o", type = Path, default = script_dir / "output")
    parser.add_argument("--scene", "-s", nargs = "*")
    parser.add_argument("--list-only", action = "store_true")
    parser.add_argument("--pklFileName", default = "mini_10hz_nuscenes2d_temporal_infos_train.pkl")

    args = parser.parse_args()

    pkl_file = os.path.join(args.data_dir, args.pklFileName)

    nusc = NuScenes(version = args.dataset_name, dataroot = str(args.data_dir), verbose = True)

    if args.dataset_name == "v1.0-trainval":
        for scene_name in splits.train:
            convert_all(args.output_dir, pkl_file, nusc, scene_name)

    elif args.dataset_name == "v1.0-mini":
        for scene_name in splits.mini_train:
            convert_all(args.output_dir, pkl_file, nusc, scene_name)

    elif args.dataset_name == "v1.0-test":
        for scene_name in splits.test:
            convert_all(args.output_dir, pkl_file, nusc, scene_name)

    else:
        pass

if __name__ == "__main__":
    main()
