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

def get_translation(data):
    return Vector3(x=data["translation"][0], y=data["translation"][1], z=data["translation"][2])

def get_rotation(data):
    return foxglove_Quaternion(x=data["rotation"][1], y=data["rotation"][2], z=data["rotation"][3], w=data["rotation"][0])


def get_time(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data["timestamp"], 1_000_000)
    t.nsecs = msecs * 1000
    return t

PCD_TO_PACKED_ELEMENT_TYPE_MAP = {
    ("I", 1): PackedElementField.INT8,
    ("U", 1): PackedElementField.UINT8,
    ("I", 2): PackedElementField.INT16,
    ("U", 2): PackedElementField.UINT16,
    ("I", 4): PackedElementField.INT32,
    ("U", 4): PackedElementField.UINT32,
    ("F", 4): PackedElementField.FLOAT32,
    ("F", 8): PackedElementField.FLOAT64,
}


def get_radar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = PointCloud()
    msg.frame_id = frame_id
    msg.timestamp.FromMicroseconds(sample_data["timestamp"])
    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        msg.fields.add(name=name, offset=offset, type=PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)])
        offset += size

    msg.point_stride = offset
    msg.data = pc.pc_data.tobytes()
    return msg


def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data["filename"]
    msg = CompressedImage()
    msg.timestamp.FromMicroseconds(sample_data["timestamp"])
    msg.frame_id = frame_id
    msg.format = "jpeg"
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


def get_lidar(data_path, sample_data, frame_id) -> PointCloud:
    pc_filename = data_path / sample_data["filename"]

    with open(pc_filename, "rb") as pc_file:
        msg = PointCloud()
        msg.frame_id = frame_id
        msg.timestamp.FromMicroseconds(sample_data["timestamp"])
        msg.fields.add(name="x", offset=0, type=PackedElementField.FLOAT32),
        msg.fields.add(name="y", offset=4, type=PackedElementField.FLOAT32),
        msg.fields.add(name="z", offset=8, type=PackedElementField.FLOAT32),
        msg.fields.add(name="intensity", offset=12, type=PackedElementField.FLOAT32),
        msg.fields.add(name="ring", offset=16, type=PackedElementField.FLOAT32),
        msg.point_stride = len(msg.fields) * 4  # 4 bytes per field
        msg.data = pc_file.read()
        return msg


def get_ego_tf(ego_pose):
    ego_tf = FrameTransform()
    ego_tf.parent_frame_id = "map"
    ego_tf.timestamp.FromMicroseconds(ego_pose["timestamp"])
    ego_tf.child_frame_id = "base_link"                     #自车坐标系，随着车辆移动而变化
    ego_tf.translation.CopyFrom(get_translation(ego_pose))  #自车在世界坐标系中的位置
    ego_tf.rotation.CopyFrom(get_rotation(ego_pose))
    return ego_tf


def get_sensor_tf(nusc, sensor_id, sample_data):
    sensor_tf = FrameTransform()
    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(sample_data["timestamp"])
    sensor_tf.child_frame_id = sensor_id
    
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor_tf.translation.CopyFrom(get_translation(calibrated_sensor))
    sensor_tf.rotation.CopyFrom(get_rotation(calibrated_sensor))
    return sensor_tf

def get_car_scene_update(stamp) -> SceneUpdate:
    scene_update = SceneUpdate()
    entity = scene_update.entities.add()
    entity.frame_id = "base_link"
    entity.timestamp.FromNanoseconds(stamp)
    entity.id = "car"
    entity.frame_locked = True
    model = entity.models.add()
    model.pose.position.x = 1
    model.pose.orientation.w = 1
    model.scale.x = 1
    model.scale.y = 1
    model.scale.z = 1
    model.url = "https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    return scene_update

def get_num_sample_data(nusc: NuScenes, scene):
    num_sample_data = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sample_data = nusc.get("sample_data", sample_token)
        while sample_data is not None:
            if(sample_data["next"] != "" and sample_data["is_key_frame"]):
                num_sample_data += 1
            sample_data = nusc.get("sample_data", sample_data["next"]) if sample_data["next"] != "" else None
    return num_sample_data


def write_scene_to_mcap(nusc: NuScenes, nusc_can: NuScenesCanBus, scene, filepath):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    data_path = Path(nusc.dataroot)

    cur_sample = nusc.get("sample", scene["first_sample_token"])
    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene_name} Sample Data", leave=False)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    Dict_Token_id = dict()  # change the token of each intance to id
    id_idx = 0  # the first id is 0,第一次出现在视野中的id为0
    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)

        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")
        Line_point_memory = {}
        Line_point_color = {}
        while cur_sample is not None:
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            e2g_r = ego_pose['rotation']
            e2g_t = ego_pose["translation"]
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix  # used convert velo from global to ego
            stamp = get_time(ego_pose)

            # # publish /tf
            protobuf_writer.write_message("/tf", get_ego_tf(ego_pose), stamp.to_nsec())

            # iterate sensors
            for sensor_id, sample_token in cur_sample["data"].items():
                pbar.update(1)
                sample_data = nusc.get("sample_data", sample_token)
                topic = "/" + sensor_id

                # create sensor transform
                protobuf_writer.write_message("/tf", get_sensor_tf(nusc, sensor_id, sample_data), stamp.to_nsec())

                # write the sensor data
                if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message(topic, msg, stamp.to_nsec())
                elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message(topic, msg, stamp.to_nsec())
                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message(topic + "/image_rect_compressed", msg, stamp.to_nsec())
                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    protobuf_writer.write_message(topic + "/camera_info", msg, stamp.to_nsec())
            # publish /markers/annotations
            scene_update = SceneUpdate()
            Line_scene_update = SceneUpdate()
            gt_heading_scene_update = SceneUpdate()

            for annotation_id in cur_sample["anns"]:
                ann = nusc.get("sample_annotation", annotation_id)
                marker_id = ann["instance_token"][:4]
                c = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0
    
                if marker_id not in Line_point_memory:
                    Line_point_memory.setdefault(marker_id,[])
                Line_point_memory[marker_id].append([ann["translation"][0],ann["translation"][1],ann["translation"][2]])
                
                if marker_id not in Line_point_color:
                    Line_point_color[marker_id] = [random.random(),random.random(),random.random(),random.random()]

                delete_entity_all = scene_update.deletions.add()
                delete_entity_all.type = 1
                delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                entity = scene_update.entities.add()
                entity.frame_id = "map"
                entity.timestamp.FromNanoseconds(stamp.to_nsec())
                entity.id = marker_id
                entity.frame_locked = True
                cube = entity.cubes.add()
                cube.pose.position.x = ann["translation"][0]
                cube.pose.position.y = ann["translation"][1]
                cube.pose.position.z = ann["translation"][2]
                cube.pose.orientation.w = ann["rotation"][0]
                cube.pose.orientation.x = ann["rotation"][1]
                cube.pose.orientation.y = ann["rotation"][2]
                cube.pose.orientation.z = ann["rotation"][3]
                cube.size.x = ann["size"][1]
                cube.size.y = ann["size"][0]
                cube.size.z = ann["size"][2]
                cube.color.r = c[0]
                cube.color.g = c[1]
                cube.color.b = c[2]
                cube.color.a = 0.5

                texts = entity.texts.add()
                texts.pose.position.x = ann["translation"][0]
                texts.pose.position.y = ann["translation"][1]
                texts.pose.position.z = ann["translation"][2] + ann["size"][2]/2
                texts.pose.orientation.w = ann["rotation"][0]
                texts.pose.orientation.x = ann["rotation"][1]
                texts.pose.orientation.y = ann["rotation"][2]
                texts.pose.orientation.z = ann["rotation"][3]
                texts.font_size = 0.7
                texts.color.r = c[0]
                texts.color.g = c[1]
                texts.color.b = c[2]
                texts.color.a = 1
              
                 # 将nuscene目标的原始ID 映射为数字
                token_id = Dict_Token_id.get(ann["instance_token"])
                if token_id is None:
                    Dict_Token_id[ann["instance_token"]] = str(id_idx)
                    id_idx += 1
                    texts.text = Dict_Token_id[ann["instance_token"]]
                else:
                    texts.text = token_id

                # 目标heading
                ann_center = np.array([ann["translation"][0], ann["translation"][1], ann["translation"][2]])
                # conver ann_centerfrom global to ego
                ann_center = np.dot(np.linalg.inv(e2g_r_mat), ann_center-np.array(e2g_t))
                ann_orientation = np.array([ann["rotation"][0], ann["rotation"][1], ann["rotation"][2], ann["rotation"][3]])
                quaternion = Quaternion(matrix = np.linalg.inv(e2g_r_mat))
                ann_orientation = quaternion * ann_orientation

                gt_heading_entity = gt_heading_scene_update.entities.add()
                gt_heading_delete_entity_all = gt_heading_scene_update.deletions.add()
                gt_heading_delete_entity_all.type = 1
                gt_heading_delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 50)
                gt_heading_entity.id = marker_id
                gt_heading_entity.frame_id = "base_link"
                gt_heading_entity.timestamp.FromNanoseconds(stamp.to_nsec())
                gt_heading_entity.frame_locked = True
                Line_heading = gt_heading_entity.arrows.add()
                  
                Line_heading.color.r = 255 / 255.0
                Line_heading.color.g = 211 / 255.0
                Line_heading.color.b = 155 / 255.0
                Line_heading.color.a = 0.5
                Line_heading.pose.position.x = ann_center[0]
                Line_heading.pose.position.y = ann_center[1]
                Line_heading.pose.position.z = ann_center[2] + ann["size"][2]/2
                Line_heading.pose.orientation.w = ann_orientation[0]
                Line_heading.pose.orientation.x = ann_orientation[1]
                Line_heading.pose.orientation.y = ann_orientation[2]
                Line_heading.pose.orientation.z = ann_orientation[3]
                Line_heading.shaft_length = ann["size"][1]/2
                Line_heading.shaft_diameter = 0.07
                Line_heading.head_length = 0.01
                Line_heading.head_diameter = 0.01


            #ID跳变可视化
            for key,points in Line_point_memory.items():
                Line_entity = Line_scene_update.entities.add()

                Line_delete_entity_all = Line_scene_update.deletions.add()
                Line_delete_entity_all.type = 1
                Line_delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                Line_entity.id = key
                Line_entity.frame_id = "map"
                Line_entity.timestamp.FromNanoseconds(stamp.to_nsec())
                Line_entity.frame_locked = True
                Line = Line_entity.lines.add()
                Line.type = 0
                Line.thickness = 0.2
                Line.color.r = Line_point_color[key][0]
                Line.color.g = Line_point_color[key][1]
                Line.color.b = Line_point_color[key][2]
                Line.color.a = Line_point_color[key][3]
                Line.pose.orientation.w = ann["rotation"][1]
                for point in points:
                    Line.points.add(x = point[0],
                                    y = point[1],
                                    z = point[2])

            protobuf_writer.write_message("/markers/gt_heading", gt_heading_scene_update, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/track_line", Line_scene_update, stamp.to_nsec())
            protobuf_writer.write_message("/markers/annotations", scene_update, stamp.to_nsec())
            # publish /markers/car
            protobuf_writer.write_message("/markers/car", get_car_scene_update(stamp.to_nsec()), stamp.to_nsec())

            # move to the next sample
            cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

        pbar.close()
        writer.finish()
        print(f"Finished writing {filepath}")


def convert_all(
    output_dir: Path,
    name: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus,
    selected_scenes,
):
    nusc.list_scenes()
    for scene in nusc.scene:
        scene_name = scene["name"]
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        mcap_name = f"NuScenes-{name}-{scene_name}.mcap"
        write_scene_to_mcap(nusc, nusc_can, scene, output_dir / mcap_name)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument("--data-dir", "-d", default=script_dir / "data")                # 原始数据集路径, 需要挂载进docker镜像
    parser.add_argument("--dataset-name", "-n", default="v1.0-mini", nargs="+")       # 选取待转换的数据集split, "+"表示这个参数可以接受多个值, 需要使用空格隔开，这个参数必须传入值。否则会报错
    parser.add_argument("--output-dir", "-o", type=Path, default=script_dir / "output") # 转化成果的mcap文件储存路径
    parser.add_argument("--scene", "-s", nargs="*")                                     # 选取特定的scene完成mcap的转换, "*"表示这个参数可以接受多个值，需要使用空格格开, 也可以没有值
    parser.add_argument("--list-only", action="store_true", help="lists the scenes and exits")

    args = parser.parse_args()

    nusc_can = NuScenesCanBus(dataroot=str(args.data_dir))

    nusc = NuScenes(version=args.dataset_name, dataroot=str(args.data_dir), verbose=True)
    convert_all(args.output_dir, args.dataset_name, nusc, nusc_can, args.scene)


if __name__ == "__main__":
    main()
