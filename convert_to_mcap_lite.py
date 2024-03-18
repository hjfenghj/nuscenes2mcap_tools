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

def get_pcd(data_path, timestamp, frame_id) -> PointCloud:
    pc_filename = data_path
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = PointCloud()
    msg.frame_id = frame_id
    msg.timestamp.FromMicroseconds(timestamp)
    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        msg.fields.add(name=name, offset=offset, type=PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)])
        offset += size

    msg.point_stride = offset
    msg.data = pc.pc_data.tobytes()
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

    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)
        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")

        # 用于轨迹可视化, 记录每个目标ID对应生命周期的所有位置信息
        trackPoints = {}
        # 用于轨迹可视化是体现真值ID跳变
        trackP2PColor = {}

        while cur_sample is not None:

            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            ego2globalTran = ego_pose["translation"]
            ego2globalRotMat = Quaternion(ego_pose['rotation']).rotation_matrix  # used convert " " from global to ego
            stamp = get_time(ego_pose)

            ############## 发布自车系到世界系的tf变换 ###############
            protobuf_writer.write_message("/tf", get_ego_tf(ego_pose), stamp.to_nsec())

            # iterate sensors
            for sensor_id, sample_token in cur_sample["data"].items():
                pbar.update(1)
                sample_data = nusc.get("sample_data", sample_token)

                ############ 发布自车系到各传感器坐标系的tf变换 ##################
                protobuf_writer.write_message("/tf", get_sensor_tf(nusc, sensor_id, sample_data), stamp.to_nsec())

                ############ 发布各传感器数据  ##################
                if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message( "/" + sensor_id, msg, stamp.to_nsec())

                elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message( "/" + sensor_id, msg, stamp.to_nsec())
                
                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message( "/" + sensor_id + "/image_rect_compressed", msg, stamp.to_nsec())
                    ############# 相机内外参 ##############
                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    protobuf_writer.write_message( "/" + sensor_id + "/camera_info", msg, stamp.to_nsec())

                else:
                    pass

            ############# 可视化各种marker ##############
            annsSceneUpdate           = SceneUpdate()
            annsIdSceneUpdate       = SceneUpdate()
            annsVeloXSceneUpdate      = SceneUpdate()
            annsVeloYSceneUpdate      = SceneUpdate()
            annsVeloArrowSceneUpdate  = SceneUpdate()
            annsHeadingSceneUpdate    = SceneUpdate()
            trackSceneUpdate          = SceneUpdate()

            for annotation_id in cur_sample["anns"]:
                ann = nusc.get("sample_annotation", annotation_id)
                annId = ann["instance_token"][:4]
                classColor = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

                ############ 横向/纵向分速度, 用于后续拉速度曲线使用 ################
                velo2dGlobal = nusc.box_velocity(annotation_id)[:2]                                # 世界系下的速度(大小, 方向)
                veloEgo      = np.array([*velo2dGlobal, 0.0]) @ np.linalg.inv(ego2globalRotMat).T  # 自车系下的速度(绝对速度范数不变, 方向转化为自车系下的速度方向)
                velo2dEgo    = veloEgo[:2]

                ############ GT框的方位角变换, 用于GT框可视化的方位 #################
                annCenterGlobal = np.array([ann["translation"][0], ann["translation"][1], ann["translation"][2]])                  # 世界系下的GT框位置
                annCenterEgo    = np.dot(np.linalg.inv(ego2globalRotMat), annCenterGlobal - np.array(ego2globalTran))              # 自车系下的GT框位置
                annOrientGlobal  = np.array([ann["rotation"][0], ann["rotation"][1], ann["rotation"][2], ann["rotation"][3]])      # 框在世界系下的方位角四元数
                gloabl2egoOrient = Quaternion(matrix = np.linalg.inv(ego2globalRotMat))                                            # 世界系到自车系下的四元数变换
                annOrientEgo = gloabl2egoOrient * annOrientGlobal                                                                  # 框在自车系下的方位角四元数

                ############ 记录每个ID的各轨迹点和前后两帧轨迹线的颜色, 用于可视化轨迹和观察真值的ID跳变 ################
                if annId not in trackPoints:
                    trackPoints.setdefault(annId,[])
                trackPoints[annId].append([ann["translation"][0],ann["translation"][1],ann["translation"][2]])
                
                if annId not in trackP2PColor:
                    trackP2PColor[annId] = [random.random(),random.random(),random.random(),random.random()]

                ############# 发布GT框marker ###################
                annsDeleteEntity      = annsSceneUpdate.deletions.add()
                annsDeleteEntity.type = 1
                annsDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                annsEntity = annsSceneUpdate.entities.add()
                annsEntity.timestamp.FromNanoseconds(stamp.to_nsec())

                annsEntity.frame_id         = "map"
                annsEntity.id               = annId
                annsEntity.frame_locked     = True
                annscube                    = annsEntity.cubes.add()
                annscube.pose.position.x    = ann["translation"][0]
                annscube.pose.position.y    = ann["translation"][1]
                annscube.pose.position.z    = ann["translation"][2]
                annscube.pose.orientation.w = ann["rotation"][0]
                annscube.pose.orientation.x = ann["rotation"][1]
                annscube.pose.orientation.y = ann["rotation"][2]
                annscube.pose.orientation.z = ann["rotation"][3]
                annscube.size.x             = ann["size"][1]
                annscube.size.y             = ann["size"][0]
                annscube.size.z             = ann["size"][2]
                annscube.color.r            = classColor[0]
                annscube.color.g            = classColor[1]
                annscube.color.b            = classColor[2]
                annscube.color.a            = 0.5


                #################### 发布GT框ID信息, 与GT框话题分离发布是方便关闭ID可视化 #################
                annsIdDeleteEntity      = annsIdSceneUpdate.deletions.add()
                annsIdDeleteEntity.type = 1
                annsIdDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                annsIdEntity = annsIdSceneUpdate.entities.add()
                annsIdEntity.timestamp.FromNanoseconds(stamp.to_nsec())

                annsIdEntity.frame_id          = "map"
                annsIdEntity.id                = annId
                annsIdEntity.frame_locked      = True
                annsIdTexts                    = annsIdEntity.texts.add()
                annsIdTexts.pose.position.x    = ann["translation"][0]
                annsIdTexts.pose.position.y    = ann["translation"][1]
                annsIdTexts.pose.position.z    = ann["translation"][2] + ann["size"][2]/2
                annsIdTexts.pose.orientation.w = ann["rotation"][0]
                annsIdTexts.pose.orientation.x = ann["rotation"][1]
                annsIdTexts.pose.orientation.y = ann["rotation"][2]
                annsIdTexts.pose.orientation.z = ann["rotation"][3]
                annsIdTexts.font_size          = 0.7
                annsIdTexts.color.r            = classColor[0]
                annsIdTexts.color.g            = classColor[1]
                annsIdTexts.color.b            = classColor[2]
                annsIdTexts.color.a            = 1
                annsIdTexts.text               = annId

                ################ 发布纵向速度信息marker, 用于拉速度曲线使用 ################
                annsVeloXDeleteEntity        = annsVeloXSceneUpdate.deletions.add()
                annsVeloXDeleteEntity.type   = 1

                annsVeloXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                annsVeloXEntity              = annsVeloXSceneUpdate.entities.add()
                annsVeloXEntity.frame_id     = "base_link"

                annsVeloXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                annsVeloXEntity.id           = annId
                annsVeloXEntityTexts         = annsVeloXEntity.texts.add()
                annsVeloXEntity.frame_locked = True

                annsVeloXEntityTexts.pose.position.x    = annCenterEgo[0]
                annsVeloXEntityTexts.pose.position.y    = annCenterEgo[1]
                annsVeloXEntityTexts.pose.position.z    = annCenterEgo[2]
                annsVeloXEntityTexts.pose.orientation.w = annOrientEgo[0]
                annsVeloXEntityTexts.pose.orientation.x = annOrientEgo[1]
                annsVeloXEntityTexts.pose.orientation.y = annOrientEgo[2]
                annsVeloXEntityTexts.pose.orientation.z = annOrientEgo[3]
                annsVeloXEntityTexts.font_size          = 0.01
                annsVeloXEntityTexts.text               = str(velo2dEgo[0])

                ################ 发布横向速度信息marker, 用于拉速度曲线使用 ################
                annsVeloYDeleteEntity        = annsVeloYSceneUpdate.deletions.add()
                annsVeloYDeleteEntity.type   = 1

                annsVeloYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                annsVeloYEntity              = annsVeloYSceneUpdate.entities.add()
                annsVeloYEntity.frame_id     = "base_link"

                annsVeloYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                annsVeloYEntity.id           = annId
                annsVeloYEntityTexts         = annsVeloYEntity.texts.add()
                annsVeloYEntity.frame_locked = True
       
                annsVeloYEntityTexts.pose.position.x    = annCenterEgo[0]
                annsVeloYEntityTexts.pose.position.y    = annCenterEgo[1]
                annsVeloYEntityTexts.pose.position.z    = annCenterEgo[2]
                annsVeloYEntityTexts.pose.orientation.w = annOrientEgo[0]
                annsVeloYEntityTexts.pose.orientation.x = annOrientEgo[1]
                annsVeloYEntityTexts.pose.orientation.y = annOrientEgo[2]
                annsVeloYEntityTexts.pose.orientation.z = annOrientEgo[3]
                annsVeloYEntityTexts.font_size          = 0.01
                annsVeloYEntityTexts.text               = str(velo2dEgo[1])

                ################### 发布速度箭头marker, 用于观察车辆行驶方向 ##################
                annsVeloArrowDeleteEntity      = annsVeloArrowSceneUpdate.deletions.add()
                annsVeloArrowDeleteEntity.type = 1

                annsVeloArrowDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                annsVeloArrowEntity              = annsVeloArrowSceneUpdate.entities.add()
                annsVeloArrowEntity.id           = annId
                annsVeloArrowEntity.frame_id     = "base_link"
                annsVeloArrowEntity.frame_locked = True

                annsVeloArrowEntity.timestamp.FromNanoseconds(stamp.to_nsec())

                annsVeloArrowEntityLine           = annsVeloArrowEntity.lines.add()
                annsVeloArrowEntityLine.type      = 0
                annsVeloArrowEntityLine.thickness = 0.2
                annsVeloArrowEntityLine.color.r   = 218 / 255.0
                annsVeloArrowEntityLine.color.g   = 112 / 255.0
                annsVeloArrowEntityLine.color.b   = 214 / 255.0
                annsVeloArrowEntityLine.color.a   = 0.8

                annsVeloArrowEntityLine.points.add(x = annCenterEgo[0], y = annCenterEgo[1], z = annCenterEgo[2])  

                velo2dEgoL2 = np.sqrt(velo2dEgo[0]**2 + velo2dEgo[1]**2)        # 速度的L2范数
                alpha       = 3 * np.tanh(velo2dEgoL2)/velo2dEgoL2                  
                if velo2dEgoL2 <= 6: # 21.6km/h 
                    annsVeloArrowEntityLine.points.add(x = annCenterEgo[0] + velo2dEgo[0] / 6.0, y = annCenterEgo[1] + velo2dEgo[1] / 6.0, z = annCenterEgo[2]) 
                else:
                    annsVeloArrowEntityLine.points.add(x = annCenterEgo[0] + velo2dEgo[0] * alpha, y = annCenterEgo[1] + velo2dEgo[1] * alpha, z = annCenterEgo[2]) 

                ################## 发布GT框heading信息, 用于观察车身朝向 ###############
                annsHeadingDeleteEntity      = annsHeadingSceneUpdate.deletions.add()
                annsHeadingDeleteEntity.type = 1
                annsHeadingDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                annsHeadingEntity              = annsHeadingSceneUpdate.entities.add()
                annsHeadingEntity.id           = annId
                annsHeadingEntity.frame_id     = "base_link"
                annsHeadingEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                annsHeadingEntity.frame_locked = True

                annsHeadingLine         = annsHeadingEntity.arrows.add()  
                annsHeadingLine.color.r = 255 / 255.0
                annsHeadingLine.color.g = 211 / 255.0
                annsHeadingLine.color.b = 155 / 255.0
                annsHeadingLine.color.a = 0.5

                annsHeadingLine.pose.position.x    = annCenterEgo[0]
                annsHeadingLine.pose.position.y    = annCenterEgo[1]
                annsHeadingLine.pose.position.z    = annCenterEgo[2] + ann["size"][2]/2
                annsHeadingLine.pose.orientation.w = annOrientEgo[0]
                annsHeadingLine.pose.orientation.x = annOrientEgo[1]
                annsHeadingLine.pose.orientation.y = annOrientEgo[2]
                annsHeadingLine.pose.orientation.z = annOrientEgo[3]
                annsHeadingLine.shaft_length       = ann["size"][1]/2
                annsHeadingLine.shaft_diameter     = 0.07
                annsHeadingLine.head_length        = 0.01
                annsHeadingLine.head_diameter      = 0.01

            protobuf_writer.write_message("/markers/annotations", annsSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/annotationsID", annsIdSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/veloX", annsVeloXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/veloY", annsVeloYSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/velArrow", annsVeloArrowSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gt_heading", annsHeadingSceneUpdate, stamp.to_nsec()) 

            #################### 发布轨迹线和帧真值ID跳变 ##############
            for key, points in trackPoints.items():

                trackDeleteEntity = trackSceneUpdate.deletions.add()
                trackDeleteEntity.type = 1
                trackDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                trackEntity              = trackSceneUpdate.entities.add()
                trackEntity.id           = key
                trackEntity.frame_id     = "map"
                trackEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                trackEntity.frame_locked = True

                trackEntityLine                    = trackEntity.lines.add()
                trackEntityLine.type               = 0
                trackEntityLine.thickness          = 0.2
                trackEntityLine.color.r            = trackP2PColor[key][0]
                trackEntityLine.color.g            = trackP2PColor[key][1]
                trackEntityLine.color.b            = trackP2PColor[key][2]
                trackEntityLine.color.a            = trackP2PColor[key][3]
                trackEntityLine.pose.orientation.w = ann["rotation"][1]
                for point in points:
                    trackEntityLine.points.add(x = point[0], y = point[1], z = point[2])

            protobuf_writer.write_message("/markers/track_line", trackSceneUpdate, stamp.to_nsec())
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
