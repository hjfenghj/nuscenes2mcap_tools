import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Dict, Tuple
import warnings
import lzf

import numpy as np
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from mcap.writer import Writer, CompressionType
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm
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
from nuscenes.utils.splits import create_splits_scenes
from pypcd import pypcd
import random
from scipy.spatial.transform import Rotation as R

with open(Path(__file__).parent / "turbomap.json") as f:
    TURBOMAP_DATA = np.array(json.load(f))

def get_translation(data):
    return Vector3(
        x=data["translation"][0], y=data["translation"][1], z=data["translation"][2]
    )

def get_rotation(data):
    return foxglove_Quaternion(
        x=data["rotation"][1],
        y=data["rotation"][2],
        z=data["rotation"][3],
        w=data["rotation"][0],
    )


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
        msg.fields.add(
            name=name, offset=offset, type=PCD_TO_PACKED_ELEMENT_TYPE_MAP[(ty, size)]
        )
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
    msg_info.K[:] = (
        calib["camera_intrinsic"][r][c] for r in range(3) for c in range(3)
    )
    msg_info.R[:] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg_info.P[:] = [msg_info.K[0], msg_info.K[1], msg_info.K[2], 0,
                     msg_info.K[3], msg_info.K[4], msg_info.K[5], 0,
                     0, 0, 1, 0,]
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
    ego_tf.child_frame_id = "base_link"
    ego_tf.translation.CopyFrom(get_translation(ego_pose))
    ego_tf.rotation.CopyFrom(get_rotation(ego_pose))
    return ego_tf


def get_sensor_tf(nusc, sensor_id, sample_data):
    sensor_tf = FrameTransform()
    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(sample_data["timestamp"])
    sensor_tf.child_frame_id = sensor_id
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
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

def write_scene_to_mcap(nusc, scene, filepath, basePreRes, preRes, caseInfos, use_case):

    scene_name = scene["name"]
    data_path = Path(nusc.dataroot)

    cur_sample = nusc.get("sample", scene["first_sample_token"])
    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene_name} Sample Data", leave=False)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as fp:
        print(f"Writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)

        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")

        while cur_sample is not None:
            pbar.update(1)
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            ego2GlobalRotQuat = ego_pose['rotation']
            ego2GlobalTran = ego_pose["translation"]
            ego2GlobalRotMat = Quaternion(ego2GlobalRotQuat).rotation_matrix  # used convert velo from global to ego
            stamp = get_time(ego_pose)

            ################### 发布自车系到世界系的tf变换 #################
            protobuf_writer.write_message("/tf", get_ego_tf(ego_pose), stamp.to_nsec())
            # iterate sensors
            for sensor_id, sample_token in cur_sample["data"].items():
                sample_data = nusc.get("sample_data", sample_token)

                # create sensor transform
                protobuf_writer.write_message("/tf", get_sensor_tf(nusc, sensor_id, sample_data), stamp.to_nsec())

                # write the sensor data
                if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message("/" + sensor_id, msg, stamp.to_nsec())

                elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message("/" + sensor_id, msg, stamp.to_nsec())

                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    protobuf_writer.write_message("/" + sensor_id + "/image_rect_compressed", msg, stamp.to_nsec())

                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    protobuf_writer.write_message("/" + sensor_id + "/camera_info", msg, stamp.to_nsec())

            # publish /pose
            pose_in_frame = PoseInFrame()
            pose_in_frame.timestamp.FromNanoseconds(stamp.to_nsec())
            pose_in_frame.frame_id = "base_link"
            pose_in_frame.pose.orientation.w = 1
            protobuf_writer.write_message("/pose", pose_in_frame, stamp.to_nsec())

            # 网络baseline版本
            basePreAnnSceneUpdate = SceneUpdate()              # cube
            basePreVeloEgoXSceneUpdate = SceneUpdate()     # 预测目标在自车系旋转下vx
            basePreVeloEgoYSceneUpdate = SceneUpdate()     # 预测目标在自车系旋转下vy
            basePreVeloGlobalXSceneUpdate = SceneUpdate()     # 预测目标在世界系旋转下vx
            basePreVeloGlobalYSceneUpdate = SceneUpdate()     # 预测目标在世界系旋转下vy
            basePreVeloArrowSceneUpdate = SceneUpdate()  # 预测目标速度箭头
            basePreHeadingSceneUpdate = SceneUpdate()    # 预测目标的heading
            basePreIdSceneUpdate = SceneUpdate()         # 预测目标的id

            # 网络迭代版本
            preAnnSceneUpdate = SceneUpdate()         
            preVeloEgoXSceneUpdate = SceneUpdate()
            preVeloEgoYSceneUpdate = SceneUpdate()
            preVeloGlobalXSceneUpdate = SceneUpdate()
            preVeloGlobalYSceneUpdate = SceneUpdate()
            preVeloArrowSceneUpdate = SceneUpdate() 
            preHeadingSceneUpdate = SceneUpdate() 
            preIdSceneUpdate = SceneUpdate()

            # 真值 
            gtAnnSceneUpdate = SceneUpdate()
            gtVeloEgoXSceneUpdate = SceneUpdate()
            gtVeloEgoYSceneUpdate = SceneUpdate()
            gtVeloGlobalXSceneUpdate = SceneUpdate()
            gtVeloGlobalYSceneUpdate = SceneUpdate()
            gtVeloArrowSceneUpdate = SceneUpdate() 
            gtHeadingSceneUpdate = SceneUpdate()
            gtIdSceneUpdate = SceneUpdate()

            ###################################### 网络baseline版本 ############################
            for basePreann in basePreRes[cur_sample['token']]:
                basePreAnnId = None
                if isinstance(basePreann["tracking_id"],int):
                    basePreAnnId = str(basePreann["tracking_id"])
                elif basePreann["tracking_id"][0] == 't':
                    basePreAnnId = basePreann["tracking_id"][7:-1]
                else:
                    basePreAnnId = basePreann["tracking_id"]

                ###################### 网络baseline版本预测框自车系/世界系位姿计算 #####################
                # conver ann from global to ego
                basePreAnnCenterGlobal = np.array([basePreann["translation"][0], basePreann["translation"][1], basePreann["translation"][2]])
                # conver ann_centerfrom global to ego
                basePreAnnCenterEgo = np.dot(np.linalg.inv(ego2GlobalRotMat), basePreAnnCenterGlobal - np.array(ego2GlobalTran))
                basePreAnnOrientationGlobal = np.array([basePreann["rotation"][0], basePreann["rotation"][1], basePreann["rotation"][2], basePreann["rotation"][3]])
                basePreQuaternionGlobal2Ego = Quaternion(matrix = np.linalg.inv(ego2GlobalRotMat))
                basePreAnnOrientationEgo = basePreQuaternionGlobal2Ego * basePreAnnOrientationGlobal

                ##################### 网络baseline版本预测框自车系/世界系横纵速度计算(范数不边) ##############
                basePreAnnVelo2dGlobal = basePreann["velocity"][:2]
                # convert velo from global to ego
                basePreAnnVelo3dGlobal = np.array([*basePreAnnVelo2dGlobal, 0.0])
                basePreAnnVelo3dEgo = basePreAnnVelo3dGlobal @ np.linalg.inv(ego2GlobalRotMat).T
                basePreAnnVelo2dEgo = basePreAnnVelo3dEgo[:2]

                ###################### 网络baseline版本预测框发布 ######################
                basePreAnnDeleteEntity = basePreAnnSceneUpdate.deletions.add()
                basePreAnnDeleteEntity.type = 1
                basePreAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)
                basePreAnnEntity = basePreAnnSceneUpdate.entities.add()
                basePreAnnEntity.frame_id = "base_link"
                basePreAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreAnnEntity.id = basePreAnnId
                basePreAnnEntity.frame_locked = True

                basePreAnnCube = basePreAnnEntity.cubes.add()
                basePreAnnCube.size.x = basePreann["size"][1]
                basePreAnnCube.size.y = basePreann["size"][0]
                basePreAnnCube.size.z = basePreann["size"][2]
                basePreAnnCube.color.r = 0
                basePreAnnCube.color.g = 0
                basePreAnnCube.color.b = 1
                basePreAnnCube.color.a = 0.3

                basePreAnnCube.pose.position.x = basePreAnnCenterEgo[0]
                basePreAnnCube.pose.position.y = basePreAnnCenterEgo[1]
                basePreAnnCube.pose.position.z = basePreAnnCenterEgo[2]
                basePreAnnCube.pose.orientation.w = basePreAnnOrientationEgo[0]
                basePreAnnCube.pose.orientation.x = basePreAnnOrientationEgo[1]
                basePreAnnCube.pose.orientation.y = basePreAnnOrientationEgo[2]
                basePreAnnCube.pose.orientation.z = basePreAnnOrientationEgo[3]
 
                ###################### 网络baseline版本预测框自车系纵向速度发布 ######################
                basePreVeloEgoXDeleteEntity = basePreVeloEgoXSceneUpdate.deletions.add()
                basePreVeloEgoXDeleteEntity.type = 1
                basePreVeloEgoXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                basePreVeloEgoXEntity = basePreVeloEgoXSceneUpdate.entities.add()
                basePreVeloEgoXEntity.frame_id = "base_link"
                basePreVeloEgoXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreVeloEgoXEntity.id = basePreAnnId
                basePreVeloEgoXEntity.frame_locked = True

                basePreVeloEgoXTexts = basePreVeloEgoXEntity.texts.add()
                basePreVeloEgoXTexts.pose.position.x = basePreAnnCenterEgo[0]
                basePreVeloEgoXTexts.pose.position.y = basePreAnnCenterEgo[1]
                basePreVeloEgoXTexts.pose.position.z = basePreAnnCenterEgo[2]
                basePreVeloEgoXTexts.pose.orientation.w = basePreAnnOrientationEgo[0]
                basePreVeloEgoXTexts.pose.orientation.x = basePreAnnOrientationEgo[1]
                basePreVeloEgoXTexts.pose.orientation.y = basePreAnnOrientationEgo[2]
                basePreVeloEgoXTexts.pose.orientation.z = basePreAnnOrientationEgo[3]
                basePreVeloEgoXTexts.font_size = 0.01
                basePreVeloEgoXTexts.text = str(basePreAnnVelo2dEgo[0])

                ###################### 网络baseline版本预测框自车系横向速度发布 ######################
                basePreVeloEgoYDeleteEntity = basePreVeloEgoYSceneUpdate.deletions.add()
                basePreVeloEgoYDeleteEntity.type = 1
                basePreVeloEgoYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                basePreVeloEgoYEntity = basePreVeloEgoYSceneUpdate.entities.add()
                basePreVeloEgoYEntity.frame_id = "base_link"
                basePreVeloEgoYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreVeloEgoYEntity.id = basePreAnnId
                basePreVeloEgoYEntity.frame_locked = True

                basePreVeloEgoYTexts = basePreVeloEgoYEntity.texts.add()
                basePreVeloEgoYTexts.pose.position.x = basePreAnnCenterEgo[0]
                basePreVeloEgoYTexts.pose.position.y = basePreAnnCenterEgo[1]
                basePreVeloEgoYTexts.pose.position.z = basePreAnnCenterEgo[2]
                basePreVeloEgoYTexts.pose.orientation.w = basePreAnnOrientationEgo[0]
                basePreVeloEgoYTexts.pose.orientation.x = basePreAnnOrientationEgo[1]
                basePreVeloEgoYTexts.pose.orientation.y = basePreAnnOrientationEgo[2]
                basePreVeloEgoYTexts.pose.orientation.z = basePreAnnOrientationEgo[3]
                basePreVeloEgoYTexts.font_size = 0.01
                basePreVeloEgoYTexts.text = str(basePreAnnVelo2dEgo[1])

                ###################### 网络baseline版本预测框世界系纵向速度发布 ######################
                basePreVeloGlobalXDeleteEntity = basePreVeloGlobalXSceneUpdate.deletions.add()
                basePreVeloGlobalXDeleteEntity.type = 1
                basePreVeloGlobalXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                basePreVeloGlobalXEntity = basePreVeloGlobalXSceneUpdate.entities.add()
                basePreVeloGlobalXEntity.frame_id = "map"
                basePreVeloGlobalXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreVeloGlobalXEntity.id = basePreAnnId
                basePreVeloGlobalXEntity.frame_locked = True

                basePreVeloGlobalXTexts = basePreVeloGlobalXEntity.texts.add()
                basePreVeloGlobalXTexts.pose.position.x = basePreAnnCenterGlobal[0]
                basePreVeloGlobalXTexts.pose.position.y = basePreAnnCenterGlobal[1]
                basePreVeloGlobalXTexts.pose.position.z = basePreAnnCenterGlobal[2]
                basePreVeloGlobalXTexts.pose.orientation.w = basePreAnnOrientationGlobal[0]
                basePreVeloGlobalXTexts.pose.orientation.x = basePreAnnOrientationGlobal[1]
                basePreVeloGlobalXTexts.pose.orientation.y = basePreAnnOrientationGlobal[2]
                basePreVeloGlobalXTexts.pose.orientation.z = basePreAnnOrientationGlobal[3]
                basePreVeloGlobalXTexts.font_size = 0.01
                basePreVeloGlobalXTexts.text = str(basePreAnnVelo2dGlobal[0])


                ###################### 网络baseline版本预测框世界系横向速度发布 ######################
                basePreVeloGlobalYDeleteEntity = basePreVeloGlobalYSceneUpdate.deletions.add()
                basePreVeloGlobalYDeleteEntity.type = 1
                basePreVeloGlobalYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                basePreVeloGlobalYEntity = basePreVeloGlobalYSceneUpdate.entities.add()
                basePreVeloGlobalYEntity.frame_id = "map"
                basePreVeloGlobalYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreVeloGlobalYEntity.id = basePreAnnId
                basePreVeloGlobalYEntity.frame_locked = True

                basePreVeloGlobalYTexts = basePreVeloGlobalYEntity.texts.add()
                basePreVeloGlobalYTexts.pose.position.x = basePreAnnCenterGlobal[0]
                basePreVeloGlobalYTexts.pose.position.y = basePreAnnCenterGlobal[1]
                basePreVeloGlobalYTexts.pose.position.z = basePreAnnCenterGlobal[2]
                basePreVeloGlobalYTexts.pose.orientation.w = basePreAnnOrientationGlobal[0]
                basePreVeloGlobalYTexts.pose.orientation.x = basePreAnnOrientationGlobal[1]
                basePreVeloGlobalYTexts.pose.orientation.y = basePreAnnOrientationGlobal[2]
                basePreVeloGlobalYTexts.pose.orientation.z = basePreAnnOrientationGlobal[3]
                basePreVeloGlobalYTexts.font_size = 0.01
                basePreVeloGlobalYTexts.text = str(basePreAnnVelo2dGlobal[1])

                ###################### 网络baseline版本预测框速度箭头发布, 用于观察车辆的走向 ######################
                basePreVeloArrowDeletEntity = basePreVeloArrowSceneUpdate.deletions.add()
                basePreVeloArrowDeletEntity.type = 1
                basePreVeloArrowDeletEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                basePreVeloArrowEntity = basePreVeloArrowSceneUpdate.entities.add()
                basePreVeloArrowEntity.id = basePreAnnId
                basePreVeloArrowEntity.frame_id = "base_link"
                basePreVeloArrowEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreVeloArrowEntity.frame_locked = True
                basePreVeloArrowLine = basePreVeloArrowEntity.lines.add()
                basePreVeloArrowLine.type = 0
                basePreVeloArrowLine.thickness = 0.1
                basePreVeloArrowLine.color.r = 218 / 255.0
                basePreVeloArrowLine.color.g = 112 / 255.0
                basePreVeloArrowLine.color.b = 214 / 255.0
                basePreVeloArrowLine.color.a = 1.0
                basePreVeloArrowLine.points.add(x = basePreAnnCenterEgo[0], y = basePreAnnCenterEgo[1], z = basePreAnnCenterEgo[2])                
                basePreAnnL2 = np.sqrt(basePreAnnVelo2dEgo[0]**2 + basePreAnnVelo2dEgo[1]**2)
                if basePreAnnL2 >= 1: 
                    basePreVeloArrowLine.points.add(x = basePreAnnCenterEgo[0] + basePreAnnVelo2dEgo[0]/basePreAnnL2 * 2, y = basePreAnnCenterEgo[1] + basePreAnnVelo2dEgo[1]/basePreAnnL2 * 2, z = basePreAnnCenterEgo[2]) 
                else:
                    basePreVeloArrowLine.points.add(x = basePreAnnCenterEgo[0] + basePreAnnVelo2dEgo[0], y = basePreAnnCenterEgo[1] + basePreAnnVelo2dEgo[1], z = basePreAnnCenterEgo[2])        


                ###################### 网络baseline版本预测框车身heading发布, 用于观察车辆的走向 ######################
                basePreHeadingDeleteEntity = basePreHeadingSceneUpdate.deletions.add()
                basePreHeadingDeleteEntity.type = 1
                basePreHeadingDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                basePreHeadingEntity = basePreHeadingSceneUpdate.entities.add()
                basePreHeadingEntity.id = basePreAnnId
                basePreHeadingEntity.frame_id = "base_link"
                basePreHeadingEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreHeadingEntity.frame_locked = True

                basePreHeadingLine = basePreHeadingEntity.arrows.add()
                basePreHeadingLine.color.r = 238 / 255.0
                basePreHeadingLine.color.g = 121 / 255.0
                basePreHeadingLine.color.b = 66 / 255.0
                basePreHeadingLine.color.a = 0.4
                basePreHeadingLine.pose.position.x = basePreAnnCenterEgo[0]
                basePreHeadingLine.pose.position.y = basePreAnnCenterEgo[1]
                basePreHeadingLine.pose.position.z = basePreAnnCenterEgo[2] + basePreann["size"][2]/2
                basePreHeadingLine.pose.orientation.w = basePreAnnOrientationEgo[0]
                basePreHeadingLine.pose.orientation.x = basePreAnnOrientationEgo[1]
                basePreHeadingLine.pose.orientation.y = basePreAnnOrientationEgo[2]
                basePreHeadingLine.pose.orientation.z = basePreAnnOrientationEgo[3]
                basePreHeadingLine.shaft_length = basePreann["size"][1]/2
                basePreHeadingLine.shaft_diameter = 0.1
                basePreHeadingLine.head_length = 0.01
                basePreHeadingLine.head_diameter = 0.01

                ################# 发布网络baseline结果的预测目标ID信息  #############
                basePreIdDeleteEntity = basePreIdSceneUpdate.deletions.add()
                basePreIdDeleteEntity.type = 1
                basePreIdDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                basePreIdEntity = basePreIdSceneUpdate.entities.add()
                basePreIdEntity.frame_id = "base_link"
                basePreIdEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                basePreIdEntity.id = basePreAnnId
                basePreIdEntity.frame_locked = True
                
                basePreIdTexts = basePreIdEntity.texts.add()
                basePreIdTexts.pose.position.x = basePreAnnCenterEgo[0]
                basePreIdTexts.pose.position.y = basePreAnnCenterEgo[1]
                basePreIdTexts.pose.position.z = basePreAnnCenterEgo[2]
                basePreIdTexts.pose.orientation.w = basePreAnnOrientationEgo[0]
                basePreIdTexts.pose.orientation.x = basePreAnnOrientationEgo[1]
                basePreIdTexts.pose.orientation.y = basePreAnnOrientationEgo[2]
                basePreIdTexts.pose.orientation.z = basePreAnnOrientationEgo[3]
                basePreIdTexts.font_size = 0.5
                basePreIdTexts.text = basePreAnnId

            protobuf_writer.write_message("/markers/basePreAnns", basePreAnnSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/basePreVeloEgoX", basePreVeloEgoXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/basePreVeloEgoY", basePreVeloEgoYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/basePreVeloGlobalX", basePreVeloGlobalXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/basePreVeloGlobalY", basePreVeloGlobalYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/basePreArrow", basePreVeloArrowSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/basePreHeading", basePreHeadingSceneUpdate, stamp.to_nsec())     
            protobuf_writer.write_message("/markers/basePreId", basePreIdSceneUpdate, stamp.to_nsec())    

            #################### 迭代网络模型结果 ###############################
            for preAnn in preRes[cur_sample['token']]:
                preAnnId = None
                if isinstance(preAnn["tracking_id"],int):
                    preAnnId = str(preAnn["tracking_id"])
                elif preAnn["tracking_id"][0] == 't':
                    preAnnId = preAnn["tracking_id"][7:-1]
                else:
                    preAnnId = preAnn["tracking_id"]

                #################### 计算自车系/世界系下的速度横纵值(只是方向, 没有基于自车速度做速度补偿，#####################
                #################### 还是世界系下的绝对速度，然后基于自车的朝向做了分解) ####################
                preAnnVelo2dGlobal = preAnn["velocity"][:2]
                # convert velo from global to ego
                preAnnVelo3dGlobal = np.array([*preAnnVelo2dGlobal, 0.0])
                preAnnVelo3dEgo = preAnnVelo3dGlobal @ np.linalg.inv(ego2GlobalRotMat).T
                preAnnVelo2dEgo = preAnnVelo3dEgo[:2]

                ################## 计算自车系/世界系下的预测框位置和方位角 ################
                # conver ann from global to ego
                preAnnCenterGlobal = np.array([preAnn["translation"][0], preAnn["translation"][1], preAnn["translation"][2]])
                # conver ann_centerfrom global to ego
                preAnnCenterEgo = np.dot(np.linalg.inv(ego2GlobalRotMat), preAnnCenterGlobal - np.array(ego2GlobalTran))
                preAnnOrientationGlobal = np.array([preAnn["rotation"][0], preAnn["rotation"][1], preAnn["rotation"][2], preAnn["rotation"][3]])
                preQuatGlobal2Ego = Quaternion(matrix = np.linalg.inv(ego2GlobalRotMat))
                preAnnOrientationEgo = preQuatGlobal2Ego * preAnnOrientationGlobal

                #################### 发布迭代模型预测框位姿信息 ######################
                preAnnDeleteEntity = preAnnSceneUpdate.deletions.add()
                preAnnDeleteEntity.type = 1
                preAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preAnnEntity = preAnnSceneUpdate.entities.add()
                preAnnEntity.frame_id = "base_link"
                preAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnEntity.id = preAnnId
                preAnnEntity.frame_locked = True

                preAnnCube = preAnnEntity.cubes.add()
                preAnnCube.size.x = preAnn["size"][1]
                preAnnCube.size.y = preAnn["size"][0]
                preAnnCube.size.z = preAnn["size"][2]
                preAnnCube.color.r = 1
                preAnnCube.color.g = 1
                preAnnCube.color.b = 0
                preAnnCube.color.a = 0.3

                preAnnCube.pose.position.x = preAnnCenterEgo[0]
                preAnnCube.pose.position.y = preAnnCenterEgo[1]
                preAnnCube.pose.position.z = preAnnCenterEgo[2]
                preAnnCube.pose.orientation.w = preAnnOrientationEgo[0]
                preAnnCube.pose.orientation.x = preAnnOrientationEgo[1]
                preAnnCube.pose.orientation.y = preAnnOrientationEgo[2]
                preAnnCube.pose.orientation.z = preAnnOrientationEgo[3]
 
                ###################### 发布迭代网络预测框自车系下纵向速度信息 ###################
                preVeloEgoXDeleteEntity = preVeloEgoXSceneUpdate.deletions.add()
                preVeloEgoXDeleteEntity.type = 1
                preVeloEgoXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                preVeloEgoXEntity = preVeloEgoXSceneUpdate.entities.add()
                preVeloEgoXEntity.frame_id = "base_link"
                preVeloEgoXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preVeloEgoXEntity.id = preAnnId
                preVeloEgoXEntity.frame_locked = True

                preVeloEgoXTexts = preVeloEgoXEntity.texts.add()
                preVeloEgoXTexts.pose.position.x = preAnnCenterEgo[0]
                preVeloEgoXTexts.pose.position.y = preAnnCenterEgo[1]
                preVeloEgoXTexts.pose.position.z = preAnnCenterEgo[2]
                preVeloEgoXTexts.pose.orientation.w = preAnnOrientationEgo[0]
                preVeloEgoXTexts.pose.orientation.x = preAnnOrientationEgo[1]
                preVeloEgoXTexts.pose.orientation.y = preAnnOrientationEgo[2]
                preVeloEgoXTexts.pose.orientation.z = preAnnOrientationEgo[3]
                preVeloEgoXTexts.font_size = 0.01
                preVeloEgoXTexts.text = str(preAnnVelo2dEgo[0])

                ###################### 发布迭代网络预测框自车系下横向速度信息 ###################
                preVeloEgoYDeleteEntity = preVeloEgoYSceneUpdate.deletions.add()
                preVeloEgoYDeleteEntity.type = 1
                preVeloEgoYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                preVeloEgoYEntity = preVeloEgoYSceneUpdate.entities.add()
                preVeloEgoYEntity.frame_id = "base_link"
                preVeloEgoYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preVeloEgoYEntity.id = preAnnId
                preVeloEgoYEntity.frame_locked = True
                preVeloEgoYTexts = preVeloEgoYEntity.texts.add()

                preVeloEgoYTexts.pose.position.x = preAnnCenterEgo[0]
                preVeloEgoYTexts.pose.position.y = preAnnCenterEgo[1]
                preVeloEgoYTexts.pose.position.z = preAnnCenterEgo[2]
                preVeloEgoYTexts.pose.orientation.w = preAnnOrientationEgo[0]
                preVeloEgoYTexts.pose.orientation.x = preAnnOrientationEgo[1]
                preVeloEgoYTexts.pose.orientation.y = preAnnOrientationEgo[2]
                preVeloEgoYTexts.pose.orientation.z = preAnnOrientationEgo[3]
                preVeloEgoYTexts.font_size = 0.01
                preVeloEgoYTexts.text = str(preAnnVelo2dEgo[1])

                ###################### 发布迭代网络预测框世界系下纵向速度信息 ###################
                preVeloGlobalXDeleteEntity = preVeloGlobalXSceneUpdate.deletions.add()
                preVeloGlobalXDeleteEntity.type = 1
                preVeloGlobalXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                preVeloGlobalXEntity = preVeloGlobalXSceneUpdate.entities.add()
                preVeloGlobalXEntity.frame_id = "map"
                preVeloGlobalXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preVeloGlobalXEntity.id = preAnnId
                preVeloGlobalXEntity.frame_locked = True

                preVeloGlobalXTexts = preVeloGlobalXEntity.texts.add()
                preVeloGlobalXTexts.pose.position.x = preAnnCenterGlobal[0]
                preVeloGlobalXTexts.pose.position.y = preAnnCenterGlobal[1]
                preVeloGlobalXTexts.pose.position.z = preAnnCenterGlobal[2]
                preVeloGlobalXTexts.pose.orientation.w = preAnnOrientationGlobal[0]
                preVeloGlobalXTexts.pose.orientation.x = preAnnOrientationGlobal[1]
                preVeloGlobalXTexts.pose.orientation.y = preAnnOrientationGlobal[2]
                preVeloGlobalXTexts.pose.orientation.z = preAnnOrientationGlobal[3]
                preVeloGlobalXTexts.font_size = 0.01
                preVeloGlobalXTexts.text = str(preAnnVelo2dGlobal[0])

                ###################### 发布迭代网络预测框世界系下横向速度信息 ###################
                preVeloGlobalYDeleteEntity = preVeloGlobalYSceneUpdate.deletions.add()
                preVeloGlobalYDeleteEntity.type = 1
                preVeloGlobalYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                preVeloGlobalYEntity = preVeloGlobalYSceneUpdate.entities.add()
                preVeloGlobalYEntity.frame_id = "map"
                preVeloGlobalYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preVeloGlobalYEntity.id = preAnnId
                preVeloGlobalYEntity.frame_locked = True

                preVeloGlobalYTexts = preVeloGlobalYEntity.texts.add()
                preVeloGlobalYTexts.pose.position.x = preAnnCenterGlobal[0]
                preVeloGlobalYTexts.pose.position.y = preAnnCenterGlobal[1]
                preVeloGlobalYTexts.pose.position.z = preAnnCenterGlobal[2]
                preVeloGlobalYTexts.pose.orientation.w = preAnnOrientationGlobal[0]
                preVeloGlobalYTexts.pose.orientation.x = preAnnOrientationGlobal[1]
                preVeloGlobalYTexts.pose.orientation.y = preAnnOrientationGlobal[2]
                preVeloGlobalYTexts.pose.orientation.z = preAnnOrientationGlobal[3]
                preVeloGlobalYTexts.font_size = 0.01
                preVeloGlobalYTexts.text = str(preAnnVelo2dGlobal[1])
                
                ###################### 发布迭代网络预测框速度箭头信息(可以跟随foxglove选定的基坐标系完成转化) ###################
                preVeloArrowDeleteEntity = preVeloArrowSceneUpdate.deletions.add()
                preVeloArrowDeleteEntity.type = 1
                preVeloArrowDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preVeloArrowEntity = preVeloArrowSceneUpdate.entities.add()
                preVeloArrowEntity.id = preAnnId
                preVeloArrowEntity.frame_id = "base_link"
                preVeloArrowEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preVeloArrowEntity.frame_locked = True

                preVeloArrowLine = preVeloArrowEntity.lines.add()
                preVeloArrowLine.type = 0
                preVeloArrowLine.thickness = 0.1
                preVeloArrowLine.color.r = 0
                preVeloArrowLine.color.g = 0
                preVeloArrowLine.color.b = 0
                preVeloArrowLine.color.a = 0.8
                preVeloArrowLine.points.add(x = preAnnCenterEgo[0], y = preAnnCenterEgo[1], z = preAnnCenterEgo[2]) 
                preVeloL2 = np.sqrt(preAnnVelo2dEgo[0]**2 + preAnnVelo2dEgo[1]**2)
                if preVeloL2 >= 1: 
                    preVeloArrowLine.points.add(x = preAnnCenterEgo[0] + preAnnVelo2dEgo[0]/preVeloL2 * 2, y = preAnnCenterEgo[1] + preAnnVelo2dEgo[1]/preVeloL2 * 2, z = preAnnCenterEgo[2]) 
                else:
                    preVeloArrowLine.points.add(x = preAnnCenterEgo[0] + preAnnVelo2dEgo[0], y = preAnnCenterEgo[1] + preAnnVelo2dEgo[1], z = preAnnCenterEgo[2])        

                ###################### 网络迭代版本预测框车身heading发布, 用于观察车辆的走向 ######################
                preHeadingDeleteEntity = preHeadingSceneUpdate.deletions.add()
                preHeadingDeleteEntity.type = 1
                preHeadingDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preHeadingEntity = preHeadingSceneUpdate.entities.add()
                preHeadingEntity.id = preAnnId
                preHeadingEntity.frame_id = "base_link"
                preHeadingEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preHeadingEntity.frame_locked = True

                preHeadingLine = preHeadingEntity.arrows.add()
                preHeadingLine.color.r = 238 / 255.0
                preHeadingLine.color.g = 223 / 255.0
                preHeadingLine.color.b = 204 / 255.0
                preHeadingLine.color.a = 0.4
                preHeadingLine.pose.position.x = preAnnCenterEgo[0]
                preHeadingLine.pose.position.y = preAnnCenterEgo[1]
                preHeadingLine.pose.position.z = preAnnCenterEgo[2] + preAnn["size"][2]/2
                preHeadingLine.pose.orientation.w = preAnnOrientationEgo[0]
                preHeadingLine.pose.orientation.x = preAnnOrientationEgo[1]
                preHeadingLine.pose.orientation.y = preAnnOrientationEgo[2]
                preHeadingLine.pose.orientation.z = preAnnOrientationEgo[3]
                preHeadingLine.shaft_length = preAnn["size"][1]/2
                preHeadingLine.shaft_diameter = 0.1
                preHeadingLine.head_length = 0.01
                preHeadingLine.head_diameter = 0.01

                ################# 发布网络迭代结果的预测目标ID信息  #############
                preIdDeleteEntity = preIdSceneUpdate.deletions.add()
                preIdDeleteEntity.type = 1
                preIdDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preIdEntity = preIdSceneUpdate.entities.add()
                preIdEntity.frame_id = "base_link"
                preIdEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preIdEntity.id = preAnnId
                preIdEntity.frame_locked = True
                
                preIdTexts = preIdEntity.texts.add()
                preIdTexts.pose.position.x = preAnnCenterEgo[0]
                preIdTexts.pose.position.y = preAnnCenterEgo[1]
                preIdTexts.pose.position.z = preAnnCenterEgo[2]
                preIdTexts.pose.orientation.w = preAnnOrientationEgo[0]
                preIdTexts.pose.orientation.x = preAnnOrientationEgo[1]
                preIdTexts.pose.orientation.y = preAnnOrientationEgo[2]
                preIdTexts.pose.orientation.z = preAnnOrientationEgo[3]
                preIdTexts.font_size = 0.5
                preIdTexts.text = preAnnId

            protobuf_writer.write_message("/markers/preAnns", preAnnSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preVeloEgoX", preVeloEgoXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preVeloEgoY", preVeloEgoYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/preVeloGlobalX", preVeloGlobalXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preVeloGlobalY", preVeloGlobalYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/preArrow", preVeloArrowSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/preHeading", preHeadingSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/preId", preIdSceneUpdate, stamp.to_nsec())                

            ######################### GT真值信息 ##################################
            for gtAnnToken in cur_sample["anns"]:

                gtAnn = nusc.get("sample_annotation", gtAnnToken)
                gtId = gtAnn["instance_token"][:4]

                ################# 计算真值框在世界系/自车系下速度值(范数不变, 仅仅是自车系下的绝对速度的横纵分解) ##################
                # gt_anns global velo 
                gtAnnVelo2dGlobal = nusc.box_velocity(gtAnnToken)[:2]                  # narray
                # convert velo from global to ego
                gtAnnVelo3dGlobal = np.array([*gtAnnVelo2dGlobal, 0.0])
                gtAnnVelo3dEgo = gtAnnVelo3dGlobal @ np.linalg.inv(ego2GlobalRotMat).T
                gtAnnVelo2dEgo = gtAnnVelo3dEgo[:2]

                ################ 计算真值框世界系/自车系下的位姿 ####################
                # conver box from global to ego
                gtAnnCenterGlobal = np.array([gtAnn["translation"][0], gtAnn["translation"][1], gtAnn["translation"][2]])
                # conver ann_centerfrom global to ego
                gtAnnCenterEgo = np.dot(np.linalg.inv(ego2GlobalRotMat), gtAnnCenterGlobal - np.array(ego2GlobalTran))
                gtAnnOrientationGlobal = np.array([gtAnn["rotation"][0], gtAnn["rotation"][1], gtAnn["rotation"][2], gtAnn["rotation"][3]])
                gtQuatGlobal2Ego = Quaternion(matrix = np.linalg.inv(ego2GlobalRotMat))
                gtAnnOrientationEgo = gtQuatGlobal2Ego * gtAnnOrientationGlobal

                ################# 发布真值框位置信息 #############################
                gtAnnDeleteEntity = gtAnnSceneUpdate.deletions.add()
                gtAnnDeleteEntity.type = 1
                gtAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)    

                gtAnnEntity = gtAnnSceneUpdate.entities.add()
                gtAnnEntity.frame_id = "base_link"
                gtAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnEntity.id = gtId   # 为了拉曲线输入id比较方便              
                gtAnnEntity.frame_locked = True

                gtAnnMetadata = gtAnnEntity.metadata.add()
                gtAnnMetadata.key = "category"
                gtAnnMetadata.value = gtAnn["category_name"]

                gtAnnCube = gtAnnEntity.cubes.add()
                gtAnnCube.size.x = gtAnn["size"][1]
                gtAnnCube.size.y = gtAnn["size"][0]
                gtAnnCube.size.z = gtAnn["size"][2]
                if use_case and gtId in caseInfos[scene_name]:  # 可以高亮要观察的case目标, 比较清晰的观察到
                    gtAnnCube.color.r = 1
                    gtAnnCube.color.g = 0
                    gtAnnCube.color.b = 0
                    gtAnnCube.color.a = 0.3
                else:
                    gtAnnCube.color.r = 0
                    gtAnnCube.color.g = 1
                    gtAnnCube.color.b = 0
                    gtAnnCube.color.a = 0.3 

                gtAnnCube.pose.position.x = gtAnnCenterEgo[0]
                gtAnnCube.pose.position.y = gtAnnCenterEgo[1]
                gtAnnCube.pose.position.z = gtAnnCenterEgo[2]
                gtAnnCube.pose.orientation.w = gtAnnOrientationEgo[0]
                gtAnnCube.pose.orientation.x = gtAnnOrientationEgo[1]
                gtAnnCube.pose.orientation.y = gtAnnOrientationEgo[2]
                gtAnnCube.pose.orientation.z = gtAnnOrientationEgo[3]

                ################# 发布真值框自车系下纵向速度 #############################
                gtVeloEgoXDeleteEntity = gtVeloEgoXSceneUpdate.deletions.add()
                gtVeloEgoXDeleteEntity.type = 1
                gtVeloEgoXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtVeloEgoXEntity = gtVeloEgoXSceneUpdate.entities.add()
                gtVeloEgoXEntity.frame_id = "base_link"
                gtVeloEgoXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtVeloEgoXEntity.id = gtId
                gtVeloEgoXEntity.frame_locked = True

                gtVeloEgoXTexts = gtVeloEgoXEntity.texts.add()
                gtVeloEgoXTexts.pose.position.x = gtAnnCenterEgo[0]
                gtVeloEgoXTexts.pose.position.y = gtAnnCenterEgo[1]
                gtVeloEgoXTexts.pose.position.z = gtAnnCenterEgo[2]
                gtVeloEgoXTexts.pose.orientation.w = gtAnnOrientationEgo[0]
                gtVeloEgoXTexts.pose.orientation.x = gtAnnOrientationEgo[1]
                gtVeloEgoXTexts.pose.orientation.y = gtAnnOrientationEgo[2]
                gtVeloEgoXTexts.pose.orientation.z = gtAnnOrientationEgo[3]
                gtVeloEgoXTexts.font_size = 0.01
                gtVeloEgoXTexts.text = str(gtAnnVelo2dEgo[0])

                ################# 发布真值框自车系下横向速度 #############################
                gtVeloEgoYDeleteEntity = gtVeloEgoYSceneUpdate.deletions.add()
                gtVeloEgoYDeleteEntity.type = 1
                gtVeloEgoYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtVeloEgoYEntity = gtVeloEgoYSceneUpdate.entities.add()
                gtVeloEgoYEntity.frame_id = "base_link"
                gtVeloEgoYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtVeloEgoYEntity.id = gtId
                gtVeloEgoYEntity.frame_locked = True

                gtVeloEgoYTexts = gtVeloEgoYEntity.texts.add()
                gtVeloEgoYTexts.pose.position.x = gtAnnCenterEgo[0]
                gtVeloEgoYTexts.pose.position.y = gtAnnCenterEgo[1]
                gtVeloEgoYTexts.pose.position.z = gtAnnCenterEgo[2]
                gtVeloEgoYTexts.pose.orientation.w = gtAnnOrientationEgo[0]
                gtVeloEgoYTexts.pose.orientation.x = gtAnnOrientationEgo[1]
                gtVeloEgoYTexts.pose.orientation.y = gtAnnOrientationEgo[2]
                gtVeloEgoYTexts.pose.orientation.z = gtAnnOrientationEgo[3]
                gtVeloEgoYTexts.font_size = 0.01
                gtVeloEgoYTexts.text = str(gtAnnVelo2dEgo[1])

                ################# 发布真值框世界系下纵向速度 #############################
                gtVeloGlobalXDeleteEntity = gtVeloGlobalXSceneUpdate.deletions.add()
                gtVeloGlobalXDeleteEntity.type = 1
                gtVeloGlobalXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtVeloGlobalXEntity = gtVeloGlobalXSceneUpdate.entities.add()
                gtVeloGlobalXEntity.frame_id = "map"
                gtVeloGlobalXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtVeloGlobalXEntity.id = gtId
                gtVeloGlobalXEntity.frame_locked = True

                gtVeloGlobalXTexts = gtVeloGlobalXEntity.texts.add()
                gtVeloGlobalXTexts.pose.position.x = gtAnnCenterGlobal[0]
                gtVeloGlobalXTexts.pose.position.y = gtAnnCenterGlobal[1]
                gtVeloGlobalXTexts.pose.position.z = gtAnnCenterGlobal[2]
                gtVeloGlobalXTexts.pose.orientation.w = gtAnnOrientationGlobal[0]
                gtVeloGlobalXTexts.pose.orientation.x = gtAnnOrientationGlobal[1]
                gtVeloGlobalXTexts.pose.orientation.y = gtAnnOrientationGlobal[2]
                gtVeloGlobalXTexts.pose.orientation.z = gtAnnOrientationGlobal[3]
                gtVeloGlobalXTexts.font_size = 0.01
                gtVeloGlobalXTexts.text = str(gtAnnVelo2dGlobal[0])

                ################# 发布真值框世界系下横向速度 #############################
                gtVeloGlobalYDeleteEntity = gtVeloGlobalYSceneUpdate.deletions.add()
                gtVeloGlobalYDeleteEntity.type = 1
                gtVeloGlobalYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtVeloGlobalYEntity = gtVeloGlobalYSceneUpdate.entities.add()
                gtVeloGlobalYEntity.frame_id = "map"
                gtVeloGlobalYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtVeloGlobalYEntity.id = gtId
                gtVeloGlobalYEntity.frame_locked = True

                gtVeloGlobalYTexts = gtVeloGlobalYEntity.texts.add()
                gtVeloGlobalYTexts.pose.position.x = gtAnnCenterGlobal[0]
                gtVeloGlobalYTexts.pose.position.y = gtAnnCenterGlobal[1]
                gtVeloGlobalYTexts.pose.position.z = gtAnnCenterGlobal[2]
                gtVeloGlobalYTexts.pose.orientation.w = gtAnnOrientationGlobal[0]
                gtVeloGlobalYTexts.pose.orientation.x = gtAnnOrientationGlobal[1]
                gtVeloGlobalYTexts.pose.orientation.y = gtAnnOrientationGlobal[2]
                gtVeloGlobalYTexts.pose.orientation.z = gtAnnOrientationGlobal[3]
                gtVeloGlobalYTexts.font_size = 0.01
                gtVeloGlobalYTexts.text = str(gtAnnVelo2dGlobal[1])

                ###################### 发布真值框速度箭头信息(可以跟随foxglove选定的基坐标系完成转化) ###################
                gtVeloArrowDeleteEntity = gtVeloArrowSceneUpdate.deletions.add()
                gtVeloArrowDeleteEntity.type = 1
                gtVeloArrowDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                gtVeloArrowEntity = gtVeloArrowSceneUpdate.entities.add()
                gtVeloArrowEntity.id = gtId
                gtVeloArrowEntity.frame_id = "base_link"
                gtVeloArrowEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtVeloArrowEntity.frame_locked = True

                gtVeloArrowLine = gtVeloArrowEntity.lines.add()
                gtVeloArrowLine.type = 0
                gtVeloArrowLine.thickness = 0.1
                gtVeloArrowLine.color.r = 0
                gtVeloArrowLine.color.g = 0
                gtVeloArrowLine.color.b = 0
                gtVeloArrowLine.color.a = 0.8
                gtVeloArrowLine.points.add(x = gtAnnCenterEgo[0], y = gtAnnCenterEgo[1], z = gtAnnCenterEgo[2]) 
                gtVeloL2 = np.sqrt(gtAnnVelo2dEgo[0]**2 + gtAnnVelo2dEgo[1]**2)
                if gtVeloL2 >= 1: 
                    gtVeloArrowLine.points.add(x = gtAnnCenterEgo[0] + gtAnnVelo2dEgo[0]/gtVeloL2 * 2, y = gtAnnCenterEgo[1] + gtAnnVelo2dEgo[1]/gtVeloL2 * 2, z = gtAnnCenterEgo[2]) 
                else:
                    gtVeloArrowLine.points.add(x = gtAnnCenterEgo[0] + gtAnnVelo2dEgo[0], y = gtAnnCenterEgo[1] + gtAnnVelo2dEgo[1], z = gtAnnCenterEgo[2]) 
                
                ################# 发布真值框框heading信息 #############################
                gtHeadingDeleteEntity = gtHeadingSceneUpdate.deletions.add()
                gtHeadingDeleteEntity.type = 1
                gtHeadingDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                gtHeadingEntity = gtHeadingSceneUpdate.entities.add()
                gtHeadingEntity.id = gtId
                gtHeadingEntity.frame_id = "map"
                gtHeadingEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtHeadingEntity.frame_locked = True
                gtHeadingLine = gtHeadingEntity.arrows.add()
                  
                gtHeadingLine.color.r = 255 / 255.0
                gtHeadingLine.color.g = 211 / 255.0
                gtHeadingLine.color.b = 155 / 255.0
                gtHeadingLine.color.a = 0.5
                gtHeadingLine.pose.position.x = gtAnnCenterGlobal[0]
                gtHeadingLine.pose.position.y = gtAnnCenterGlobal[1]
                gtHeadingLine.pose.position.z = gtAnnCenterGlobal[2] + gtAnn["size"][2]/2
                gtHeadingLine.pose.orientation.w = gtAnnOrientationGlobal[0]
                gtHeadingLine.pose.orientation.x = gtAnnOrientationGlobal[1]
                gtHeadingLine.pose.orientation.y = gtAnnOrientationGlobal[2]
                gtHeadingLine.pose.orientation.z = gtAnnOrientationGlobal[3]
                gtHeadingLine.shaft_length = gtAnn["size"][1]/2
                gtHeadingLine.shaft_diameter = 0.07
                gtHeadingLine.head_length = 0.01
                gtHeadingLine.head_diameter = 0.01

                ################# 发布真值框目标ID信息  #############
                gtIdDeleteEntity = gtIdSceneUpdate.deletions.add()
                gtIdDeleteEntity.type = 1
                gtIdDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                gtIdEntity = gtIdSceneUpdate.entities.add()
                gtIdEntity.frame_id = "base_link"
                gtIdEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtIdEntity.id = gtId
                gtIdEntity.frame_locked = True
                
                gtIdTexts = gtIdEntity.texts.add()
                gtIdTexts.pose.position.x = gtAnnCenterEgo[0]
                gtIdTexts.pose.position.y = gtAnnCenterEgo[1]
                gtIdTexts.pose.position.z = gtAnnCenterEgo[2]
                gtIdTexts.pose.orientation.w = gtAnnOrientationEgo[0]
                gtIdTexts.pose.orientation.x = gtAnnOrientationEgo[1]
                gtIdTexts.pose.orientation.y = gtAnnOrientationEgo[2]
                gtIdTexts.pose.orientation.z = gtAnnOrientationEgo[3]
                gtIdTexts.font_size = 0.5
                gtIdTexts.text = gtId

            protobuf_writer.write_message("/markers/gtHeading", gtHeadingSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/gtVeloEgoX", gtVeloEgoXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gtVeloEgoY", gtVeloEgoYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/gtVeloGlobalX", gtVeloGlobalXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gtVeloGlobalY", gtVeloGlobalYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/gtArrow", gtVeloArrowSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/gtAnns", gtAnnSceneUpdate, stamp.to_nsec())              
            protobuf_writer.write_message("/markers/gtId", gtIdSceneUpdate, stamp.to_nsec())      

            # publish /markers/car
            protobuf_writer.write_message("/markers/car", get_car_scene_update(stamp.to_nsec()), stamp.to_nsec())

            # move to the next sample
            cur_sample = (
                nusc.get("sample", cur_sample["next"])
                if cur_sample.get("next") != ""
                else None
            )

        pbar.close()
        writer.finish()
        print(f"Finished writing {filepath}")


def convert_all(
    output_dir: Path,
    name: str,
    nusc: NuScenes,
    basePreJson,
    preJson,
    caseJsonFile,
    use_case = False,
):
    with open(basePreJson,'r') as J:
        basePreResInfos = json.load(J)
    basePreRes = basePreResInfos['results']

    with open(preJson) as J:
        preResInfos = json.load(J)
        preRes = preResInfos['results']   

    with open(caseJsonFile,"r") as fp:
        caseInfos = json.load(fp)

    splits = create_splits_scenes()

    for scene in nusc.scene:
        if use_case:
            if scene['name'] not in splits['val'] or scene["name"] not in caseInfos.keys():
                continue
        else:
            if scene['name'] not in splits['val']:
                continue
        scene_name = scene["name"]
        mcap_name = f"NuScenes-{name}-{scene_name}.mcap"
        write_scene_to_mcap(nusc, scene, output_dir / mcap_name, basePreRes, preRes, caseInfos, use_case)

def main():
    root_injson = "./input_json"
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_json_name", "-p", default= "tracking_result_base.json", help="path to infer result")
    parser.add_argument("--result_json_name2", default= "tracking_result_update.json", help="path to infer result")
    
    parser.add_argument("--case_json_name", type=str, default="case.json")
    parser.add_argument("--data-dir","-d",default="./data")
    parser.add_argument("--dataset-name", "-n", default=["v1.0-trainval"], nargs="+", help="dataset to convert")
    parser.add_argument("--output-dir", "-o", type=Path, default="output", help="path to write MCAP files into")
    parser.add_argument("--list-only",  action="store_true",  help="lists the scenes and exits")
    parser.add_argument("--use-case", type = int,  default= 1, help="use case.json or not")

    args = parser.parse_args()

    pred_json_file = os.path.join(root_injson, args.result_json_name)    # baseline模型版本结果
    pred_json_file2 = os.path.join(root_injson, args.result_json_name2)  # 更新后的网络版本结果
    case_json_file = os.path.join(root_injson, args.case_json_name)

    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return
        convert_all(args.output_dir, name, nusc, pred_json_file, pred_json_file2, case_json_file, use_case=args.use_case)

if __name__ == "__main__":
    main()
