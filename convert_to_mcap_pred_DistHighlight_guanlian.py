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

def write_scene_to_mcap(nusc: NuScenes, scene, outputPath, predAnns, disthighlightInfo, guanlianInfo, idswInfo):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    data_path = Path(nusc.dataroot)

    cur_sample = nusc.get("sample", scene["first_sample_token"])
    pbar = tqdm(total = get_num_sample_data(nusc, scene), unit = "sample_data", desc = f"{scene_name} Sample Data", leave = False)

    outputPath.parent.mkdir(parents = True, exist_ok = True)

    with open(outputPath, "wb") as fp:
        print(f"Writing to {outputPath}")
        writer = Writer(fp, compression=CompressionType.LZ4)

        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile = "", library = "nuscenes2mcap")

        while cur_sample is not None:
            pbar.update(1)
            sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
            ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            ego2globalRotQuat  = ego_pose['rotation']
            ego2globalTran = ego_pose["translation"]
            ego2globalRotMat = Quaternion(ego2globalRotQuat).rotation_matrix  # used convert velo from global to ego
            stamp = get_time(ego_pose)

            # publish /tf
            protobuf_writer.write_message("/tf", get_ego_tf(ego_pose), stamp.to_nsec())
            # iterate sensors
            for sensor_id, sample_token in cur_sample["data"].items():
                sample_data = nusc.get("sample_data", sample_token)

                # create sensor transform
                protobuf_writer.write_message(
                    "/tf", get_sensor_tf(nusc, sensor_id, sample_data), stamp.to_nsec()
                )

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

            # publish /markers/annotations
            preAnnSceneUpdate = SceneUpdate()
            preAnnIdSceneUpdate = SceneUpdate()
            preAnnVeloEgoXSceneUpdate = SceneUpdate()
            preAnnVeloEgoYSceneUpdate = SceneUpdate()
            preAnnVeloGlobalXSceneUpdate = SceneUpdate()
            preAnnVeloGlobalYSceneUpdate = SceneUpdate()
            disthighlightPredAnnSceneUpdate = SceneUpdate()
            idswPreAnnSceneUpdate = SceneUpdate()
            gtAnnSceneUpdate = SceneUpdate()
            gtAnnVeloEgoXSceneUpdate = SceneUpdate()
            gtAnnVeloEgoYSceneUpdate = SceneUpdate()
            gtAnnVeloGlobalXSceneUpdate = SceneUpdate()
            gtAnnVeloGlobalYSceneUpdate = SceneUpdate()
            guanlianLineSceneUpdate = SceneUpdate()
 
            highlightPreAnns = []
            idswPreAnns      = []
            if str(stamp.to_nsec())[:-3] in disthighlightInfo.keys():
                highlightPreAnns = disthighlightInfo[str(stamp.to_nsec())[:-3]]        # list  预测结果中因为距离原理gt的高亮目标
            if str(stamp.to_nsec())[:-3] in idswInfo.keys():
                idswPreAnns = idswInfo[str(stamp.to_nsec())[:-3]]                         # list

            # 添加关联线
            guanlianGtAnns = {}
            guanlianPreAnns = {}
            guanlianIdpairs = {}   # 预测框与真值框的关联映射Map表
            if str(stamp.to_nsec())[:-3] in guanlianInfo:
                guanlianIdpairs = guanlianInfo[str(stamp.to_nsec())[:-3]] 

            # 预测框的cube
            for ann in predAnns[cur_sample['token']]:
                preAnnID = None
                # 不同形式的tracking结果的预测json文件, 这里的逻辑用于支持MUTR3D的tracking json文件
                if isinstance(ann["tracking_id"], int):
                    preAnnID = str(ann["tracking_id"])

                elif ann["tracking_id"][0] == 't':
                    preAnnID = ann["tracking_id"][7:-1]

                else:
                    preAnnID = ann["tracking_id"]

                ################### 计算预测框速度在自车系和世界系下的横向纵向速度分量, 用于拉ego系和global系下的横纵速度曲线 ##################
                preVelo2dGloabal = ann["velocity"][:2]
                # convert velo from global to ego
                preVelo3dGlobal = np.array([*preVelo2dGloabal, 0.0])
                preVelo3dEgo = preVelo3dGlobal @ np.linalg.inv(ego2globalRotMat).T
                preVelo2dEgo = preVelo3dEgo[:2]

                ###################  计算预测框在ego系和global系下的信息  #####################
                # conver ann from global to ego
                preAnnCenterGlobal = np.array([ann["translation"][0], ann["translation"][1], ann["translation"][2]])
                # conver ann_centerfrom global to ego
                preAnnCenterEgo = np.dot(np.linalg.inv(ego2globalRotMat), preAnnCenterGlobal - np.array(ego2globalTran))
                preAnnOrientationQuatGlobal = np.array([ann["rotation"][0], ann["rotation"][1], ann["rotation"][2], ann["rotation"][3]])
                preAnnQuatGlobal2Ego = Quaternion(matrix = np.linalg.inv(ego2globalRotMat))
                preAnnOrientationQuatEgo = preAnnQuatGlobal2Ego * preAnnOrientationQuatGlobal

                ######################  发布预测框位置框信息  #####################
                preAnnDeleteEntity      = preAnnSceneUpdate.deletions.add()
                preAnnDeleteEntity.type = 1
                preAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preAnnEntity          = preAnnSceneUpdate.entities.add()
                preAnnEntity.frame_id = "base_link"
                preAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnEntity.id       = preAnnID
                preAnnEntity.frame_locked = True


                preAnnMetadata = preAnnEntity.metadata.add()
                preAnnMetadata.key =  "category"
                preAnnMetadata.value = ann["tracking_name"]

                preAnncube = preAnnEntity.cubes.add()
                preAnncube.size.x  = ann["size"][1]
                preAnncube.size.y  = ann["size"][0]
                preAnncube.size.z  = ann["size"][2]
                preAnncube.color.r = 0
                preAnncube.color.g = 0
                preAnncube.color.b = 1
                preAnncube.color.a = 0.3

                preAnncube.pose.position.x    = preAnnCenterEgo[0]
                preAnncube.pose.position.y    = preAnnCenterEgo[1]
                preAnncube.pose.position.z    = preAnnCenterEgo[2]
                preAnncube.pose.orientation.w = preAnnOrientationQuatEgo[0]
                preAnncube.pose.orientation.x = preAnnOrientationQuatEgo[1]
                preAnncube.pose.orientation.y = preAnnOrientationQuatEgo[2]
                preAnncube.pose.orientation.z = preAnnOrientationQuatEgo[3]

                guanlianPreAnns[ann["tracking_id"]] = [preAnnCenterEgo[0], preAnnCenterEgo[1], preAnnCenterEgo[2], preAnnOrientationQuatEgo[1]] 

                ###################### 发布预测框ID信息(与预测框的话题分离时为了可视化方便操作)  #####################
                preAnnIdDeleteEntity = preAnnIdSceneUpdate.deletions.add()
                preAnnIdDeleteEntity.type = 1
                preAnnIdDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                preAnnIdEntity = preAnnIdSceneUpdate.entities.add()
                preAnnIdEntity.frame_id = "base_link"
                preAnnIdEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnIdEntity.id = preAnnID
                preAnnIdEntity.frame_locked = True

                preAnnIdTexts = preAnnIdEntity.texts.add()
                preAnnIdTexts.pose.position.x = preAnnCenterEgo[0]
                preAnnIdTexts.pose.position.y = preAnnCenterEgo[1]
                preAnnIdTexts.pose.position.z = preAnnCenterEgo[2]
                preAnnIdTexts.pose.orientation.w = preAnnOrientationQuatEgo[0]
                preAnnIdTexts.pose.orientation.x = preAnnOrientationQuatEgo[1]
                preAnnIdTexts.pose.orientation.y = preAnnOrientationQuatEgo[2]
                preAnnIdTexts.pose.orientation.z = preAnnOrientationQuatEgo[3]
                preAnnIdTexts.font_size = 0.5
                preAnnIdTexts.text = preAnnID

                ################ 发布预测框自车系纵向速度内容, 用于拉曲线 ##########################
                predAnnVeloEgoXDeleteEntity = preAnnVeloEgoXSceneUpdate.deletions.add()
                predAnnVeloEgoXDeleteEntity.type = 1
                predAnnVeloEgoXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                
                predAnnVeloEgoXEntity = preAnnVeloEgoXSceneUpdate.entities.add()
                predAnnVeloEgoXEntity.frame_id = "base_link"
                predAnnVeloEgoXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                predAnnVeloEgoXEntity.id = preAnnID
                predAnnVeloEgoXEntity.frame_locked = True

                predAnnVeloEgoXTexts = predAnnVeloEgoXEntity.texts.add()
                predAnnVeloEgoXTexts.pose.position.x = preAnnCenterEgo[0]
                predAnnVeloEgoXTexts.pose.position.y = preAnnCenterEgo[1]
                predAnnVeloEgoXTexts.pose.position.z = preAnnCenterEgo[2]
                predAnnVeloEgoXTexts.pose.orientation.w = preAnnOrientationQuatEgo[0]
                predAnnVeloEgoXTexts.pose.orientation.x = preAnnOrientationQuatEgo[1]
                predAnnVeloEgoXTexts.pose.orientation.y = preAnnOrientationQuatEgo[2]
                predAnnVeloEgoXTexts.pose.orientation.z = preAnnOrientationQuatEgo[3]
                predAnnVeloEgoXTexts.font_size = 0.01
                predAnnVeloEgoXTexts.text = str(preVelo2dEgo[0])

                ################ 发布预测框自车系横向速度内容, 用于拉曲线 ##########################
                preAnnVeloEgoYDeleteEntity = preAnnVeloEgoYSceneUpdate.deletions.add()
                preAnnVeloEgoYDeleteEntity.type = 1
                preAnnVeloEgoYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                preAnnVeloEgoYEntity = preAnnVeloEgoYSceneUpdate.entities.add()
                preAnnVeloEgoYEntity.frame_id = "base_link"
                preAnnVeloEgoYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnVeloEgoYEntity.id = preAnnID
                preAnnVeloEgoYEntity.frame_locked = True

                preAnnVeloEgoYTexts = preAnnVeloEgoYEntity.texts.add()
                preAnnVeloEgoYTexts.pose.position.x = preAnnCenterEgo[0]
                preAnnVeloEgoYTexts.pose.position.y = preAnnCenterEgo[1]
                preAnnVeloEgoYTexts.pose.position.z = preAnnCenterEgo[2]
                preAnnVeloEgoYTexts.pose.orientation.w = preAnnOrientationQuatEgo[0]
                preAnnVeloEgoYTexts.pose.orientation.x = preAnnOrientationQuatEgo[1]
                preAnnVeloEgoYTexts.pose.orientation.y = preAnnOrientationQuatEgo[2]
                preAnnVeloEgoYTexts.pose.orientation.z = preAnnOrientationQuatEgo[3]
                preAnnVeloEgoYTexts.font_size = 0.01
                preAnnVeloEgoYTexts.text = str(preVelo2dEgo[1])

                ################ 发布预测框世界系纵向速度内容, 用于拉曲线 ##########################
                preAnnVeloGlobalXDeleteEntity = preAnnVeloGlobalXSceneUpdate.deletions.add()
                preAnnVeloGlobalXDeleteEntity.type = 1
                preAnnVeloGlobalXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                
                preAnnVeloGlobalXEntity = preAnnVeloGlobalXSceneUpdate.entities.add()
                preAnnVeloGlobalXEntity.frame_id = "map"
                preAnnVeloGlobalXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnVeloGlobalXEntity.id = preAnnID
                preAnnVeloGlobalXEntity.frame_locked = True

                preAnnVeloGlobalXTexts = preAnnVeloGlobalXEntity.texts.add()
                preAnnVeloGlobalXTexts.pose.position.x = preAnnCenterGlobal[0]
                preAnnVeloGlobalXTexts.pose.position.y = preAnnCenterGlobal[1]
                preAnnVeloGlobalXTexts.pose.position.z = preAnnCenterGlobal[2]
                preAnnVeloGlobalXTexts.pose.orientation.w = preAnnOrientationQuatGlobal[0]
                preAnnVeloGlobalXTexts.pose.orientation.x = preAnnOrientationQuatGlobal[1]
                preAnnVeloGlobalXTexts.pose.orientation.y = preAnnOrientationQuatGlobal[2]
                preAnnVeloGlobalXTexts.pose.orientation.z = preAnnOrientationQuatGlobal[3]
                preAnnVeloGlobalXTexts.font_size = 0.01
                preAnnVeloGlobalXTexts.text = str(preVelo2dGloabal[0])

                ################ 发布预测框世界系横向速度内容, 用于拉曲线 ##########################
                preAnnVeloGlobalYDeleteEntity = preAnnVeloGlobalYSceneUpdate.deletions.add()
                preAnnVeloGlobalYDeleteEntity.type = 1
                preAnnVeloGlobalYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                
                preAnnVeloGlobalYEntity = preAnnVeloGlobalYSceneUpdate.entities.add()
                preAnnVeloGlobalYEntity.frame_id = "map"
                preAnnVeloGlobalYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                preAnnVeloGlobalYEntity.id = preAnnID
                preAnnVeloGlobalYEntity.frame_locked = True

                preAnnVeloGlobalYTexts = preAnnVeloGlobalYEntity.texts.add()
                preAnnVeloGlobalYTexts.pose.position.x = preAnnCenterGlobal[0]
                preAnnVeloGlobalYTexts.pose.position.y = preAnnCenterGlobal[1]
                preAnnVeloGlobalYTexts.pose.position.z = preAnnCenterGlobal[2]
                preAnnVeloGlobalYTexts.pose.orientation.w = preAnnOrientationQuatGlobal[0]
                preAnnVeloGlobalYTexts.pose.orientation.x = preAnnOrientationQuatGlobal[1]
                preAnnVeloGlobalYTexts.pose.orientation.y = preAnnOrientationQuatGlobal[2]
                preAnnVeloGlobalYTexts.pose.orientation.z = preAnnOrientationQuatGlobal[3]
                preAnnVeloGlobalYTexts.font_size = 0.01
                preAnnVeloGlobalYTexts.text = str(preVelo2dGloabal[1])

                ################ 发布距离关联的真值框超过阈值的预测高亮框(结合评估工具箱理解)  #################
                ################ 评估工具箱链接地址: https://github.com/hjfenghj/streampetr_nuscenens_tracking_eval ###############
                if preAnnID in highlightPreAnns:  # 高亮的预测目标
                    disthighlightPredAnnDeleteEntity = disthighlightPredAnnSceneUpdate.deletions.add()
                    disthighlightPredAnnDeleteEntity.type = 1
                    disthighlightPredAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                    disthighlightPredAnnEntity = disthighlightPredAnnSceneUpdate.entities.add()
                    disthighlightPredAnnEntity.frame_id = "base_link"
                    disthighlightPredAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                    disthighlightPredAnnEntity.id = preAnnID
                    disthighlightPredAnnEntity.frame_locked = True

                    disthighlightPredAnnCube = disthighlightPredAnnEntity.cubes.add()
                    disthighlightPredAnnCube.pose.position.x = preAnnCenterGlobal[0]
                    disthighlightPredAnnCube.pose.position.y = preAnnCenterGlobal[1]
                    disthighlightPredAnnCube.pose.position.z = preAnnCenterGlobal[2]
                    disthighlightPredAnnCube.pose.orientation.w = preAnnOrientationQuatGlobal[0]
                    disthighlightPredAnnCube.pose.orientation.x = preAnnOrientationQuatGlobal[1]
                    disthighlightPredAnnCube.pose.orientation.y = preAnnOrientationQuatGlobal[2]
                    disthighlightPredAnnCube.pose.orientation.z = preAnnOrientationQuatGlobal[3]
                    disthighlightPredAnnCube.size.x = ann["size"][1]
                    disthighlightPredAnnCube.size.y = ann["size"][0]
                    disthighlightPredAnnCube.size.z = ann["size"][2]
                    disthighlightPredAnnCube.color.r = 1
                    disthighlightPredAnnCube.color.g = 0
                    disthighlightPredAnnCube.color.b = 0
                    disthighlightPredAnnCube.color.a = 0.6

                ################ 发布发生ID跳变的预测高亮框(结合评估工具箱理解)  #################
                ################ 评估工具箱链接地址: https://github.com/hjfenghj/streampetr_nuscenens_tracking_eval ###############
                if preAnnID in idswPreAnns:
                    c = np.array([160, 32, 240]) / 255.0
                    preIdswAnnDeleteEntity = idswPreAnnSceneUpdate.deletions.add()
                    preIdswAnnDeleteEntity.type = 1
                    preIdswAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                    preIdswAnnEntity = idswPreAnnSceneUpdate.entities.add()
                    preIdswAnnEntity.frame_id = "base_link"
                    preIdswAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                    preIdswAnnEntity.id = preAnnID
                    preIdswAnnEntity.frame_locked = True

                    preIdswAnnCube = preIdswAnnEntity.cubes.add()
                    preIdswAnnCube.pose.position.x = preAnnCenterGlobal[0]
                    preIdswAnnCube.pose.position.y = preAnnCenterGlobal[1]
                    preIdswAnnCube.pose.position.z = preAnnCenterGlobal[2]
                    preIdswAnnCube.pose.orientation.w = preAnnOrientationQuatGlobal[0]
                    preIdswAnnCube.pose.orientation.x = preAnnOrientationQuatGlobal[1]
                    preIdswAnnCube.pose.orientation.y = preAnnOrientationQuatGlobal[2]
                    preIdswAnnCube.pose.orientation.z = preAnnOrientationQuatGlobal[3]
                    preIdswAnnCube.size.x = ann["size"][1]
                    preIdswAnnCube.size.y = ann["size"][0]
                    preIdswAnnCube.size.z = ann["size"][2]
                    preIdswAnnCube.color.r = 160.0 / 255.0
                    preIdswAnnCube.color.g = 32.0 / 255.0
                    preIdswAnnCube.color.b = 240.0 / 255.0
                    preIdswAnnCube.color.a = 0.8
                
            protobuf_writer.write_message("/markers/preAnns", preAnnSceneUpdate, stamp.to_nsec())   
            protobuf_writer.write_message("/markers/preAnnsId", preAnnIdSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preAnnVeloEgoX", preAnnVeloEgoXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preAnnVeloEgoY", preAnnVeloEgoYSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preAnnVeloGlobalX", preAnnVeloGlobalXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/preAnnVeloGlobalY", preAnnVeloGlobalYSceneUpdate, stamp.to_nsec())             
            protobuf_writer.write_message("/markers/disthighlightPredAnn", disthighlightPredAnnSceneUpdate, stamp.to_nsec())
            protobuf_writer.write_message("/markers/pred_idsw", idswPreAnnSceneUpdate, stamp.to_nsec())         

            # gt框的cube
            for gtAnnToken in cur_sample["anns"]:
                gtAnnInfo = nusc.get("sample_annotation", gtAnnToken)

                ################### 计算真值框速度在自车系和世界系下的横向纵向速度分量, 用于拉ego系和global系下的横纵速度曲线 ##################
                # gt_anns global velo 
                gtVelo2dGlobal = nusc.box_velocity(gtAnnToken)[:2]                
                # convert velo from global to ego
                gtVelo3dGlobal = np.array([*gtVelo2dGlobal, 0.0])
                gtVelo3dEgo = gtVelo3dGlobal @ np.linalg.inv(ego2globalRotMat).T
                gtVelo2dEgo = gtVelo3dEgo[:2]

                ###################  计算预测框在ego系和global系下的信息  #####################
                # conver box from global to ego
                gtAnnCenterGlobal = np.array([gtAnnInfo["translation"][0], gtAnnInfo["translation"][1], gtAnnInfo["translation"][2]])
                # conver ann_centerfrom global to ego
                gtAnnCenterEgo = np.dot(np.linalg.inv(ego2globalRotMat), gtAnnCenterGlobal - np.array(ego2globalTran))
                gtAnnOrientationQuatGlobal = np.array([gtAnnInfo["rotation"][0], gtAnnInfo["rotation"][1], gtAnnInfo["rotation"][2], gtAnnInfo["rotation"][3]])
                gtAnnQuatGlobal2Ego = Quaternion(matrix = np.linalg.inv(ego2globalRotMat))
                gtAnnOrientationQuatEgo = gtAnnQuatGlobal2Ego * gtAnnOrientationQuatGlobal

                gtAnnID = gtAnnInfo["instance_token"][:4]

                ######################  发布真值框信息  #####################
                gtAnnDeleteEntity = gtAnnSceneUpdate.deletions.add()
                gtAnnDeleteEntity.type = 1
                gtAnnDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)   
                         
                gtAnnEntity = gtAnnSceneUpdate.entities.add()
                gtAnnEntity.frame_id = "base_link"
                gtAnnEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnEntity.id = gtAnnID   # 为了拉曲线输入id比较方便               
                gtAnnEntity.frame_locked = True

                gtAnnMetadata = gtAnnEntity.metadata.add()
                gtAnnMetadata.key = "category"
                gtAnnMetadata.value = gtAnnInfo["category_name"]

                gtAnnCube = gtAnnEntity.cubes.add()
                gtAnnCube.size.x = gtAnnInfo["size"][1]
                gtAnnCube.size.y = gtAnnInfo["size"][0]
                gtAnnCube.size.z = gtAnnInfo["size"][2]
                gtAnnCube.color.r = 0
                gtAnnCube.color.g = 1
                gtAnnCube.color.b = 0
                gtAnnCube.color.a = 0.3
                gtAnnCube.pose.position.x = gtAnnCenterEgo[0]
                gtAnnCube.pose.position.y = gtAnnCenterEgo[1]
                gtAnnCube.pose.position.z = gtAnnCenterEgo[2]
                gtAnnCube.pose.orientation.w = gtAnnOrientationQuatEgo[0]
                gtAnnCube.pose.orientation.x = gtAnnOrientationQuatEgo[1]
                gtAnnCube.pose.orientation.y = gtAnnOrientationQuatEgo[2]
                gtAnnCube.pose.orientation.z = gtAnnOrientationQuatEgo[3]

                guanlianGtAnns[gtAnnInfo["instance_token"]] = [gtAnnCenterEgo[0], gtAnnCenterEgo[1], gtAnnCenterEgo[2], gtAnnOrientationQuatEgo[1]]

                ################ 发布真值框自车系纵向速度内容, 用于拉曲线 ##########################
                gtAnnVeloEgoXDeleteEntity = gtAnnVeloEgoXSceneUpdate.deletions.add()
                gtAnnVeloEgoXDeleteEntity.type = 1
                gtAnnVeloEgoXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                gtAnnVeloEgoXEntity = gtAnnVeloEgoXSceneUpdate.entities.add()
                gtAnnVeloEgoXEntity.frame_id = "base_link"
                gtAnnVeloEgoXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnVeloEgoXEntity.id = gtAnnID
                gtAnnVeloEgoXEntity.frame_locked = True

                gtAnnVeloEgoXTexts = gtAnnVeloEgoXEntity.texts.add()
                gtAnnVeloEgoXTexts.pose.position.x = gtAnnCenterEgo[0]
                gtAnnVeloEgoXTexts.pose.position.y = gtAnnCenterEgo[1]
                gtAnnVeloEgoXTexts.pose.position.z = gtAnnCenterEgo[2]
                gtAnnVeloEgoXTexts.pose.orientation.w = gtAnnOrientationQuatEgo[0]
                gtAnnVeloEgoXTexts.pose.orientation.x = gtAnnOrientationQuatEgo[1]
                gtAnnVeloEgoXTexts.pose.orientation.y = gtAnnOrientationQuatEgo[2]
                gtAnnVeloEgoXTexts.pose.orientation.z = gtAnnOrientationQuatEgo[3]
                gtAnnVeloEgoXTexts.font_size = 0.01
                gtAnnVeloEgoXTexts.text = str(gtVelo2dEgo[0])

                ################ 发布真值框自车系横向速度内容, 用于拉曲线 ##########################
                gtAnnVeloEgoYDeleteEntity = gtAnnVeloEgoYSceneUpdate.deletions.add()
                gtAnnVeloEgoYDeleteEntity.type = 1
                gtAnnVeloEgoYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtAnnVeloEgoYEntity = gtAnnVeloEgoYSceneUpdate.entities.add()
                gtAnnVeloEgoYEntity.frame_id = "base_link"
                gtAnnVeloEgoYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnVeloEgoYEntity.id = gtAnnID
                gtAnnVeloEgoYEntity.frame_locked = True

                gtAnnVeloEgoYTexts = gtAnnVeloEgoYEntity.texts.add()
                gtAnnVeloEgoYTexts.pose.position.x = gtAnnCenterEgo[0]
                gtAnnVeloEgoYTexts.pose.position.y = gtAnnCenterEgo[1]
                gtAnnVeloEgoYTexts.pose.position.z = gtAnnCenterEgo[2]
                gtAnnVeloEgoYTexts.pose.orientation.w = gtAnnOrientationQuatEgo[0]
                gtAnnVeloEgoYTexts.pose.orientation.x = gtAnnOrientationQuatEgo[1]
                gtAnnVeloEgoYTexts.pose.orientation.y = gtAnnOrientationQuatEgo[2]
                gtAnnVeloEgoYTexts.pose.orientation.z = gtAnnOrientationQuatEgo[3]
                gtAnnVeloEgoYTexts.font_size = 0.01
                gtAnnVeloEgoYTexts.text = str(gtVelo2dEgo[1])
                
                ################ 发布真值框世界系纵向速度内容, 用于拉曲线(因为横纵速度值不会自动跟随基坐标系变化) ##########################
                gtAnnVeloGlobalXDeleteEntity = gtAnnVeloGlobalXSceneUpdate.deletions.add()
                gtAnnVeloGlobalXDeleteEntity.type = 1
                gtAnnVeloGlobalXDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtAnnVeloGlobalXEntity = gtAnnVeloGlobalXSceneUpdate.entities.add()
                gtAnnVeloGlobalXEntity.frame_id = "map"
                gtAnnVeloGlobalXEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnVeloGlobalXEntity.id = gtAnnID
                gtAnnVeloGlobalXEntity.frame_locked = True

                gtAnnVeloGlobalXTexts = gtAnnVeloGlobalXEntity.texts.add()
                gtAnnVeloGlobalXTexts.pose.position.x = gtAnnCenterGlobal[0]
                gtAnnVeloGlobalXTexts.pose.position.y = gtAnnCenterGlobal[1]
                gtAnnVeloGlobalXTexts.pose.position.z = gtAnnCenterGlobal[2]
                gtAnnVeloGlobalXTexts.pose.orientation.w = gtAnnOrientationQuatGlobal[0]
                gtAnnVeloGlobalXTexts.pose.orientation.x = gtAnnOrientationQuatGlobal[1]
                gtAnnVeloGlobalXTexts.pose.orientation.y = gtAnnOrientationQuatGlobal[2]
                gtAnnVeloGlobalXTexts.pose.orientation.z = gtAnnOrientationQuatGlobal[3]
                gtAnnVeloGlobalXTexts.font_size = 0.01
                gtAnnVeloGlobalXTexts.text = str(gtVelo2dGlobal[0])

                ################ 发布真值框世界系横向速度内容, 用于拉曲线 ##########################
                gtAnnVeloGlobalYDeleteEntity = gtAnnVeloGlobalYSceneUpdate.deletions.add()
                gtAnnVeloGlobalYDeleteEntity.type = 1
                gtAnnVeloGlobalYDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                gtAnnVeloGlobalYEntity = gtAnnVeloGlobalYSceneUpdate.entities.add()
                gtAnnVeloGlobalYEntity.frame_id = "map"
                gtAnnVeloGlobalYEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                gtAnnVeloGlobalYEntity.id = gtAnnID
                gtAnnVeloGlobalYEntity.frame_locked = True

                gtAnnVeloGlobalYTexts = gtAnnVeloGlobalYEntity.texts.add()
                gtAnnVeloGlobalYTexts.pose.position.x = gtAnnCenterGlobal[0]
                gtAnnVeloGlobalYTexts.pose.position.y = gtAnnCenterGlobal[1]
                gtAnnVeloGlobalYTexts.pose.position.z = gtAnnCenterGlobal[2]
                gtAnnVeloGlobalYTexts.pose.orientation.w = gtAnnOrientationQuatGlobal[0]
                gtAnnVeloGlobalYTexts.pose.orientation.x = gtAnnOrientationQuatGlobal[1]
                gtAnnVeloGlobalYTexts.pose.orientation.y = gtAnnOrientationQuatGlobal[2]
                gtAnnVeloGlobalYTexts.pose.orientation.z = gtAnnOrientationQuatGlobal[3]
                gtAnnVeloGlobalYTexts.font_size = 0.01
                gtAnnVeloGlobalYTexts.text = str(gtVelo2dGlobal[1])

            protobuf_writer.write_message("/markers/gtAnnVeloEgoX", gtAnnVeloEgoXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gtAnnVeloEgoY", gtAnnVeloEgoYSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gtAnnVeloGlobalX", gtAnnVeloGlobalXSceneUpdate, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/gtAnnVeloGlobalY", gtAnnVeloGlobalYSceneUpdate, stamp.to_nsec())  
            protobuf_writer.write_message("/markers/gtAnns", gtAnnSceneUpdate, stamp.to_nsec())


            ################ 发布预测框与真值框的关联线(结合评估工具箱理解)  #################
            ################ 评估工具箱链接地址: https://github.com/hjfenghj/streampetr_nuscenens_tracking_eval ###############
            for gtId, predId in guanlianIdpairs.items():
                if gtId not in guanlianGtAnns or predId not in guanlianPreAnns:
                    continue

                guanlianLineEntityDeleteEntity = guanlianLineSceneUpdate.deletions.add()
                guanlianLineEntityDeleteEntity.type = 1
                guanlianLineEntityDeleteEntity.timestamp.FromNanoseconds(stamp.to_nsec() + 50)

                guanlianLineEntity = guanlianLineSceneUpdate.entities.add()
                guanlianLineEntity.id = gtId + predId
                guanlianLineEntity.frame_id = "base_link"
                guanlianLineEntity.timestamp.FromNanoseconds(stamp.to_nsec())
                guanlianLineEntity.frame_locked = True

                guanlianLineLine = guanlianLineEntity.lines.add()
                guanlianLineLine.type = 0
                guanlianLineLine.thickness = 0.2
                guanlianLineLine.color.r = 1.0
                guanlianLineLine.color.g = 97/255
                guanlianLineLine.color.b = 0
                guanlianLineLine.color.a = 0.5
                guanlianLineLine.points.add(x = guanlianGtAnns[gtId][0], y = guanlianGtAnns[gtId][1], z = guanlianGtAnns[gtId][2])
                guanlianLineLine.points.add(x = guanlianPreAnns[predId][0], y = guanlianPreAnns[predId][1], z = guanlianPreAnns[predId][2])
            protobuf_writer.write_message("/markers/guanlianLine", guanlianLineSceneUpdate, stamp.to_nsec())

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
        print(f"Finished writing {outputPath}")


def convert_all(output_dir, name, nusc, predJsonFile, DisthighlightJsonFile, guanlianJsonFile, idswJsonFile):
    with open(predJsonFile, 'r') as J:
        preResultInfos = json.load(J)
    preResultInfo = preResultInfos['results']

    with open(guanlianJsonFile, 'r') as fp:
        guanlianInfos = json.load(fp)

    with open(DisthighlightJsonFile, 'r') as fp:
        disthighlightInfos = json.load(fp)

    with open(idswJsonFile, 'r') as fp:
        idswInfos = json.load(fp)

    splits = create_splits_scenes()

    for scene in nusc.scene:
        # 判断是否为val的场景, 用于网络推理结果的可视化
        # 因此推理结果分为不同的分支 val train test等     
        if scene['name'] not in splits['val']:    # 这里需要注意，以后的你，如果需要可视化val部分的结果，就需要去掉这个。另外也可以写成指定场景可视化，留给你做吧，加油
            continue
        scene_name = scene["name"]
        mcap_name = f"NuScenes-{name}-{scene_name}.mcap"
        
        disthighlightInfo = {}
        guanlianInfo = {}
        if scene['token'] in disthighlightInfos.keys():
            disthighlightInfo = disthighlightInfos[scene['token']]
            guanlianInfo = guanlianInfos[scene['token']]

        idswInfo = {}
        if scene_name in idswInfos.keys():
            idswInfo = idswInfos[scene_name]

        write_scene_to_mcap(nusc, scene, output_dir / mcap_name, preResultInfo, disthighlightInfo, guanlianInfo, idswInfo)


def main():
    rootInjson = "./input_json"
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_json_name", "-p", default= "tracking_result.json", help="path to infer result",)
    parser.add_argument("--idsw_json_name", type=str, default="idsw_clear_info.json")
    parser.add_argument("--highlight_info", type=str, default="highlight_info.json")
    parser.add_argument("--guanlian_json_name",type=str,default="guanlian_info.json")     
    parser.add_argument("--data-dir","-d",default="./data")
    parser.add_argument("--dataset-name", "-n", default=["v1.0-trainval"], nargs="+", help="dataset to convert")
    parser.add_argument("--output-dir", "-o", type=Path, default="output", help="path to write MCAP files into",)
    # parser.add_argument("--scene", "-s", default = ["scene-0103","scene-0916"] ,nargs="+", type=list,help="specific scene(s) to write")
    parser.add_argument("--list-only", action="store_true", help="lists the scenes and exits")

    args = parser.parse_args()

    predJsonFile = os.path.join(rootInjson, args.pred_json_name)
    disthighlightJsonFile = os.path.join(rootInjson, args.highlight_info)
    guanlianJsonFile = os.path.join(rootInjson, args.guanlian_json_name)
    idswJsonFile = os.path.join(rootInjson, args.idsw_json_name)

    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return

        convert_all(args.output_dir, name, nusc, predJsonFile, disthighlightJsonFile, guanlianJsonFile, idswJsonFile)

if __name__ == "__main__":
    main()
