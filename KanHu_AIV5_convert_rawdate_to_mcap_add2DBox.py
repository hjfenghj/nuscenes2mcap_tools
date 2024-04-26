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
from scipy.spatial.transform import Rotation as R
import datetime
import glob
import re

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
import xml.etree.ElementTree as ET

cam_name_list = ["camblf", "camfcf", "cambrf", "camflb", "cambcb", "camfrb"]
cate_list = []

def get_color(idx):
    classname_to_color = [ # RGB.
        (70, 130, 180),  # Steelblue
        (0, 0, 230),  # Blue
        (135, 206, 235),  # Skyblue,
        (100, 149, 237),  # Cornflowerblue
        (219, 112, 147),  # Palevioletred
        (0, 0, 128),  # Navy,
        (240, 128, 128),  # Lightcoral
        (138, 43, 226),  # Blueviolet
        (112, 128, 144),  # Slategrey
        (210, 105, 30),  # Chocolate
        (105, 105, 105),  # Dimgrey
        (47, 79, 79),  # Darkslategrey
        (188, 143, 143),  # Rosybrown
        (220, 20, 60),  # Crimson
        (255, 127, 80),  # Coral
        (255, 69, 0),  # Orangered
        (255, 158, 0),  # Orange
        (233, 150, 70),  # Darksalmon
        (255, 83, 0),
        (255, 215, 0),  # Gold
        (255, 61, 99),  # Red
        (255, 140, 0),  # Darkorange
        (255, 99, 71),  # Tomato
        (0, 207, 191),  # nuTonomy green
        (175, 0, 75),
        (75, 0, 75),
        (112, 180, 60),
        (222, 184, 135),  # Burlywood
        (255, 228, 196),  # Bisque
        (0, 175, 0),  # Green
        (255, 240, 245),
        (83, 255, 0),
        (215, 255, 0),  # Gold
        (61, 255, 99),  # Red
        (140, 255, 0),  # Darkorange
        (99, 255, 71),  # Tomato
        (75, 75, 0),
        (112, 60, 180),
        (222, 135, 184),  # Burlywood
        (255, 228, 196),  # Bisque
        (0, 175, 0),  # Green
        (255, 240, 245),
        (83, 255, 0),
        (215, 255, 0),  # Gold
        (61, 255, 99),  # Red
        (140, 255, 0),  # Darkorange
        (99, 255, 71)]
    return  classname_to_color[idx]


def write_boxes_image_annotations(protobuf_writer, ImageBoxs, topic_ns, stamp, cur_2dID_3dID_map):
    """
    ImageBoxs -> List
    """
    CurCamIDMap = cur_2dID_3dID_map[topic_ns]
    msg1 = ImageAnnotations()
    msg2 = ImageAnnotations()
    msg3 = ImageAnnotations()
    points_ann = msg1.points.add()
    points_ann.timestamp.FromMicroseconds(stamp)
    points_ann.type = PointsAnnotation.Type.LINE_LIST
    points_ann.thickness = 2
    for imagebox in ImageBoxs:
        points2d = []
        if imagebox["sideCorUpX"] == "":
            x1 = imagebox["bodyRectX"]
            y1 = imagebox["bodyRectY"]
            x2 = imagebox["bodyRectX"] + imagebox["bodyRectWidth"]
            y2 = imagebox["bodyRectY"] + imagebox["bodyRectHeight"]
            points2d.append((x1,y1))      # 0
            points2d.append((x2,y1))      # 1
            points2d.append((x2,y2))      # 2
            points2d.append((x1,y2))      # 3
            points2d.append((x2,y1))      # 4
            points2d.append((x2,y2))      # 5
            points2d.append((x1,y2))      # 6
            points2d.append((x1,y1))      # 7
        else:
            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2
            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3

            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 4
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 5
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6
            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7

            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2
            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3
            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 5
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6
            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7
            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 4

            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 5
            points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6 
            points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2   

            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 4
            points2d.append((imagebox["sideCorUpX"] - imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7
            points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3


            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2
            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3

            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 4
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 5
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6
            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7

            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2
            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3
            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 5
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6
            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7
            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 4

            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"]))  # 1
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"])) # 5
            # points2d.append((imagebox["sideCorUpX"] + imagebox["bodyRectWidth"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 6
            # points2d.append((imagebox["bodyRectX"] + imagebox["bodyRectWidth"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 2   

            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"]))  # 0
            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"])) # 4
            # points2d.append((imagebox["sideCorUpX"], imagebox["sideCorUpY"] + imagebox["bodyRectHeight"])) # 7
            # points2d.append((imagebox["bodyRectX"], imagebox["bodyRectY"] + imagebox["bodyRectHeight"]))  # 3
        for p in points2d:
            points_ann.points.add(x=p[0], y=p[1])

        # texts 2D cate
        texts_ann = msg2.texts.add()
        texts_ann.timestamp.FromMicroseconds(stamp)
        texts_ann.font_size = 10
        # texts_ann.position.x = min(map(lambda pt: pt[0], points2d))
        # texts_ann.position.y = min(map(lambda pt: pt[1], points2d))
        texts_ann.position.x = imagebox["bodyRectX"]
        texts_ann.position.y = imagebox["bodyRectY"]
        texts_ann.text = imagebox["type"]
        texts_ann.text_color.r = 0
        texts_ann.text_color.g = 0
        texts_ann.text_color.b = 0
        texts_ann.text_color.a = 0.3
        texts_ann.background_color.r = 1
        texts_ann.background_color.g = 1
        texts_ann.background_color.b = 1
        texts_ann.background_color.a = 0

        # text 3D box id
        texts3 = msg3.texts.add()
        texts3.timestamp.FromMicroseconds(stamp)
        texts3.font_size =  36
        texts3.position.x = imagebox["bodyRectX"]
        texts3.position.y = imagebox["bodyRectY"] + imagebox["bodyRectHeight"]
        if str(imagebox["ID"]) in CurCamIDMap.keys():
            texts3.text = str(CurCamIDMap[str(imagebox["ID"])])
        texts3.text_color.r = 0
        texts3.text_color.g = 0
        texts3.text_color.b = 0
        texts3.text_color.a = 0.8
        texts3.background_color.r = 1
        texts3.background_color.g = 1
        texts3.background_color.b = 1
        texts3.background_color.a = 0    
    protobuf_writer.write_message(topic_ns + "/annotations_3D_ID", msg3, points_ann.timestamp.ToNanoseconds())
    protobuf_writer.write_message(topic_ns + "/annotations_cate", msg2, points_ann.timestamp.ToNanoseconds())
    protobuf_writer.write_message(topic_ns + "/annotations", msg1, points_ann.timestamp.ToNanoseconds())

def get_translation(data):
    return Vector3(x=data["translation"][0], y=data["translation"][1], z=data["translation"][2])

def get_rotation(data):
    return foxglove_Quaternion(x=data["rotation"][1], y=data["rotation"][2], z=data["rotation"][3], w=data["rotation"][0])


def get_time(data):
    t = rospy.Time()
    t.secs, msecs = divmod(data, 1_000_000)
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


def get_camera(file_path, timestamp, frame_id):
    jpg_filename = file_path
    msg = CompressedImage()
    msg.timestamp.FromMicroseconds(timestamp)
    msg.frame_id = frame_id
    msg.format = "jpg"
    if os.path.exists(jpg_filename):
        with open(jpg_filename, "rb") as jpg_file:
            msg.data = jpg_file.read()
    return msg


def get_camera_info(timestamp,frame_id,H,W,cam_intrinsic,cam_D):
    msg_info = CameraCalibration()
    msg_info.timestamp.FromMicroseconds(timestamp)
    msg_info.frame_id = frame_id
    msg_info.height = H
    msg_info.width = W
    msg_info.K[:] = (cam_intrinsic[r][c] for r in range(3) for c in range(3))
    # msg_info.D[:] = (float(d) for d in cam_D)
    msg_info.R[:] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg_info.P[:] = [msg_info.K[0], msg_info.K[1], msg_info.K[2], 0, msg_info.K[3], msg_info.K[4], msg_info.K[5], 0, 0, 0, 1, 0]
    return msg_info


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

# bin文件读取
def get_bin(data_path, timestamp, frame_id) -> PointCloud:
    pc_filename = data_path 

    with open(pc_filename, "rb") as pc_file:
        msg = PointCloud()
        msg.frame_id = frame_id
        msg.timestamp.FromMicroseconds(int(str(timestamp)[:-3]))
        msg.fields.add(name="x", offset=0, type=PackedElementField.FLOAT32),
        msg.fields.add(name="y", offset=4, type=PackedElementField.FLOAT32),
        msg.fields.add(name="z", offset=8, type=PackedElementField.FLOAT32),
        msg.fields.add(name="intensity", offset=12, type=PackedElementField.FLOAT32),
        msg.fields.add(name="ring", offset=16, type=PackedElementField.FLOAT32),
        msg.point_stride = len(msg.fields) * 4  # 4 bytes per field
        msg.data = pc_file.read()
        return msg


def get_ego_tf(timestamp,translation,rotation):
    ego_tf = FrameTransform()
    ego_tf.parent_frame_id = "map"
    ego_tf.timestamp.FromMicroseconds(timestamp)
    ego_tf.child_frame_id = "base_link"    #自车坐标系，随着车辆移动而变化
    ego_tf.translation.CopyFrom(Vector3(x=translation[0], y=translation[1], z=translation[2]))  #自车在世界坐标系中的位置
    ego_tf.rotation.CopyFrom(foxglove_Quaternion(x=rotation[1], y=rotation[2], z=rotation[3], w=rotation[0]))
    return ego_tf

def get_lidar_tf(sensor_id, timestamp,translation,rotation):
    sensor_tf = FrameTransform()
    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(timestamp)
    sensor_tf.child_frame_id = sensor_id
    sensor_tf.translation.CopyFrom(Vector3(x=translation[0], y=translation[1], z=translation[2]))
    sensor_tf.rotation.CopyFrom(foxglove_Quaternion(x=rotation[1], y=rotation[2], z=rotation[3], w=rotation[0])) 
    return sensor_tf

def get_sensor_tf(sensor_id, timestamp, translation, rotation):
    sensor_tf = FrameTransform()
    sensor_tf.parent_frame_id = "base_link"
    sensor_tf.timestamp.FromMicroseconds(timestamp)
    sensor_tf.child_frame_id = sensor_id

    sensor_tf.translation.CopyFrom(Vector3(x=translation[0], y=translation[1], z=translation[2]))
    sensor_tf.rotation.CopyFrom(foxglove_Quaternion(x=rotation[1], y=rotation[2], z=rotation[3], w=rotation[0]))
    return sensor_tf

def get_car_scene_update(stamp) -> SceneUpdate:
    scene_update = SceneUpdate()
    entity = scene_update.entities.add()
    entity.frame_id = "base_link"
    entity.timestamp.FromNanoseconds(stamp)
    entity.id = "car"
    entity.frame_locked = True
    model = entity.models.add()
    model.pose.position.x = 0
    model.pose.orientation.w = 1
    model.scale.x = 1
    model.scale.y = 1
    model.scale.z = 1
    model.url = "https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    return scene_update

def conver_time(label_file_path):
    weimiao = os.path.basename(label_file_path)[-14:-11]
    timestamp_str = os.path.basename(label_file_path)[-34:-15]
    # 解析时间字符串，最后三位是毫秒
    timestamp_obj = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S-%f')
    # 提取时间信息
    year = timestamp_obj.year
    month = timestamp_obj.month
    day = timestamp_obj.day
    hour = timestamp_obj.hour
    minute = timestamp_obj.minute
    second = timestamp_obj.second
    millisecond = timestamp_obj.microsecond // 1000  # 将微秒转换为毫秒
    # 构建带有毫秒的时间对象
    timestamp_with_milli = datetime(year, month, day, hour, minute, second, millisecond * 1000)
    # 转换为 Unix 时间戳（秒）
    unix_timestamp = datetime.timestamp(timestamp_with_milli)
    unix_timestamp += int(weimiao)*0.000001
    unix_timestamp = unix_timestamp*1000000
    return unix_timestamp     # 16位时间戳

def read_ins(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()  # 逐行读取文件内容
    result = {}
    # 解析每一行的 JSON 数据
    for line in data:
        json_data = json.loads(line)

        # 提取特定键值对应的值
        time_meas = json_data['header']['timeMeas']
        values_dict = {
            "lla": json_data['lla'],                   ## 表示车辆所处位置的纬度、经度和海拔高度。
            "llaStd": json_data['llaStd'],             # 经纬度高度的标准差
            "velocityEnu": json_data['velocityEnu'],   # 东北天（ENU）坐标系下的速度信息
            "velocityStd": json_data['velocityStd'],   # 速度的标准差
            "attitudeRpy": json_data['attitudeRpy'],   ## 分别对应于车辆围绕X、Y和Z轴的旋转
            "attitudeStd": json_data['attitudeStd'],   # 姿态角度的标准差
            "insStatus": json_data['insStatus'],       # INS 状态信息
            "insDiagnose": json_data['insDiagnose']    # INS 诊断信息        
        }
        # 将键值对添加到结果字典中
        result[time_meas] = values_dict
    return result

def parse_ins_info(ins_info, lla=True):
    loc = ins_info["lla"]
    if lla:
        cur_pos = np.array([loc['y'], loc['x'], loc['z']])  # lon lat alt
        cur_pos[:2] = np.rad2deg(cur_pos)[:2]
    else:
        cur_pos = np.array([loc['x'], loc['y'], loc['z']])  # x y z

    attitudeRpy = ins_info['attitudeRpy']
    cur_euler = np.array([attitudeRpy['x'], attitudeRpy['y'], attitudeRpy['z']])
    return cur_pos, cur_euler

def get_closest_ins(ins_infos,sensor_time):
    closest_timestamp = min(ins_infos.keys(), key=lambda x: abs(int(x) - sensor_time))
    closest_ins = ins_infos[closest_timestamp]
    cur_pos, cur_euler = parse_ins_info(closest_ins)
    return cur_pos, cur_euler
 
def read_lidar_params_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_dict = {}
    for line in lines:
        line = line.strip()  # 去除换行符或空格
        key, value = line.split(':')
        key = key.strip()  # 去除键的空格
        if ' ' in value:
            value = value.split()
            value = [float(v) if '.' in v else int(v) for v in value]
        else:
            value = value.strip()
        data_dict[key] = value
    return data_dict

def get_data_float_list_from_str(ele_str):
    data_str_list = ele_str.split(" ")
    data_float_list = []
    for num in data_str_list:
        if num != '':
            data_float_list.append(float(num)) 
    return data_float_list

def get_np_matrix_from_str(row, col, ele_str): 
    ele_list = get_data_float_list_from_str(ele_str)
    my_matrix = np.array(ele_list).reshape((row, col))
    return my_matrix

def write_scene_to_mcap(data_root, filepath, sub_label_path, delete_files):

    ins_file = os.path.join(data_root,"ins",*sub_label_path.split("/")[2:-1],"ins.json")
    # ins_file = 'ins/' + sub_label_path.split('/', 1)[1] + "/ins.json" # 以第一个斜杠分割路径
    ins_infos = read_ins(ins_file)

    os.path.join(data_root,"lidar",*sub_label_path.split("/")[2:])
    lidar_path = os.path.join(data_root,"lidar",*sub_label_path.split("/")[2:])

    label_jsons = os.listdir(sub_label_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(label_jsons), leave=False)

    with open(filepath,"wb") as fp:
        print(f"writing to {filepath}")
        writer = Writer(fp, compression=CompressionType.LZ4)
        protobuf_writer = ProtobufWriter(writer)
        writer.start(profile="", library="nuscenes2mcap")

        for label_json in label_jsons:
            pbar.update(1)
            
            if label_json in delete_files:
                continue

            label_file = os.path.join(sub_label_path,label_json)
            sub_lidar_path = label_file.replace(".json", "")

            Temp = sub_lidar_path.replace('data/xml', 'data/lidar')
            images_path = os.path.join(Temp,"images")
            merger_path = os.path.join(Temp,"merge_lidar")
            param_path = os.path.join(Temp,"params")
            lidar_params_file = os.path.join(Temp,"params","lidar_params.txt")
            with open(label_file,"rb") as fp:
                infos = json.load(fp)
            LidarObjects = infos["LidarObjects"]["objects"]     # ->List
            ImageObjects = infos["ImageObjects"]["objects"]     # ->dict  六个键代表相机，每个相机的value是个list
            CamTimestamps = infos["ImageObjects"]["frameInfo"]   # 6个键值表示6个相机，每个相机字段对应readtime和unixtime,19位
            LidarTime = infos["LidarObjects"]["frameInfo"]["mergelidar"]["unixTime"]   # 19位时间戳
            lidar_ins_pos,lidar_ins_euler = get_closest_ins(ins_infos,int(LidarTime))
            rotation_matrix = R.from_euler('XYZ', lidar_ins_euler.tolist()).as_matrix()
            lidar_q = R.from_matrix(rotation_matrix).as_quat()  # [x,y,z,w]
            lidar_ego2global_translation =  lidar_ins_pos.tolist()  # list
            lidar_ego2global_rotation =lidar_q.tolist()[-1:] + lidar_q.tolist()[:-1]  
            lidar2ego_translation = [0,0,0]
            lidar2ego_rotation = [1,0,0,0]
            lidar_params_dic = read_lidar_params_txt(lidar_params_file)
            stamp = get_time(int(LidarTime[:-3]))

            # 储存当前帧不同视野每个2D box与lidar box的id映射
            cur_2dID_3dID_map = {"camblf":{},    # {2D id:3D id}
                                 "camfcf":{},
                                 "cambrf":{},
                                 "camflb":{},
                                 "cambcb":{},
                                 "camfrb":{}}
            for lidarobject in LidarObjects:
                for cam_name in cam_name_list:
                    if lidarobject["fusion_id"][cam_name]["Status"] == "0":  # 出现在该视野中
                        cur_Id = lidarobject["fusion_id"][cam_name]["Id"]
                        cur_2dID_3dID_map[cam_name][cur_Id] = lidarobject["id"]
            # tf
            protobuf_writer.write_message("/tf", get_ego_tf(int(LidarTime[:-3]),lidar_ego2global_translation, \
                                                            lidar_ego2global_rotation), stamp.to_nsec())
            
            # publish /MERGE_LIDAR
            lidar_file = os.path.join(merger_path,os.listdir(merger_path)[0])
            msg = get_pcd(lidar_file, int(LidarTime[:-3]), "MERGR_LIDAR")
            protobuf_writer.write_message("/MERGR_LIDAR", msg, stamp.to_nsec())
            protobuf_writer.write_message("/tf", get_lidar_tf("MERGR_LIDAR", int(LidarTime[:-3]),\
                                                                lidar2ego_translation,lidar2ego_rotation), stamp.to_nsec())
            
            LidarAnn_sceneUpdate = SceneUpdate()
            scene_update_ID = SceneUpdate()
            scene_update_Head = SceneUpdate()
            # lidar 3D框发布
            for lidarobject in LidarObjects:
                delete_entity_all = LidarAnn_sceneUpdate.deletions.add()
                delete_entity_all.type = 1
                delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 100)

                entity = LidarAnn_sceneUpdate.entities.add()
                metadata = entity.metadata.add()
                metadata.key = "category"
                metadata.value = lidarobject["type"]
                if lidarobject["type"].lower() not in cate_list:
                    cate_list.append(lidarobject["type"].lower())
                entity.frame_id = "base_link"
                entity.timestamp.FromNanoseconds(stamp.to_nsec())
                entity.id =  str(lidarobject["id"])
                entity.frame_locked = True
                cube = entity.cubes.add()

                Q =  R.from_euler('z',lidarobject["yaw"], degrees=False).as_quat()  # [x,y,z,w]
                cube.pose.position.x = lidarobject["x"]
                cube.pose.position.y = lidarobject["y"]
                cube.pose.position.z = lidarobject["z"]
                cube.size.x = lidarobject["length"]
                cube.size.y = lidarobject["width"]
                cube.size.z = lidarobject["height"]
                cube.pose.orientation.w = Q[3]
                cube.pose.orientation.x = Q[0]
                cube.pose.orientation.y = Q[1]
                cube.pose.orientation.z = Q[2]
                c = get_color(cate_list.index(lidarobject["type"].lower()))
                cube.color.r = c[0]/255
                cube.color.g = c[1]/255
                cube.color.b = c[2]/255
                cube.color.a = 0.5

                # publish ID text
                ID_delete_entity_all = scene_update_ID.deletions.add()
                ID_delete_entity_all.type = 1
                ID_delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                ID_entity = scene_update_ID.entities.add()
                ID_entity.frame_id = "base_link"
                ID_entity.timestamp.FromNanoseconds(stamp.to_nsec())
                ID_entity.id = str(lidarobject["id"])
                ID_entity.frame_locked = True
                texts = ID_entity.texts.add()
                texts.pose.position.x = lidarobject["x"]
                texts.pose.position.y = lidarobject["y"]
                texts.pose.position.z = lidarobject["z"]
                texts.pose.orientation.w = Q[3]
                texts.pose.orientation.x = Q[0]
                texts.pose.orientation.y = Q[1]
                texts.pose.orientation.z = Q[2]
                texts.font_size = 3
                texts.text = str(lidarobject["id"])

                # publish Heading
                pred_heading_entity = scene_update_Head.entities.add()
                pred_heading_delete_entity_all = scene_update_Head.deletions.add()
                pred_heading_delete_entity_all.type = 1
                pred_heading_delete_entity_all.timestamp.FromNanoseconds(stamp.to_nsec() + 100)
                pred_heading_entity.id = str(lidarobject["id"])
                pred_heading_entity.frame_id = "base_link"
                pred_heading_entity.timestamp.FromNanoseconds(stamp.to_nsec())
                pred_heading_entity.frame_locked = True
                Line_heading = pred_heading_entity.arrows.add()
                Line_heading.color.r = 0
                Line_heading.color.g = 0
                Line_heading.color.b = 0
                Line_heading.color.a = 0.4
                Line_heading.pose.position.x = lidarobject["x"]
                Line_heading.pose.position.y = lidarobject["y"]
                Line_heading.pose.position.z = lidarobject["z"] + lidarobject["height"]/2
                Line_heading.pose.orientation.w = Q[3]
                Line_heading.pose.orientation.x = Q[0]
                Line_heading.pose.orientation.y = Q[1]
                Line_heading.pose.orientation.z = Q[2]
                Line_heading.shaft_length = lidarobject["length"]/2
                Line_heading.shaft_diameter = 0.1
                Line_heading.head_length = 0.01
                Line_heading.head_diameter = 0.01

            protobuf_writer.write_message("/markers/GT_heading", scene_update_Head, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/GT_ID", scene_update_ID, stamp.to_nsec()) 
            protobuf_writer.write_message("/markers/annotations", LidarAnn_sceneUpdate, stamp.to_nsec())

            protobuf_writer.write_message(
                "/markers/car", get_car_scene_update(stamp.to_nsec()), stamp.to_nsec()
            )

            for cam_name in cam_name_list:
                CamTimestamp = CamTimestamps[cam_name]["unixTime"]
                ImageBoxs = ImageObjects[cam_name]          # List
                # ImageAnnotation
                write_boxes_image_annotations(protobuf_writer, ImageBoxs, cam_name, int(CamTimestamp[:-3]), cur_2dID_3dID_map)

                sensor2lidar_t = np.array(lidar_params_dic["cam_"+cam_name[3:]+"_to_velodyne64"][:3])
                sensor2lidar_r = R.from_quat(np.array(lidar_params_dic["cam_"+cam_name[3:]+"_to_velodyne64"][3:])).as_matrix()

                CurCam2ego_translation =  -np.linalg.inv(sensor2lidar_r) @ sensor2lidar_t.tolist()
                CurCam2ego_rotation = np.roll(R.from_matrix(np.linalg.inv(sensor2lidar_r)).as_quat().tolist(),1).tolist()
                cur_param_xml_file_path = os.path.join(Temp,"params","para_"+cam_name+".xml")
                cur_etree = ET.parse(cur_param_xml_file_path)
                cur_root = cur_etree.getroot()
                CurCamIntrinsic = get_np_matrix_from_str(3, 3, cur_root.findall('new_mat_fisheye')[0].findall('data')[0].text)
                CurCamDist_coeff = get_data_float_list_from_str(cur_root.findall('distortion_coefficients')[0].findall('data')[0].text)                
                topic = "/" + cam_name
                protobuf_writer.write_message("/tf", get_sensor_tf(cam_name, int(CamTimestamp[:-3]), CurCam2ego_translation, CurCam2ego_rotation), stamp.to_nsec())

                cam_data_path =  os.path.join(images_path,label_json.replace("lidar.json", cam_name+".jpg"))
                msg = get_camera(cam_data_path ,int(CamTimestamp[:-3]), cam_name)
                protobuf_writer.write_message(topic + "/image_rect_compressed", msg, stamp.to_nsec())
                image = Image.open(cam_data_path)
                H,W = np.array(image).shape[:2]

                msg = get_camera_info(int(CamTimestamp[:-3]),cam_name, H, W, CurCamIntrinsic, CurCamDist_coeff)
                protobuf_writer.write_message(topic + "/camera_info", msg, stamp.to_nsec())
        pbar.close()
        writer.finish()
        print(f"Finished writing {filepath}")

# 找到image不全的帧
def get_case1(data_dir,output_dir, version):
    root_path = os.path.join(data_dir,"xml")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq))

    for cur_seq in sub_seqs:
        label_jsons = sorted(os.listdir(cur_seq))
        for label_json in label_jsons:
            lidar_sub = label_json.replace(".json","")
            lidar_root_path = cur_seq.replace("data/xml","data/lidar")
            lidar_file = os.path.join(lidar_root_path, lidar_sub)
            if os.path.exists(lidar_file):
                infos = os.listdir(os.path.join(lidar_root_path,lidar_sub))
                if  "images" not in infos or \
                    "merge_lidar" not in infos or \
                    "params"  not in infos or \
                    len(os.listdir(os.path.join(lidar_root_path,lidar_sub,"images"))) != 6 or \
                    len(os.listdir(os.path.join(lidar_root_path,lidar_sub,"merge_lidar"))) != 1 or \
                    len(os.listdir(os.path.join(lidar_root_path,lidar_sub,"params"))) != 7:
                    data_to_append = {
                        "case1": os.path.join(cur_seq,label_json)
                    }
                    # 以追加模式打开文件，并写入新的数据
                    os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                    with open(os.path.join(output_dir, version, 'case1.json'), 'a') as json_file:
                        json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                        json.dump(data_to_append, json_file, ensure_ascii=False)
                    
def get_case2(data_dir,output_dir, version):
    """
    # “ImageObjects”有两个键,“frmaeInfo”和"objects"。其中“objects”为6路相机的标注框，"frameInfo" 需要提供6路相机的“readtime”和“unixtime”
    # 但是数据本身的问题,导致 "ImageObjects" -> “frameInfo”  有些相机给的是标注框信息，没有提供unixtime
    # 需要筛选出对应的数据,通过label json文件筛选
    """
    target_keys = ["readTime","unixTime"]
    root_path = os.path.join(data_dir,"xml")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq))   
            
    for cur_seq in sub_seqs:
        lidar_jsons = sorted(os.listdir(cur_seq))
        for lidar_json in lidar_jsons:
            if  lidar_json[-4:] != "json":
                continue
            with open(os.path.join(cur_seq,lidar_json),"r") as fp:
                infos = json.load(fp)
            cam_framinfo_keys = infos["ImageObjects"]["frameInfo"].keys()
            for cam_key in cam_framinfo_keys:   
                if target_keys[0] not in infos["ImageObjects"]["frameInfo"][cam_key] or \
                    target_keys[1] not in infos["ImageObjects"]["frameInfo"][cam_key]:
                        data_to_append = {
                            "case2": os.path.join(cur_seq,lidar_json)
                        }
                        # 以追加模式打开文件，并写入新的数据
                        os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                        with open(os.path.join(output_dir, version, 'case2.json'), 'a') as json_file:
                            json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                            json.dump(data_to_append, json_file, ensure_ascii=False)
                            break
                else:
                    continue 

def get_case3(data_dir, output_dir, version):
    """
    # “ImageObjects”有两个键,“frmaeInfo”和"objects"。其中“objects”为6路相机的标注框,"frameInfo" 需要提供6路相机的“readtime”和“unixtime”
    # 常规操作下,若某路相机没有出现物体,应该给一个value为空的字段,但是有些对于没有的出现目标的视野,对应的键值字段缺失(与frameInfo一起消失)
    # 筛出该部分的数据
    """
    cam_lists = ["camfcf","camfrb","cambrf","cambcb","camblf","camflb"]
    root_path = os.path.join(data_dir,"xml")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq))  
            
    for cur_seq in sub_seqs:
        lidar_jsons = sorted(os.listdir(cur_seq))
        for lidar_json in lidar_jsons:
            if  lidar_json[-4:] != "json":
                continue
            with open(os.path.join(cur_seq,lidar_json),"r") as fp:
                infos = json.load(fp)
            cam_frameinfo_keys = infos["ImageObjects"]["frameInfo"].keys()
            cam_objects_keys = infos["ImageObjects"]["objects"].keys()
            for cam_key in cam_lists:
                if cam_key not in cam_frameinfo_keys or cam_key not in cam_objects_keys:
                    data_to_append = {
                        "case3": os.path.join(cur_seq,lidar_json)
                    }
                    # 以追加模式打开文件，并写入新的数据
                    os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                    with open(os.path.join(output_dir, version, 'case3.json'), 'a') as json_file:
                        json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                        json.dump(data_to_append, json_file, ensure_ascii=False)
                        break    
                else:
                    continue 

def get_case4(data_dir,output_dir, version):
    """
    # 标注文件中的LidarObjects->obejcts->fusion_id中  使用字段“Status”字段给出了某个3D object出现在的视野信息
    # 但是部分数据的fusion_id不全,外在表现是有些3D物体A, 只给出了是否出现在4个相机,没有给出关于另外两个相机的信息。
    """
    cam_lists = ["camfcf","camfrb","cambrf","cambcb","camblf","camflb"]
    root_path = os.path.join(data_dir,"xml")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq))  

    for cur_seq in sub_seqs:
        lidar_jsons = sorted(os.listdir(cur_seq))
        for lidar_json in lidar_jsons:
            if  lidar_json[-4:] != "json":
                continue
            with open(os.path.join(cur_seq,lidar_json),"r") as fp:
                infos = json.load(fp)
            LidarObjects = infos["LidarObjects"]["objects"]
            break_flag = False
            for LidarObject in LidarObjects:
                for cur_key in cam_lists:
                    if cur_key not in LidarObject["fusion_id"].keys():
                        data_to_append = {
                            "case4": os.path.join(cur_seq,lidar_json)
                        }
                        # 以追加模式打开文件，并写入新的数据
                        os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                        with open(os.path.join(output_dir, version, 'case4.json'), 'a') as json_file:
                            json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                            json.dump(data_to_append, json_file, ensure_ascii=False)
                            break 
                    else:
                        continue
                if break_flag:
                    break

def get_case5(data_dir,output_dir, version):
    """
    # LidarObjects->的坐标属性“xyzwlh yaw”   出现null
    """
    root_path = os.path.join(data_dir,"xml")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq)) 
            
    for cur_seq in sub_seqs:
        lidar_jsons = sorted(os.listdir(cur_seq))
        for lidar_json in lidar_jsons:
            if lidar_json[-4:] != "json":
                continue
            with open(os.path.join(cur_seq,lidar_json),"r") as fp:
                infos = json.load(fp)
            LidarObjects = infos["LidarObjects"]["objects"]

            for LidarObject in LidarObjects:
                if LidarObject["length"] == None or \
                    LidarObject["width"] == None or \
                    LidarObject["height"] == None or \
                    LidarObject["x"] == None or \
                    LidarObject["y"] == None or \
                    LidarObject["z"] == None or \
                    LidarObject["yaw"] == None:
                    data_to_append = {
                        "case5": os.path.join(cur_seq,lidar_json)
                    }
                    # 以追加模式打开文件，并写入新的数据
                    os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                    with open(os.path.join(output_dir, version, 'case5.json'), 'a') as json_file:
                        json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                        json.dump(data_to_append, json_file, ensure_ascii=False)
                        break 

def get_case6(data_dir, output_dir, version):
    """
    部分label文件名字与lidar文件夹下的每一帧数据文件夹名字不同
    """
    root_path = os.path.join(data_dir,"xml")
    ImageLidar_root = os.path.join(data_dir,"lidar")
    dirs = os.listdir(root_path)
    sub_seqs = []
    for dir in dirs:
        if "5001" in dir:
            continue
        sub_dir = os.path.join(root_path,dir)
        cur_seqs = os.listdir(sub_dir)
        for cur_seq in cur_seqs:
            sub_seqs.append(os.path.join(sub_dir,cur_seq)) 

    for sub_seq in sub_seqs:
        lidar_path = os.path.join(ImageLidar_root,*sub_seq.split("/")[2:])
        label_files = os.listdir(sub_seq)
        for label_file in label_files:
            file_name_prefix = label_file.replace("_lidar.json", "")
            lidar_file_name = file_name_prefix + "_mergelidar.pcd" 
            lidar_path1 = os.path.join(lidar_path,file_name_prefix+"_lidar","merge_lidar",lidar_file_name)
            if not os.path.exists(lidar_path1):
                data_to_append = {
                    "case6": os.path.join(sub_seq, label_file)
                }
                # 以追加模式打开文件，并写入新的数据
                os.makedirs(os.path.join(output_dir, version),exist_ok=True)
                with open(os.path.join(output_dir, version, 'case6.json'), 'a') as json_file:
                    json_file.write('\n')  # 添加换行符以确保新数据单独占一行
                    json.dump(data_to_append, json_file, ensure_ascii=False) 

# 过滤格式bug
def filter_case(args, version):
    print("=========start case1 filter=========")
    get_case1(args.data_dir,args.output_dir, version)
    print("=========start case2 filter=========")
    get_case2(args.data_dir,args.output_dir, version)
    print("=========start case3 filter=========")  
    get_case3(args.data_dir,args.output_dir, version)
    print("=========start case4 filter=========")
    get_case4(args.data_dir,args.output_dir, version)
    print("=========start case5 filter=========")
    get_case5(args.data_dir,args.output_dir, version)
    print("=========start case6 filter=========")
    get_case6(args.data_dir,args.output_dir, version)
    print("=========end filter =========")

def convert_all(
    data_root,
    output_dir: Path,
    sub_seq_name,
    delete_files,
):
    scene_name = os.path.basename(sub_seq_name)
    mcap_name = f"kanhuAIV-{scene_name}-pkl_1hz.mcap"
    write_scene_to_mcap(data_root, output_dir / mcap_name, sub_seq_name,delete_files)

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument("--data-dir", "-d", default=script_dir / "data", help="path to aiv5 data directory")
    parser.add_argument("--output-dir", "-o", type=Path, default=script_dir / "output", help="path to write MCAP files into")

    parser.add_argument("--version", type=str, default=" ", help= "v0:原始版本数据的格式清洗,将格式没有问题的数据转为mcap,支持标注验收(读取整个数据文件夹),\
                                                                   v1:返修格式问题的数据验收,转化因格式问题没有进行标注验收的数据为mcap,进行标注验收(读取对应的格式小包数据文件夹)\
                                                                   v2:返修标注问题的数据验收,将标注问题的返修转为mcap,进行二次验收(读取对应的标注小包数据文件)")
    
    ##########  标注问题记为一个case_label.json返给数据组 ##############
    args = parser.parse_args()
    assert args.version != " ", print("please choose a mode:v0 or v1 or v2")

    # filter_case(args, args.version)
    json_files = glob.glob(os.path.join(args.output_dir, args.version) + "/" + "/*.json")
    case_files = set() 
    for json_file in json_files:
        with open(json_file, 'rb') as file:
            data = file.readlines()  # 逐行读取文件内容
        for line in data[1:]:
            json_data = json.loads(line)
            label_name = os.path.basename(list(json_data.values())[0])
            if label_name[-5:] != ".json":
                label_name = label_name+".json"
            case_files.add(label_name)
    print(len(case_files))
    root_path =  os.path.join(args.data_dir,"xml")
    sub_seq_names = []
    Ls = os.listdir(root_path)
    for L in Ls:
        if "5001" in L:
            continue
        cur_root_path = os.path.join(root_path,L)
        for sub_cur_root_path in os.listdir(cur_root_path):
            sub_seq_names.append(os.path.join(root_path,L,sub_cur_root_path))
    for sub_seq_name in sub_seq_names:
        convert_all(args.data_dir, args.output_dir, sub_seq_name, case_files)
        
if __name__ == "__main__":
    main()
