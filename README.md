# Release Notes
该工具是将一款数据从原始文件夹形式转化为mcap，mcap中包含2D框，3D框，ID信息。并可以通过不同的颜色显示不同的类别。并且可以实现点云的反投影和3D框的反投影


1. 数据层级如下:
lidar: 第一级是场景标识，第二级是不同的sequence序列，每个序列包含的数据帧有长有短;
       每个序列中包含多帧，每帧包含一路激光雷达数据、6路相机数据和6路相机内外参(文件组织形式请看samll_data文件夹)

xml: xml文件夹的层级与lidar一致，为每帧数据的真值信息，包含2D真值、3D真值等信息(具体信息请看small_data文件夹)

ins: ins文件夹为每个sequence所有时间的里程计数据，需要解析完成自车系到世界系的变换

output: mcap的储存文件夹，会将每个sequence的数据保存为一个mcap文件



├── ins
│   ├── AIV_SH_#5024_20230911_QCJCJ
│   │   ├── ins.json
│   │   └── timestamp.txt
│   ├── AIV_SH_#5024_20230913_JGBD_hualong_2
│   │   ├── ins.json
│   │   └── timestamp.txt
├── lidar
│   ├── AIV_SH_#5024_20230911_QCJCJ
│   │   ├── AIV_SH_#5024_20230911_QCJCJ_0001
|   |   |   |—— QZDCV_5024_20230911-161232-512-737_lidar
|   |   |   |   ├── images
|   |   |   |   │   ├── QZDCV_5024_20230911-161232-512-737_cambcb.jpg
|   |   |   |   │   ├── QZDCV_5024_20230911-161232-512-737_camblf.jpg
|   |   |   |   │   ├── QZDCV_5024_20230911-161232-512-737_cambrf.jpg
|   |   |   |   │   ├── QZDCV_5024_20230911-161232-512-737_camfcf.jpg
|   |   |   |   │   ├── QZDCV_5024_20230911-161232-512-737_camflb.jpg
|   |   |   |   │   └── QZDCV_5024_20230911-161232-512-737_camfrb.jpg
|   |   |   |   ├── merge_lidar
|   |   |   |   │   └── QZDCV_5024_20230911-161232-512-737_mergelidar.pcd
|   |   |   |   ├── params
|   |   |   |   │   ├── lidar_params.txt
|   |   |   |   │   ├── para_cambcb.xml
|   |   |   |   │   ├── para_camblf.xml
|   |   |   |   │   ├── para_cambrf.xml
|   |   |   |   │   ├── para_camfcf.xml
|   |   |   |   │   ├── para_camflb.xml
|   |   |   |   │   └── para_camfrb.xml
|   |   |   |   └── QZDCV_5024_20230911-161232-512-737_lidar.json
|   |   |   |—— QZDCV_5024_20230911-161233-513-042_lidar
│   │   ├── AIV_SH_#5024_20230911_QCJCJ_0002
│   │   ├── AIV_SH_#5024_20230911_QCJCJ_0003
│   │   ├── AIV_SH_#5024_20230911_QCJCJ_0004
│   │   ├── AIV_SH_#5024_20230911_QCJCJ_0005
│   │   └── AIV_SH_#5024_20230911_QCJCJ_0006
│   ├── AIV_SH_#5024_20230913_JGBD_hualong_2
│   │   ├── AIV_SH#5024_20230913_JGBD_hualong_2_0001
│   │   └── AIV_SH#5024_20230913_JGBD_hualong_2_0002
├── output
│   ├── kanhuAIV-AIV_SH_#5024_20230911_QCJCJ_0001-pkl_1hz.mcap
|   |—— kanhuAIV-AIV_SH_#5024_20230911_QCJCJ_0002-pkl_1hz.mcap
│   └── v0
│       ├── case1.json
│       ├── case2.json
│       ├── case3.json
│       ├── case4.json
│       ├── case5.json
│       └── case6.json
└── xml
    ├── AIV_SH_#5024_20230911_QCJCJ
    │   ├── AIV_SH_#5024_20230911_QCJCJ_0001
    │   ├── AIV_SH_#5024_20230911_QCJCJ_0002
    │   ├── AIV_SH_#5024_20230911_QCJCJ_0003
    │   ├── AIV_SH_#5024_20230911_QCJCJ_0004
    │   ├── AIV_SH_#5024_20230911_QCJCJ_0005
    │   └── AIV_SH_#5024_20230911_QCJCJ_0006
    ├── AIV_SH_#5024_20230913_JGBD_hualong_2
    │   ├── AIV_SH#5024_20230913_JGBD_hualong_2_0001
    │   └── AIV_SH#5024_20230913_JGBD_hualong_2_0002


2. 使用说明
包含三个模式:v0,v1,v2
v0:原始版本数据的格式清洗,将格式没有问题的数据转为mcap,支持标注验收(读取整个数据文件夹),将标注格式等有问题的数据场景时间戳村为case.json,返回给数剧组
v1:返修格式问题的数据验收,转化因格式问题没有进行标注验收的数据为mcap,进行标注验收(读取对应的格式小包数据文件夹)，就是将数剧组修正好的数据，重新确认一下
v2:返修标注问题的数据验收,将标注问题的返修转为mcap,进行二次验收(读取对应的标注小包数据文件)"

这个仓乎的目的，主要是为了记录mcap中的点云反投影, 2D框绘制的方法。


3. 运行方式
进入docker镜像环境
python3 KanHu_AIV5_convert_rawdate_to_mcap_add2DBox.py --version v0 