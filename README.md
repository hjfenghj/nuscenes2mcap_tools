# Release Notes
We have the following change categories:
### Improvements
* 添加conert_raw2hzpkl_to_mcap.py脚本，用于完成原版streampetr对应的原版2hz pkl文件转化为mcap
### Fixes

### Refactoring


### Testing






## WIP
---
## 2.0
### Improvements
* 添加轻量化脚本convert_to_mcap_lite.py, 仅转化2hz标注数据. 去除地图,车道线等元素
* 在轻量化脚本中加入轨迹可视化, 可用于观察真值ID是否发生跳变
* 加入车身朝向可视化和速度箭头可视化
* 分离GT框可视化和ID可视化
* 在protubuf中存入速度大小，用于后续拉速度曲线
* 增加读取pcd点云文件转化为PointCloud格式消息的函数
* 替换布局layout_config文件
### Refactoring
* 重构conert_to_mcap_lite.py, 规整变量命名等



## 1.0
### Improvements
* 加载nuscenes数据集完成mcap格式的转换
* 可选单场景转换和全场景转换
* 可选不同数据split(v1.0-mini, v1.0-trainval等)