# Release Notes
该工具箱用于评估streampetr的网络迭代结果，输入两个不同的tracking_results.json文件，输出包含真值框和两个不同的预测结果(速度值, 速度方向和预测框位置)
另外，支持选定场景和目标Case，可以定向比较某个场景的某个目标的迭代效果

json_input文件夹下有五个文件 case_v.json, case_pos.json, case.json, tracking_result_base.json和tracking_result_update.json;
tracking_result_base.json为streampetr网络的24e基线网络在本地3090服务器上跑出来的结果。
tracking_result_update.json为加入深度图监督后的结果, 也就是网络迭代后的结果
case.json, case_v.json和case_pos.json三个文件可以选定预计对比的场景case和case目标, 收参数use_case的控制.

脚本参数: 
use_case:  表示是否只输出指定场景mcap,以及是否完成指定目标的高亮。 若为false, 则输出全场景的对比
result_json_name: 基线网络结果
result_json_name1: 迭代网络结果
case_json_name: case文件选择, case_v.json表示对比速度的目标case文件, case_pos.json表示对比位置的目标case文件，case.json表示全部case


### Improvements



### Fixes

### Refactoring

### Testing



## WIP
---
