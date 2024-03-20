# Release Notes


### Improvements

### Fixes

### Refactoring


### Testing



## WIP
---
## 4.0
### Improvements
* 添加input_json文件夹，用于存储各种需要的json文件
* 添加高亮转换工具，将评估工具箱输出的几个json，结合streampetr网络的跟踪预测json结果转化为mcap
* 新拉分支，去除本分支工具箱不需要的文件和脚本
* 加入四个json文件(来自streampetr的0-50m范围, 全视野的评估结果)
* 增加对比真值速度曲线和预测速度曲线的layout_config
### Fixes

### Refactoring
* 重构convert_to_mcap_pred_DishHighlight_guanlian.py 脚本， 并使其可以支持ego系和global系的观察
