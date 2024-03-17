sudo docker run -ti -v $(pwd)/data_link:/work/data -v $(pwd)/output:/work/output  uto_nuscenes2mcap_tools /bin/bash
# data_link为原始数据所在路径，可以是软连接
