# Simple-implementation-of-Dijkstra-and-ESDF-by-nn
use neural network to implement Dijkstra and ESDF with matrix as input
1. src\task1_nn.py: 实现任务1的神经网络
2. src\task2_nn.py: 实现任务2的神经网络
3. src\task2_nn_for_compare.py: 使用为未调整障碍物区域值的数据进行训练，用于效果对比
4. src\task1_test.py: 测试任务1
5. src\task2_test.py: 测试任务2
6. src\task2_test_for_compare: 测试使用为未调整障碍物区域值的数据进行训练的神经网络，用于对比
7. taks1_model.ckpt: 任务1训练后的网络参数
8. model_esdf.ckpt: 任务2训练后的网络参数
9. model_esdf_0.ckpt: 任务2中用于对比的网络参数
10. all_maps_and_paths.csv: 任务1中生成的数据，可重新生成
11. esdf_0.csv: 任务2中使用的数据
12. esdf_100.csv: 任务2中使用的数据，障碍物区域经过了调整
13. path.py: 用于生成任务1的数据
14. esdf.py: 用于生成任务2的数据，通过修改135行最后一个参数，可以设置是否生成进行特殊处理的数据。还需注意需修改126行中的文件名。
需保持csv，ckpt文件位于代码的上一级目录。
如果有任何问题，请随时与我联系。
