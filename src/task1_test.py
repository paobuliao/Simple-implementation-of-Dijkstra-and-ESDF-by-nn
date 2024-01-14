import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 第一个卷积层，批标准化
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(4)

        # 第二个卷积层，批标准化
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(16)

        # 全连接层
        self.fc1 = nn.Linear(16 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 64 * 64)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)  # Assuming input size is 1x64x64
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = x.view(-1, 16 * 64 * 64)  # Reshape for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Assuming output is between 0 and 1
        return x.view(-1, 64, 64)


# 定义数据集类
def get_data(csv_file_path):
    X=[]
    Y=[]
    with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0].startswith('Map_'):
                    X.append([])
                    Y.append([])
                elif row[0].startswith('Mapsg'):
                    # 读取地图
                    # 接下来的64行是地图的数据
                    for i in range(64):
                        row = next(reader)
                        X[-1].append(np.array(row, dtype=np.float32))
                elif row[0].startswith('Path'):
                    # 读取路径
                    # 接下来的64行是路径的数据
                    for i in range(64):
                        row = next(reader)
                        Y[-1].append(np.array(row, dtype=np.float32))
                else:
                    continue
    return np.array(X),np.array(Y)

if __name__ == '__main__':
    # 指定CSV文件路径
    # 获取当前脚本所在的目录
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)

    # 上一级目录下的文件路径
    csv_file_path = os.path.join(parent_directory, 'all_maps_and_paths.csv')
    # 创建数据集和数据加载器
    X, Y = get_data(csv_file_path)
    # 数据格式变换
    X = X.reshape(-1, 64, 64)
    Y = Y.reshape(-1, 64, 64)
    # 转为tensor并移动到GPU
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
    model = MyModel().to(device)
    #加载momel配置
    # 在上一级文件夹
    ckpt_path=os.path.join(parent_directory, 'task1_model.ckpt')
    model.load_state_dict(torch.load(ckpt_path))
    print(model)
    
    fig_folder_path=os.path.join(parent_directory, 'figs')
    # 测试
    model.eval()
    with torch.no_grad():
        # 随机选取一些测试样本
        num_test=10
        for i in range(num_test):
            idx = np.random.randint(0, len(X_test))
            map = X_test[idx]
            path = Y_test[idx]
            
            #计算推理耗时
            start_time=time.time()
            output = model(map)
            end_time=time.time()
            print('推理耗时：',end_time-start_time)
            #计算loss
            path = path.view(-1, 64, 64)
            output = output.view(-1, 64, 64)
            loss = F.binary_cross_entropy(output, path)
            print('loss:',loss)

            # 可视化结果
            # 画矩阵
            plt.subplot(1, 3, 1)
            #数据转为 64*64
            plt_map=map.cpu().numpy().reshape(64,64)
            plt.imshow(plt_map, cmap='gray')
            plt.title('Map')
            plt.subplot(1, 3, 2)
            plt_path=path.cpu().numpy().reshape(64,64)
            plt.imshow(plt_path, cmap='gray')
            plt.title('Ground Truth')
            plt.subplot(1, 3, 3)
            plt_output=output.cpu().numpy().reshape(64,64)
            plt.imshow(plt_output, cmap='gray')
            plt.title('Prediction')
            fig_path=os.path.join(fig_folder_path, 'task1_fig_'+str(i)+'.png')
            plt.savefig(fig_path,dpi=600)
            plt.show()