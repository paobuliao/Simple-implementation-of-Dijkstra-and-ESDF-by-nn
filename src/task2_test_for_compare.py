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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # 第一个卷积层，批标准化，池化
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # 第二个卷积层，批标准化，池化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 64 * 64)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)  # Assuming input size is 1x64x64
        x = self.pool1(F.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, 32 * 16 * 16)  # Reshape for fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 64, 64) 
        return x




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
                    for i in range(64):
                        row = next(reader)
                        X[-1].append(np.array(row, dtype=np.float32))
                elif row[0].startswith('Esdf'):
                    for i in range(64):
                        row = next(reader)
                        Y[-1].append(np.array(row, dtype=np.float32))
                else:
                    continue
    return np.array(X),np.array(Y)

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    # 指定CSV文件路径
    csv_file_path = os.path.join(parent_directory, 'esdf_0.csv')

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
    # 随机种子固定确保每次划分结果一致
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 将数据移动到GPU
    X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


    model = MyModel().to(device)
    #加载momel配置
    ckpt_path=os.path.join(parent_directory, 'model_esdf_0.ckpt')
    model.load_state_dict(torch.load(ckpt_path))
    print(model)

    # 测试
    model.eval()
    with torch.no_grad():
        # 随机选择一个测试样本
        for i in range(5):
            idx = np.random.randint(0, len(X_test))
            map = X_test[idx]
            esdff = Y_test[idx]
            
            #计算推理耗时
            start_time=time.time()
            output = model(map)
            end_time=time.time()
            
            # 计算loss,使用mse
            output=output.view(-1,64,64)
            esdff=esdff.view(-1,64,64)
            loss = F.mse_loss(output, esdff)
            print('loss1:',loss)
            
            print('推理耗时：',end_time-start_time)
            # 可视化结果
            # 设置图像大小
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            #数据转为 64*64
            plt_map=map.cpu().numpy().reshape(64,64)
            plt.imshow(plt_map)
            plt.title('Map',fontsize=10)
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.colorbar()
            # 画路径
            plt.subplot(1, 3, 2)
            plt_esdf=esdff.cpu().numpy().reshape(64,64)
            plt.imshow(plt_esdf)
            plt.title('Ground Truth',fontsize=10)
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.colorbar()
            # 画路径
            plt.subplot(1, 3, 3)
            plt_output=output.cpu().numpy().reshape(64,64)
            plt.imshow(plt_output)
            plt.title('Output',fontsize=10)
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.colorbar()
            fig_path=os.path.join(parent_directory, 'figs', 'task2_compare_'+str(i)+'.png')
            plt.savefig(fig_path,dpi=600)
            plt.show()