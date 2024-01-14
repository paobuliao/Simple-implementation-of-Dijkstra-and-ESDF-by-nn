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



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        # Custom loss implementation
        pixel_loss = F.mse_loss(predictions, targets, reduction='none')

        # Apply your custom conditions for values -1 and 255
        invalid_mask = ((targets == 255) & (predictions != 255)) | ((targets == -1) & (predictions != -1))

        # Penalty for isolated 255 points
        isolated_255_penalty = self.isolated_255_penalty(targets)

        # Combine the losses with your desired weights
        total_loss = pixel_loss + isolated_255_penalty

        return total_loss.mean()

    def isolated_255_penalty(self, targets):
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=targets.device)
        kernel[0, 0, 1, 1] = 0  # Set the center element to 0, keeping surrounding elements as 1

        dilated_targets = F.conv2d(targets.to(torch.float32).unsqueeze(1), kernel, padding=1)

        # Create a mask for isolated 255 points
        isolated_255_mask = (targets == 255) & (dilated_targets == 0)

        # Calculate penalty as the count of isolated 255 points
        penalty = isolated_255_mask.float().sum()

        return penalty


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
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
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
    
    # 将数据移动到GPU
    X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


    model = MyModel().to(device)
    #加载momel配置
    ckpt_path=os.path.join(parent_directory, 'task1_model.ckpt')
    # model.load_state_dict(torch.load(ckpt_path))
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # loss_function = CustomLoss()

    loss_list = []
    # 训练模型
    start=time.time()
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (maps, paths) in enumerate(train_loader):
            # 前向传播
            outputs = model(maps)

            # 计算损失
            loss = criterion(outputs, paths)


            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 保存损失
            

            # 打印损失
            if (i+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            #每50次保存一次模型
            if (i+1) % 50 == 0:
                torch.save(model.state_dict(), ckpt_path)
        end=time.time()
        print('epoch耗时：',end-start)
        start=end
        loss_list.append(loss.item())
    #保存模型
    torch.save(model.state_dict(), ckpt_path)

    # 绘制loss
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    fig_path=os.path.join(parent_directory, 'task1_loss_curve.png')
    plt.savefig(fig_path)
    plt.show()


    # 测试
    model.eval()
    with torch.no_grad():
        # 随机选择一个测试样本
        for i in range(5):
            idx = np.random.randint(0, len(X_test))
            map = X_test[idx]
            path = Y_test[idx]
            
            #计算推理耗时
            start_time=time.time()
            output = model(map)
            end_time=time.time()
            print('推理耗时：',end_time-start_time)
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

            plt.show()