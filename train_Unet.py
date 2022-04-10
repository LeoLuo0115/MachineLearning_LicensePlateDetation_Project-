import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import random
import PIL.Image as Image
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Unet1 import Unet1
import matplotlib.pyplot as plt

# 按批次取数据
class CAR_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(self.data_path+'/image/*.jpg')
        self.label_path = glob.glob(self.data_path+'/label/*.jpg')
        # 数据预处理，输入图片大小一致
        self.trans = transforms.Compose([
                                     transforms.Resize((512, 512)), #2^n
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        # 根据index读取图片地址
        image_path = self.imgs_path[index]
        # 根据index读取label图片地址
        label_path = self.label_path[index]
        # 读取训练图片和标签图片
        image = Image.open(image_path)
        label = Image.open(label_path)
        # 图片预处理
        image = self.trans(image)
        label = self.trans(label)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# 训练模型
def fit_model(batch_1,epochs_1,lr,net,save_path):
    train_loss_l = []
    val_loss_l = []
    #参数设置
    lr = lr
    batch_size = batch_1
    epochs = epochs_1
    best_loss = 10.0
    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_car_dataset = CAR_Loader("./data")
    train_num = len(train_car_dataset)



    #读取图片时的线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_car_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    val_car_dataset = CAR_Loader("./data")
    val_num = len(val_car_dataset)
    validate_loader = torch.utils.data.DataLoader(val_car_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)

    train_steps = len(train_loader)
    val_steps = len(validate_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running1_loss = 0.0
        running2_loss = 0.0
        for data in train_loader:
            images, labels = data
            optimizer.zero_grad()
            # print(images.shape)
            outputs=net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step() # 根据反向传播区队优化器更新

            # print statistics
            running1_loss += loss.item()

        train_loss_l.append(running1_loss / train_steps)
        # validate
        net.eval()
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                val_loss = loss_function(outputs, val_labels.to(device))
                running2_loss += val_loss.item()

        val_loss_l.append(running2_loss / val_steps)

        print('[epoch %d] train_loss: %.3f val_loss: %.3f  \n' %
              (epoch + 1, running1_loss / train_steps, running2_loss / val_steps))

        if (running2_loss / val_steps) < best_loss:
            best_loss = (running2_loss / val_steps)
            torch.save(net.state_dict(), save_path) #找到最优loss，保存训练的权重和地址

    p = best_loss
    np.save('val_loss.npy', val_loss_l)
    np.save('train_loss.npy', train_loss_l)
    print('Finished Training')
    return p


def main():
    epoch = 100
    batch_size = 64
    lr = 0.0001
    save_path = './model.pth'
    net1 = Unet1()
    # net1.load_state_dict(torch.load('model.pth'))
    p1 = fit_model(batch_size, epoch, lr, net1, save_path)
    print(" epoch(", epoch, ") model train_max:", p1)


if __name__ == '__main__':
    main()