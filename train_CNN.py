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
from CNN1 import Net
import torch.nn.functional as F
# char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
#              "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
#              "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
#              "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
#              "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
#              "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
#              "W": 61, "X": 62, "Y": 63, "Z": 64}

INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                  "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                  "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}

INDEX_LETTER = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,"H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
                "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}

with open('name_label.txt', 'r') as f:
    data_img_label = f.readlines()
    data_img = [i.split(' ')[0] for i in data_img_label]
    data_label = [i.split(' ')[1].split('\n')[0] for i in data_img_label]
    data_label = np.array([[np.asarray(i.split(','), dtype=np.float32)]for i in data_label]).reshape((-1,238))
    # print(data_label.shape)



class CAR_Loader(Dataset):
    def __init__(self, data_img, data_label):
        # 初始化函数，读取所有data_path下的图片
        self.data_img_list = data_img
        self.data_img_label = data_label
        self.trans = transforms.Compose([
                                     transforms.Resize((80, 240)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.data_img_list[index]
        image = Image.open('./CNN_img/'+image_path)
        # image.show()
        image = self.trans(image)
        label = self.data_img_label[index]
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.data_img_list)


def fit_model(batch_1,epochs_1,lr,net,save_path):
    train_loss_l = []
    val_loss_l = []
    #参数设置
    lr = lr
    batch_size = batch_1
    epochs=epochs_1
    best_loss = 10.0
    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_car_dataset = CAR_Loader(data_img, data_label)
    train_num = len(train_car_dataset)



    #读取图片时的线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_car_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    val_car_dataset = CAR_Loader(data_img, data_label)
    val_num = len(val_car_dataset)
    validate_loader = torch.utils.data.DataLoader(val_car_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr)

    train_steps = len(train_loader)
    val_steps = len(validate_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running1_loss = 0.0
        running2_loss = 0.0
        for data in train_loader:
            # loss_temp_train = 0.0
            images, labels = data
            # print(labels)
            optimizer.zero_grad()
            # print(images.shape)
            outputs=net(images.to(device))
            # print('out',outputs.shape)
            # print('labels',labels.shape)
            loss = loss_function(outputs,labels)
            # print(len(labels))
            # for i in range(7):
            #     loss_temp_train += loss_function(outputs[i], labels[i].to(device))
            # loss = [loss_temp+=loss_function(outputs[i], labels[i].to(device))  for i in range(7)]
            loss.backward()
            optimizer.step()

            # print statistics
            running1_loss += loss.item()

        train_loss_l.append(running1_loss / train_steps)
        # validate
        net.eval()
        with torch.no_grad():
            for val_data in validate_loader:
                # loss_temp_val = 0.0
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                # val_loss = loss_function(outputs, val_labels.to(device))
                # for i in range(7):
                #     loss_temp_val += loss_function(outputs[i], val_labels[i].to(device))
                val_loss = loss_function(outputs,val_labels)
                running2_loss += val_loss.item()

        val_loss_l.append(running2_loss / val_steps)
        print('[epoch %d] train_loss: %.3f val_loss: %.3f  \n' %
              (epoch + 1, running1_loss / train_steps, running2_loss / val_steps))

        if (running2_loss / val_steps) < best_loss:
            best_loss = (running2_loss / val_steps)
            torch.save(net.state_dict(), save_path)

    p = best_loss
    np.save('val_loss.npy', val_loss_l)
    np.save('train_loss.npy', train_loss_l)
    print('Finished Training')
    return p


def main():
    epoch = 100
    batch_size = 12
    lr = 0.0001
    save_path = './CNN.pth'
    net1 = Net()
    # net1.load_state_dict(torch.load('model.pth'))
    p1 = fit_model(batch_size, epoch, lr, net1, save_path)
    print(" epoch(", epoch, ") model train_max:", p1)

if __name__ == '__main__':
    main()
