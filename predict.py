import torch
from torchvision import transforms
import PIL.Image as Image
from Unet1 import Unet1
import numpy as np
import cv2
from CNN1 import Net
import matplotlib.pyplot as plt

def locate_and_correct(img_src, img_mask):


    contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # contours1长度为0说明未检测到车牌
        print("未检测到车牌")
        return [], []
    else:
        Lic_img = []
        img_src_copy = img_src.copy()  # img_src_copy用于绘制出定位的车牌轮廓
        for ii, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # 获取最小外接矩形
            img_cut_mask = img_mask[y:y + h, x:x + w]  # 将标签车牌区域截取出来
            # img_cut_ = img_src[y:y + h, x:x + w]  # 将标签车牌区域截取出来
            # print(img_cut_mask.shape)
            # contours中除了车牌区域可能会有宽或高都是1或者2这样的小噪点，
            # 而待选车牌区域的均值应较高，且宽和高不会非常小，因此通过以下条件进行筛选
            if np.mean(img_cut_mask) >= 60 and w > 15 and h > 15:
                rect = cv2.minAreaRect(cont)  # 针对坐标点获取带方向角的最小外接矩形，中心点坐标，宽高，旋转角度
                box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标  [左上，右上，右下, 左下]

                box = sorted(box, key=lambda xy:xy[0])
                box_left, box_right = box[:2], box[2:]
                box_left = sorted(box_left, key=lambda  x:x[1])
                box_right = sorted(box_right, key=lambda x:x[1])
                box = np.array(box_left+box_right)

                x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]



                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
                print(l0, l1, l2, l3)
                p0 = np.float32([l0, l1, l2, l3])  # 左上角，左下角，右上角, 右下角，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # 我们所需的长方形
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
                lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))  # 进行车牌矫正
                Lic_img.append(lic)
                cv2.drawContours(img_src_copy, [np.array([l0, l1, l3, l2])], -1, (0, 255, 0), 2)  # 在img_src_copy上绘制出定位的车牌轮廓，(0, 255, 0)表示绘制线条为绿色
    return img_src_copy, Lic_img




def unet_predict(unet, img_src_path):
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_ = Image.open(img_src_path)
    # img_ = img_.resize((512, 512))
    # img_src = img_
    # # img_src=cv2.imread(img_src_path)
    # if img_src.size != (512, 512):
    #     img_src = img_src.resize((512, 512))  # dsize=(宽度,高度),[:,:,:3]是防止图片为4通道图片，后续无法reshape
    # # img_src = img_src.reshape(1, 512, 512, 3)  # 预测图片shape为(1,512,512,3)
    img_src = trans(img_)
    img_src = img_src.unsqueeze(0)
    img_mask = unet(img_src)
    # print(out.squeeze(0).shape) #(b, c, h, w) -> (c, h, w) -> (h, w, c)
    img_mask = img_mask.squeeze(0).detach().numpy().transpose([1, 2, 0])
    # img_mask = unet.predict(img_src)  # 归一化除以255后进行预测
    # img_src = img_src.reshape(512, 512, 3)  # 将原图reshape为3维
    # img_mask = img_mask.reshape(512, 512, 3)  # 将预测后图片reshape为3维
    img_mask = img_mask / np.max(img_mask) * 255  # 归一化后乘以255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
    img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型
    # img_ = np.asarray(img_)
    img_ = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2BGR)
    return img_, img_mask


def parseOutput(output):
    INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                      "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                      "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}

    INDEX_LETTER = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41,
                    "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
                    "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63,
                    "Z": 64}

    PLATE_CHARS_PROVINCE = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                            "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

    PLATE_CHARS_LETTER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
                          "K", "L", "M", "N", "P",
                          "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    index = 0
    maxValue = 0
    label = ""
    output = output[0]
    for i in range(31):
        if output[i] > maxValue:
            maxValue = output[i]
            index = i
    label = label + PLATE_CHARS_PROVINCE[index]
    for j in range(6):
        index = 0
        maxValue = 0
        for i in range(34):
            if output[i+j*34+34] > maxValue:
                maxValue = output[i+j*34+34]
                index = i
        label = label + PLATE_CHARS_LETTER[index]
    return label
def CNN_pridect(net, img_path):
    trans = transforms.Compose([
                                transforms.Resize((80, 240)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_path = cv2.cvtColor(img_path,cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img_path)
    img = trans(img).unsqueeze(0)
    output = net(img)
    output_label = parseOutput(output.detach().numpy())
    return output_label

def main():
    model_unt = Unet1()
    model_unt.load_state_dict(torch.load('model.pth',map_location='cpu'))
    model_unt.eval()

    model_cnn = Net()
    model_cnn.load_state_dict(torch.load('CNN.pth', map_location='cpu'))
    model_cnn.eval()

    img_src_path = 'pred/云D86999.jpg'
    img = Image.open(img_src_path)
    img_src, img_mask = unet_predict(model_unt, img_src_path)
    cv2.imshow('img_mask',img_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
    print(len(Lic_img))
    cv2.imshow('img_src_copy',img_src_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Lic_img = np.asarray(Lic_img[0])
    cv2.imshow('Lic_img',Lic_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_label = CNN_pridect(model_cnn, Lic_img)
    print(output_label)






if __name__ == '__main__':
    main()