import os
import numpy as np
from tqdm import tqdm
INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                  "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                  "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}

INDEX_LETTER = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,"H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
                "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

# print(convert_to_one_hot(np.asarray(INDEX_PROVINCE['京']), 32)[0])


img_path = 'CNN_img'
img_path_list = os.listdir(img_path)
with open('name_label.txt','w') as f:
    for i in tqdm(img_path_list):
        # print(i)
        labem_num = np.zeros([238])
        img_name = i.split('.')[0]
        for n, _data in enumerate(img_name):
            # print(_data)
            if n == 0:
                # convert_to_one_hot(INDEX_PROVINCE[_data], 1)
                labem_num[:31] = convert_to_one_hot(np.asarray(INDEX_PROVINCE[_data]), 31)[0]
            elif n != 0:
                # print(np.asarray(INDEX_LETTER[_data]-31))
                if _data =='I':
                    _data = '1'
                elif _data == 'O':
                    _data = '0'
                labem_num[34*n:34*(n+1)] =convert_to_one_hot(np.asarray(INDEX_LETTER[_data]-31), 34)[0]

        f.write(str(i)+' '+','.join(str(i) for i in labem_num)+'\n')
        # break

# with open('name_label.txt', 'r') as f:
#     data = f.readline()
#     print(data.split(' '))


