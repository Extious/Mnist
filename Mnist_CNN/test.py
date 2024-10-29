import cv2
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt

from LeNet_5 import LeNet_5

model = LeNet_5()
model.load_state_dict(torch.load("model.pt"))

img = cv2.imread("3.1.png")

# 图像预处理，灰度图，二值化
img = cv2.resize(img,(32,32))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)

# 图像旋转
# row,col = img.shape
# print(col,row)
# M = cv2.getRotationMatrix2D((col/2,row/2),30,1)
# img = cv2.warpAffine(img,M,(col,row))

# 图像翻转
# img = cv2.flip(img,1)

# 图像平移
# 声明变换矩阵 向右平移10个像素， 向下平移30个像素
row,col = img.shape
M = np.float32([[1, 0, 5], [0, 1, 5]])
# 进行2D 仿射变换
img = cv2.warpAffine(img, M, (col, row))


# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

# cv2.imshow(" 1",img)
# cv2.waitKey(0)

plt.figure()
plt.imshow(img,cmap="gray",interpolation="none")
plt.show()

transform = torchvision.transforms.ToTensor()
img = transform(img)
img = img.unsqueeze(0)
print(type(img),img.shape)

output = model(img)
print("output:",output)
prob,label = torch.max(output,1)
print("prob:",prob)
print("label:",label)

