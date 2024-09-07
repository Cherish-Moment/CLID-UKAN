import cv2
import torch
image = cv2.imread(r'D:\DownLoad\Awesome-U-Net-main\data\BUSI_with_GT\benign\benign (1).png')
print(image.shape)
channels = image.shape[2]
print(f"The image has {channels} channels.")