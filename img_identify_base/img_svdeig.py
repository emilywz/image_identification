# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
读取图片亮度矩阵,提取特征，奇异值分解
"""
import scipy.misc as sm
import numpy as np
import matplotlib.pyplot as mp
import cv2 as cv



# 1、openCV读取图片，亮度矩阵
img = cv.imread('../data/flower.jpg', 0)
# img = sm.imread('../data/flower.jpg', True)
img = np.mat(img)


# 2、提取特征值与特征向量，通过特征值,推导原图像,显示
eigvals, eigvecs = np.linalg.eig(img)
print(eigvals.shape, eigvecs.shape)

eigvals[50:] = 0  # 去除特征值
img2 = eigvecs * np.diag(eigvals) * eigvecs.I

# 3、奇异值分解SVD
U, sv, V = np.linalg.svd(img)
print(U, '--->U')
print(U * U.T)
print(sv, '--->sv')  # 奇异值
print(V, '--->V')
print(V * V.T)
sv[50:] = 0
# 逆向推导M
img3 = U * np.diag(sv) * V

mp.subplot(1, 3, 1)
mp.xticks([])
mp.yticks([])
mp.imshow(img, cmap='gray')

mp.subplot(1, 3, 2)
mp.xticks([])
mp.yticks([])
mp.imshow(img2.real, cmap='gray')

mp.subplot(1, 3, 3)
mp.xticks([])
mp.yticks([])
mp.imshow(img3.real, cmap='gray')

mp.tight_layout()  # 紧凑布局
mp.show()
