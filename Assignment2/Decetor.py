# <!--------------------------------------------------------------------------->
# <!--                    KU - University of Copenhagen                      -->
# <!--                          Faculty of Science                           -->
# <!--                Vision and Image Processing (VIP) Course               -->
# <!-- File       : Decetor.py                                                  -->
# <!-- Description: Feature Extraction and matching                          -->
# <!-- Author     : Weisi Li (email missing)                                 -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019091501 $"

####################################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np


####################################################################################


# 1 Detecting interest points (Features)

def load(filename, flags=cv2.IMREAD_GRAYSCALE):
    image = cv2.imread(filename, flags)
    return image


def save_descriptor():
    file_path = "./outputs/"
    pass


def compute_M(image, ksize):
    """
    This function computes the orientation and magnitude of input image.
    """
    # Check if the input image is grayscale.
    k = 0.15  # 响应函数k
    threshold = 0.01  # 设定阈值

    if len(image.shape) == 2:
        grayscale = image.copy()
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = np.shape(grayscale)

    # Calculate the gradients.
    g_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=ksize)
    g_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=ksize)

    A, B, C = g_x ** 2, g_y ** 2, g_x * g_y

    A, B, C = [cv2.GaussianBlur(i, ksize=(ksize, ksize), sigmaX=2) for i in [A, B, C]]
    # B = cv2.GaussianBlur(B, ksize=(ksize, ksize), sigmaX=2)
    # C = cv2.GaussianBlur(C, ksize=(ksize, ksize), sigmaX=2)

    T = [np.array([[A[i, j], C[i, j]],  # compute T
                   [C[i, j], B[i, j]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D, T = list(map(np.linalg.det, T)), list(
        map(np.trace, T))  # map: https://www.w3schools.com/python/showpython.asp?filename=demo_ref_map2
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])
    # R = (A * C - B ** 2) - k * ((A + C) ** 2)

    # 5、将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
            if R[i, j] > R_max * threshold and R[i, j] == np.max(
                    R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                corner[i, j] = -1
    return corner


if __name__ == '__main__':
    # Load from Input image filename.
    filename = "./inputs/Img001_diffuse.tif"
    image = cv2.imread(filename)

    corners = compute_M(image, 3)
    image[corners == -1] = [0, 255, 0]
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

