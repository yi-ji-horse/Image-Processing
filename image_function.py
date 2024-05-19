import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
def threshold_manual(GreyImage,threshold_value=127):
    ans=GreyImage.copy()
    ans[ans>threshold_value]=255
    ans[ans<threshold_value]=0
    return ans

def threshold_Otsu(GreyImage):
    _, otsu_threshold_image = cv2.threshold(GreyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold_image = Image.fromarray(otsu_threshold_image)
    return otsu_threshold_image

def histogram(GreyImage):
    image_array = np.array(GreyImage)
    plt.figure()
    plt.hist(image_array.ravel(), bins=256, range=[0, 256])
    plt.title('灰度直方图')
    plt.xlabel('灰度值')
    plt.ylabel('像素数')
    
    # 将直方图保存为图片
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    histogram_image = Image.open(buf)
    plt.close()
    return histogram_image

def convolutionAndImageFilters(GreyImage,operator='roberts1'):
    kernels = {
        "roberts1": np.array([
            [-1,  0],
            [0,  1]
        ], dtype=np.float32),
        "roberts2": np.array([
            [0,  -1],
            [1,  0]
        ], dtype=np.float32),
        "prewitt1": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32),
        "prewitt2": np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32),
        "sobel1": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        "sobel2": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32),
        "gaussianfilter": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]/16, dtype=np.float32)
    }
    kernel = kernels.get(operator, "roberts1")
    convolved_image = cv2.filter2D(GreyImage, -1, kernel)
    return Image.fromarray(convolved_image)

def dilation_bin(GreyImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8)):
    binImage=GreyImage.copy()
    threshold_value=127
    binImage[binImage>threshold_value]=1
    binImage[binImage<=threshold_value]=0

    input_rows, input_cols = binImage.shape
    kernel_rows, kernel_cols = SE.shape

    convolution_result = np.zeros_like(input_matrix)

    for i in range(input_rows - kernel_rows + 1):
        for j in range(input_cols - kernel_cols + 1):
            input_submatrix = binImage[i:i+kernel_rows, j:j+kernel_cols]
            convolution_result[i, j] = np.sum(input_submatrix * kernel)



