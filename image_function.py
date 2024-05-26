import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
def RGB2Grey(RGBImage):
    return cv2.cvtColor(RGBImage,cv2.COLOR_BGR2GRAY)

def threshold_manual(GreyImage,threshold_value=127):
    ans=GreyImage.copy()
    ans[ans>threshold_value]=255
    ans[ans<threshold_value]=0
    return ans

def threshold_Otsu(GreyImage):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    _, otsu_threshold_image = cv2.threshold(GreyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #otsu_threshold_image = Image.fromarray(otsu_threshold_image)
    return otsu_threshold_image

def histogram(GreyImage):
    image_array = np.array(GreyImage)
    plt.figure()
    plt.hist(image_array.ravel(), bins=256, range=[0, 256])
    plt.title('Grayscale histogram')
    plt.xlabel('gray level')
    plt.ylabel('pixel number')
    
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
        "gaussian": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)/16,
        "median": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)/16
    }
    kernel = kernels.get(operator, "roberts1")
    convolved_image = cv2.filter2D(GreyImage, -1, kernel)
    return Image.fromarray(convolved_image)

def dilation_gray(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    return cv2.dilate(GreyImage, kernel, iterations = 1)

def erosion_gray(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    return cv2.erode(GreyImage, kernel, iterations = 1)

def close_gray(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    return cv2.dilate(cv2.erode(GreyImage, kernel, iterations = 1),kernel,iterations=1)

def open_gray(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    return cv2.erode(cv2.dilate(GreyImage, kernel, iterations = 1),kernel,iterations=1)


'''

def dilation_bin(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8)):

    input_rows, input_cols = binImage.shape
    kernel_rows, kernel_cols = SE.shape

    ans=np.zeros([input_rows + kernel_rows - 1,input_cols + kernel_cols - 1])
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            ans[i:i+input_rows,j:j+input_cols]=ans[i:i+input_rows,j:j+input_cols]+binImage
    ans[ans>1]=1
    return ans

def erosion_bin(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8)):
    input_rows, input_cols = binImage.shape
    kernel_rows, kernel_cols = SE.shape
    if (input_rows<kernel_rows or input_cols<kernel_cols):
        return binImage
    ans=np.zeros([input_rows - kernel_rows + 1,input_cols - kernel_cols + 1])
    for i in range(input_rows - kernel_rows + 1):
        for j in range(input_cols - kernel_cols + 1):
            fitpatch=binImage[i:i+kernel_rows,j:j+kernel_cols]
            tmp=fitpatch+SE
            tmp[tmp>1]=1
            if fitpatch==tmp:
                ans[i,j]=1
    return

def close_bin(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8)):
    return(dilation_bin(erosion_bin(binImage,SE),SE))

def open_bin(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8)):
    return(erosion_bin(dilation_bin(binImage,SE),SE))

def Distance_Transform(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8),startpoint=np.array([0,0],dtype=np.int8)):
    pass

'''
