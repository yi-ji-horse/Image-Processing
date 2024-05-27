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


def dilation_bin(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    BinImage=threshold_manual(GreyImage)
    return cv2.morphologyEx(BinImage,cv2.MORPH_DILATE,kernel, iterations = 1)

def erosion_bin(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    BinImage=threshold_manual(GreyImage)
    return cv2.morphologyEx(BinImage,cv2.MORPH_ERODE,kernel, iterations = 1)

def open_bin(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    BinImage=threshold_manual(GreyImage)
    return cv2.morphologyEx(BinImage,cv2.MORPH_OPEN,kernel, iterations = 1)

def close_bin(GreyImage,kernelsize=3):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    BinImage=threshold_manual(GreyImage)
    return cv2.morphologyEx(BinImage,cv2.MORPH_CLOSE,kernel, iterations = 1)

def DistanceTransform(GreyImage):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    BinImage=threshold_manual(GreyImage)
    ans=cv2.distanceTransform(BinImage,cv2.DIST_L2,maskSize=3)
    return cv2.convertScaleAbs(ans)

def skeleton(GreyImage):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    BinImage=threshold_manual(GreyImage)
    input_rows, input_cols = BinImage.shape
    expand=np.zeros([input_rows+2,input_cols+2])
    ans=np.zeros([input_rows+2,input_cols+2])
    expand[1:input_rows+1,1:input_cols+1]=BinImage
    kernel = np.ones((3,3), np.uint8)
    dis=0
    while np.sum(expand)>0:
        dis=dis+1
        ans=ans+(expand-cv2.morphologyEx(expand,cv2.MORPH_OPEN,kernel, iterations = 1))//255*dis
        expand=cv2.morphologyEx(expand,cv2.MORPH_ERODE,kernel, iterations = 1)
    return cv2.convertScaleAbs(ans[1:input_rows+1,1:input_cols+1])

def skeleton_recover(GreyImage):
    GreyImage = cv2.cvtColor(GreyImage, cv2.COLOR_BGR2GRAY)
    ans=np.zeros(GreyImage.shape)

    for i in range(1,1+np.max(GreyImage[:,:])):
        kernelsize=i*2+1
        kernel = np.ones((kernelsize,kernelsize), np.uint8)
        tmp=np.zeros(GreyImage.shape)
        tmp[GreyImage==i]=255
        ans=ans+cv2.morphologyEx(tmp,cv2.MORPH_DILATE,kernel, iterations = 1)
    ans[ans>0]=255
    return ans.astype(int)


'''
too slow

def dilation_bin(BinImage,kernelsize=3):
    BinImage=cv2.cvtColor(BinImage, cv2.COLOR_BGR2GRAY)
    BinImage=threshold_manual(BinImage,threshold_value=127)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    input_rows, input_cols = BinImage.shape
    ans=np.zeros([input_rows + kernelsize - 1,input_cols + kernelsize - 1])
    for i in range(kernelsize):
        for j in range(kernelsize):
            if kernel[i,j]:
                ans[i:i+input_rows,j:j+input_cols]=ans[i:i+input_rows,j:j+input_cols]+BinImage
    ans[ans>1]=255
    half=kernelsize//2
    return ans[half:half+input_rows,half:half+input_cols]


def erosion_bin(BinImage,kernelsize=3):
    BinImage=cv2.cvtColor(BinImage, cv2.COLOR_BGR2GRAY)
    BinImage=threshold_manual(BinImage,threshold_value=127)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    input_rows, input_cols = BinImage.shape
    expanded=np.zeros((input_rows+kernelsize-1,input_cols+kernelsize-1),np.uint8)
    ans=np.zeros((input_rows,input_cols),np.uint8)
    half=kernelsize//2
    expanded[half:half+input_rows,half:half+input_cols]=BinImage
    for i in range(input_rows):
        for j in range(input_cols):
            fitpatch=expanded[i:i+kernelsize,j:j+kernelsize]
            fitpatch[fitpatch>1]=1
            tmp=fitpatch+kernel
            tmp[tmp>1]=1
            if np.array_equal(fitpatch,tmp):
                ans[i,j]=255
    return  ans

def close_bin(BinImage,kernelsize=3):
    step1=erosion_bin(BinImage,kernelsize)

    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    input_rows, input_cols = step1.shape
    ans=np.zeros([input_rows + kernelsize - 1,input_cols + kernelsize - 1])
    for i in range(kernelsize):
        for j in range(kernelsize):
            if kernel[i,j]:
                ans[i:i+input_rows,j:j+input_cols]=ans[i:i+input_rows,j:j+input_cols]+BinImage
    ans[ans>1]=255
    half=kernelsize//2
    return ans[half:half+input_rows,half:half+input_cols]


def open_bin(BinImage,kernelsize=3):
    step1=dilation_bin(BinImage,kernelsize)
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    input_rows, input_cols = step1.shape
    expanded=np.zeros((input_rows+kernelsize-1,input_cols+kernelsize-1),np.uint8)
    ans=np.zeros((input_rows,input_cols),np.uint8)
    half=kernelsize//2
    expanded[half:half+input_rows,half:half+input_cols]=step1
    for i in range(input_rows):
        for j in range(input_cols):
            fitpatch=expanded[i:i+kernelsize,j:j+kernelsize]
            tmp=fitpatch+kernel
            tmp[tmp>1]=1
            if fitpatch==tmp:
                ans[i,j]=255
    return  ans
'''


def Distance_Transform(binImage,SE=np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.int8),startpoint=np.array([0,0],dtype=np.int8)):
    pass


