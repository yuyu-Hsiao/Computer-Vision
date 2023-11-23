import numpy as np
import cv2

def RGB2GRAY(image):
    return np.dot(image[...,:3], [0.21, 0.72, 0.07]).astype(np.uint8)

def ReLU(x):
    if(x < 0): return 0
    else: return x

def EDGE_DECT(image,kernel):
    height, width = image.shape[:2]
    k_hig, k_wid = np.shape(kernel)
    matrix=np.zeros((height-k_hig+1,width-k_wid+1))
    for i in range(height-k_hig+1):
        for j in range(width-k_wid+1):
            matrix[i,j]=ReLU((kernel*image[i:i+k_hig,j:j+k_wid]).sum())
    return matrix

    
def pooling(image):
    height, width = image.shape[:2]
    height_2=int(round(height/2))
    width_2=int(round(width/2))
    matrix=np.zeros((height_2,width_2))
    for i in range(height_2):
        for j in range(width_2):
            if(2*i+2<height):
                ri=2*i+2
            else:
                ri=2*i+1
            if(2*j+2<width):
                rj=2*j+2
            else:
                rj=2*j+1
            matrix[i, j] = np.max(image[2*i: ri, 2*j: rj])        
    return matrix

def binarization(image, threshold):
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            if(image[i ,j] < threshold):
                image[i ,j] = 0
            else:
                image[i, j] = 255
    return image




img=cv2.imread("test_img/liberty.png")
gray_img = RGB2GRAY(img)
cv2.imshow('RGB Image To Grayscale', gray_img) 
cv2.imwrite("result_img/liberty_Q1.png", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kern=np.array([[0,1,0],[1,-4,1],[0,1,0]])
edge_img=EDGE_DECT(gray_img,kern)
cv2.imshow('EDGE_DECT', edge_img)
cv2.imwrite("result_img/liberty_Q2.png", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pool_img=pooling(edge_img)
cv2.imshow('pooling', pool_img)
cv2.imwrite("result_img/liberty_Q3.png", pool_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

bin_img = binarization(pool_img,128)
cv2.imshow('Binarizatio', bin_img)
cv2.imwrite("result_img/liberty_Q4.png", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





img=cv2.imread("test_img/temple.jpg")
gray_img = RGB2GRAY(img)
cv2.imshow('RGB Image To Grayscale', gray_img) 
cv2.imwrite("result_img/temple_Q1.jpg", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kern=np.array([[0,1,0],[1,-4,1],[0,1,0]])
edge_img=EDGE_DECT(gray_img,kern)
cv2.imshow('EDGE_DECT', edge_img)
cv2.imwrite("result_img/temple_Q2.jpg", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pool_img=pooling(edge_img)
cv2.imshow('pooling', pool_img)
cv2.imwrite("result_img/temple_Q3.jpg", pool_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

bin_img = binarization(pool_img,128)
cv2.imshow('Binarizatio', bin_img)
cv2.imwrite("result_img/temple_Q4.jpg", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
