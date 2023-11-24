import numpy as np
import cv2
import matplotlib.pyplot as plt



def RGB2GRAY(image):
    return np.dot(image[...,:3], [0.21, 0.72, 0.07]).astype(np.uint8)

def mean_filter(image,kernel_size):
    height, width = image.shape[:2]
    matrix=np.zeros((height-kernel_size+1,width-kernel_size+1))
    for i in range(height-kernel_size+1):
        for j in range(width-kernel_size+1):
            total=0
            for a in range(kernel_size):
                for b in range(kernel_size):
                    total+=image[i+a,j+b]
            matrix[i,j]=total/(kernel_size**2)
    return matrix.astype(np.uint8)

def quicksort(data, left, right): 
    if left >= right :            # 如果左邊大於右邊，就跳出function
        return

    i = left                      # 左邊的代理人
    j = right                     # 右邊的代理人
    key = data[left]                 # 基準點

    while i != j:                  
        while data[j] > key and i < j:   # 從右邊開始找，找比基準點小的值
            j -= 1
        while data[i] <= key and i < j:  # 從左邊開始找，找比基準點大的值
            i += 1
        if i < j:                        # 當左右代理人沒有相遇時，互換值
            data[i], data[j] = data[j], data[i] 

    # 將基準點歸換至代理人相遇點
    data[left] = data[i] 
    data[i] = key

    quicksort(data, left, i-1)   # 繼續處理較小部分的子循環
    quicksort(data, i+1, right)  # 繼續處理較大部分的子循環

def median(sorted_arr):
    quicksort(sorted_arr,0, len(sorted_arr)-1)
    n=len(sorted_arr)
    if n%2==1:
        return sorted_arr[n//2]
    else:
        middle1=sorted_arr[n//2]
        middle2=sorted_arr[n//2-1]
        return (middle1+middle2)/2

def median_filter(image,kernel_size):
    height, width = image.shape[:2]
    matrix=np.zeros((height-kernel_size+1,width-kernel_size+1))
    for i in range(height-kernel_size+1):
        for j in range(width-kernel_size+1):
            arr=[]
            for a in range(kernel_size):
                for b in range(kernel_size):
                    arr.append(image[i+a,j+b])
            matrix[i,j]=median(arr)            
    return matrix.astype(np.uint8)

def his(image,title,path):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    plt.title(title)
    plt.bar(bins[:-1], hist, width=1.0)
    plt.savefig(path)
    plt.show()
    


noise_img=cv2.imread("test_img/noise_image.png")
cv2.imshow('noise_image', noise_img) 
gray_img = RGB2GRAY(noise_img)
his(gray_img,"gray_img","result_img/noise_image_his.png")
cv2.waitKey(0)
cv2.destroyAllWindows()

mean_img=mean_filter(gray_img,3)
cv2.imshow('mean_image', mean_img) 
cv2.imwrite("result_img/output1.png", mean_img)
his(mean_img,"mean_img","result_img/output1_his.png")
cv2.waitKey(0)
cv2.destroyAllWindows()

median_img=median_filter(gray_img,3)
cv2.imshow('median_image', median_img) 
cv2.imwrite("result_img/output2.png", median_img)
his(median_img,"median_img","result_img/output2_his.png")
cv2.waitKey(0)
cv2.destroyAllWindows()










































