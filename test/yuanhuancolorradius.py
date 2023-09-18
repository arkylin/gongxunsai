import cv2
import numpy as np

def nothing(x):
    pass

# 红色范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([20, 255, 255])

# 绿色范围
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# 蓝色范围
lower_blue = np.array([60, 35, 100])
upper_blue = np.array([130, 255, 255])

cv2.namedWindow('image')
cv2.createTrackbar('param1', 'image', 50, 100, nothing)
cv2.createTrackbar('param2', 'image', 30, 100, nothing)
cv2.createTrackbar('minRadius', 'image', 140, 200, nothing)
cv2.createTrackbar('maxRadius', 'image', 170, 200, nothing)

# 初始化摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建黄色和灰色的掩码
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # 将掩码进行位运算，生成最终掩码
    mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(green_mask, blue_mask))

    # 对原始画面进行掩码操作，提取红色、绿色和蓝色部分
    result = cv2.bitwise_and(frame, frame, mask=mask)

    
    # 将HSV图像转换为BGR图像
    bgr_image = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    # 将BGR图像转换为灰度图像
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


    # 读取滑块的值
    param1 = cv2.getTrackbarPos('param1', 'image')
    param2 = cv2.getTrackbarPos('param2', 'image')
    minRadius = cv2.getTrackbarPos('minRadius', 'image')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'image')

    # 应用滑块的值到圆检测函数中
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,  
                            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:  
        # 将找到的圆形转为整数  
        circles = np.uint16(np.around(circles))  
        for i in circles[0, :]:  
            # draw the outer circle  
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  
            # draw the center of the circle  
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  
    cv2.imshow('detected circles', frame)  

    # # 显示结果
    # cv2.imshow('image', image)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
