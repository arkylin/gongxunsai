import cv2
import numpy as np

frame_wh = (400,300)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 定义滑块回调函数
def nothing(x):
    pass

# 创建窗口和滑块
cv2.namedWindow('image')
cv2.createTrackbar('param1', 'image', 50, 100, nothing)
cv2.createTrackbar('param2', 'image', 30, 100, nothing)
cv2.createTrackbar('minRadius', 'image', 140, 200, nothing)
cv2.createTrackbar('maxRadius', 'image', 170, 200, nothing)

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame, frame_wh)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  
            # draw the center of the circle  
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)  
        cv2.imshow('detected circles', image)  

    # 显示滑块调节的窗口
    cv2.imshow('image', image)

    # 按下Esc键退出
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
