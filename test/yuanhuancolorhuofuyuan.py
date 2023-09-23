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

# frame_wh = (640,480)
frame_wh = (400,300)

# 初始化摄像头
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# 创建窗口和滑块
# cv2.namedWindow('image')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600, 800)

cv2.createTrackbar('dp', 'image', 1, 10, nothing)
cv2.createTrackbar('minDist', 'image', 50, 200, nothing)
cv2.createTrackbar('param1', 'image', 50, 100, nothing)
cv2.createTrackbar('param2', 'image', 30, 100, nothing)
cv2.createTrackbar('minRadius', 'image', 25, 200, nothing)
cv2.createTrackbar('maxRadius', 'image', 60, 200, nothing)


while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    frame = cv2.resize(frame, frame_wh)

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
    # result = frame
    result = cv2.cvtColor(result,cv2.COLOR_HSV2RGB)
    result = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
    result = cv2.GaussianBlur(result, (5,5), 0)
    result = cv2.medianBlur(result, 5)
    cv2.imshow('image1', result)

    # 获取滑块参数
    dp = cv2.getTrackbarPos('dp', 'image')
    minDist = cv2.getTrackbarPos('minDist', 'image')
    param1 = cv2.getTrackbarPos('param1', 'image')
    param2 = cv2.getTrackbarPos('param2', 'image')
    minRadius = cv2.getTrackbarPos('minRadius', 'image')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'image')

    # 使用霍夫圆变换检测圆
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    circles_data = []

    # 如果检测到圆
    if circles is not None:
        # 将检测到的圆转换为整数坐标
        circles = np.round(circles[0, :]).astype("int")

        # 遍历检测到的圆并绘制
        for (x, y, r) in circles:
            if y >= 50 and y<=156:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                circles_data.append([x,y,r])
        
        circles_data = sorted(circles_data, key=lambda x: x[0])

    # 显示结果
    # cv2.line(frame, (0, 80), (frame.shape[1], 80), (0, 255, 0), 2)
    # cv2.line(frame, (0, 250), (frame.shape[1], 250), (0, 255, 0), 2)
    cv2.line(frame, (0, 50), (frame.shape[1], 50), (0, 255, 0), 2)
    cv2.line(frame, (0, 156), (frame.shape[1], 156), (0, 255, 0), 2)
    cv2.imshow('image', frame)
    print(circles_data)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
