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

# 初始化摄像头
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制最小外接圆
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius >= 50 and radius <=60:
            cv2.circle(result, center, radius, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('image', result)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
