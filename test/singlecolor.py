import cv2
import numpy as np

def nothing(x):
    pass

# 创建窗口和滑动条
cv2.namedWindow('image')
cv2.createTrackbar('Lower Hue', 'image', 0, 179, nothing)
cv2.createTrackbar('Lower Saturation', 'image', 100, 255, nothing)
cv2.createTrackbar('Lower Value', 'image', 100, 255, nothing)

cv2.createTrackbar('Upper Hue', 'image', 10, 179, nothing)
cv2.createTrackbar('Upper Saturation', 'image', 255, 255, nothing)
cv2.createTrackbar('Upper Value', 'image', 255, 255, nothing)

# 初始化摄像头
cap = cv2.VideoCapture(1)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条的当前值
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'image')
    lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'image')
    lower_value = cv2.getTrackbarPos('Lower Value', 'image')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'image')
    upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'image')
    upper_value = cv2.getTrackbarPos('Upper Value', 'image')

    # 定义红色、绿色和蓝色的HSV范围
    lower = np.array([lower_hue, lower_saturation, lower_value])
    upper = np.array([upper_hue, upper_saturation, upper_value])

    # 创建红色、绿色和蓝色掩码
    mask = cv2.inRange(hsv_frame, lower, upper)

    # 对原始画面进行掩码操作，提取红色、绿色和蓝色部分
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 显示结果
    cv2.imshow('image', result)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
