import cv2
import numpy as np

# frame_wh = (640,480)
frame_wh = (400,300)

# 初始化摄像头
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# 定义黄色的颜色范围
lower_yellow = np.array([20, 70, 0])
upper_yellow = np.array([40, 255, 255])

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    frame = cv2.resize(frame, frame_wh)

    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # 对掩码进行形态学操作，以去除噪声
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    circle_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (int(frame_wh[0]/2-120),int(frame_wh[1]/6)), 40, (255, 255, 255), -1)
    mean_color = cv2.mean(yellow_mask, mask=circle_mask)[:3][0]
    print(int(mean_color))
    # gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)

    cv2.circle(yellow_mask, (int(frame_wh[0]/2-120),int(frame_wh[1]/6)), 40, (255, 255, 255), -1)
    cv2.imshow('t', yellow_mask)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
