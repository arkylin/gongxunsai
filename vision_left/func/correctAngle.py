import sys
import math
sys.path.append(r'c:\Users\i\Code\lunaaaaaaa\main')  # Adjust the path accordingly
from vision_block.rec_box import box
import cv2
import numpy as np

X1 = np.asarray([0])  # 最优估计状态
P1 = np.asarray([1])  # 最优状态协方差矩阵
Q1 = np.asarray([0.0025])  # 状态转移方差  (预测模型的方差)
F1 = np.asarray([1])  # 状态转移矩阵
H1 = np.asarray([1])  # 观测矩阵
R1 = 0.1  # 观测噪声方差
B1 = np.asarray([1])  # 控制矩阵
EYE1 = np.asarray([1])

def kalman1(val):
    global X1, P1, Q1, F1, H1, R1, B1, EYE1
    X_ = F1 * X1
    P_ = F1 * P1 * F1.T + Q1
    K = (P_ * H1.T) / (H1 * P_ * H1.T + R1)
    X1 = X_ + K * (val - H1 * X_)
    P1 = (EYE1 - K * H1) * P_
    return X1[0]
    
# 初始化摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
frame_wh = (400,300)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    frame = cv2.resize(frame, frame_wh)

    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    other_circles_data = box(hsv_frame)
    frame_data = []
    for i in range(3):
        for j in range(3):
            if j == 1:
                item_block_circle_data = int(other_circles_data[i][j]/frame_wh[0]*255)
            elif j == 2:
                item_block_circle_data = int(other_circles_data[i][j]/frame_wh[1]*255)
            else:
                item_block_circle_data = other_circles_data[i][j]
            frame_data.append(item_block_circle_data)
    # print(frame_data) 
    # [1, 67, 115, 2, 132, 112, 3, 196, 110]
    #  0  1   2    3  4    5    6  7    8
    block_box_dx = frame_data[4]-frame_data[1]
    block_box_dy = frame_data[5]-frame_data[2]
    if block_box_dx != 0:
        block_box_slope = block_box_dy / block_box_dx
        # 计算夹角
        block_box_angle_rad = math.atan(block_box_slope)
        block_box_angle_deg = math.degrees(block_box_angle_rad)
    else:
        block_box_angle_deg = 90
    
    block_box_filtered_value = kalman1(block_box_angle_deg)
    block_box_filtered_value = block_box_angle_deg
    print(block_box_filtered_value)
    # 将整数转换为4位16进制
    block_box_multiplied_value = int(block_box_filtered_value * 100)
    if block_box_multiplied_value >= 0:
        block_box_hex_representation = format(block_box_multiplied_value, '04X')
    else:
        block_box_hex_representation = format((1 << 16) + block_box_multiplied_value, '04X')

    frame_data.append(int(block_box_hex_representation[:2],16))
    frame_data.append(int(block_box_hex_representation[2:],16))
    # print((frame_data[2]+frame_data[5]+frame_data[8]/3))

    # print(frame_data)

    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
