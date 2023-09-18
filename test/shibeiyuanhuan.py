import cv2
import numpy as np
import time
import math

# 红色范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([20, 255, 255])

# 绿色范围
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# 蓝色范围
lower_blue = np.array([60, 35, 100])
upper_blue = np.array([130, 255, 255])

frame_wh = (400,300)

def check_color_range(color, lower_bound, upper_bound):
    return np.all(np.logical_and(color >= lower_bound, color <= upper_bound))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, frame_wh)

    # 将图像转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建黄色和灰色的掩码
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask = cv2.bitwise_or(red_mask,cv2.bitwise_or(green_mask,blue_mask))

    # # 对掩码进行形态学操作，以去除噪声
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Test", mask)

    # img = mask
    # # # 查找轮廓
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # block_data = []
    # #初始化变量
    # if len(contours) > 0:
    #     print(contours)
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if isinstance(contour, np.ndarray) and area > frame_wh[0]*frame_wh[1]*0.02:
    #             # 获取面积最大轮廓的凸包
    #             hull = cv2.convexHull(contour)
    #             # 创建一个掩膜图像，用于提取凸包区域
    #             hull_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #             cv2.drawContours(hull_mask, [hull], 0, 255, -1)
    #             cv2.imshow("1",hull_mask)

    #             # 计算凸包区域的平均颜色
    #             mean_color = cv2.mean(hsv_frame, mask=hull_mask)[:3]
    #             print(mean_color)

    #             if check_color_range(mean_color, lower_red, upper_red):
    #                 block_data.append("red")
    #             elif check_color_range(mean_color, lower_green, upper_green):
    #                 block_data.append("green")
    #             elif check_color_range(mean_color, lower_blue, upper_blue):
    #                 block_data.append("blue")
    #             else:
    #                 block_data.append("null")

    #             # Debug
    #             # 创建一个与原始图像相同大小的空白图像
    #             # mask = np.zeros_like(frame)

    #             # 在空白图像上绘制最大面积轮廓
    #             # print(max_contour)
    #             # cv2.drawContours(mask, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)

    #             # 将原始图像和掩码图像进行位运算，将最大面积区域涂色
    #             # image = cv2.bitwise_and(frame, mask)

    #             # cv2.imshow("Test", image)

    #             # 计算外接矩形
    #             x, y, w, h = cv2.boundingRect(hull)
    #             block_x = int(x+w/2)
    #             block_y = int(y+h/2)
    #             block_data.append(block_x)
    #             block_data.append(block_y)
    #             cv2.circle(frame, (block_x, block_y), 5, (0, 255, 0), -1)
    #             # cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
    #             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #             (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(hull)
    #             circle_center = (int(circle_x), int(circle_y))
    #             circle_radius = int(circle_radius)
    #             cv2.circle(frame, circle_center, circle_radius, (255, 255, 255), 2)  # 绘制圆形框
    # print(block_data)
    # cv2.imshow("Test", frame)
    # 按下Esc键退出
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()