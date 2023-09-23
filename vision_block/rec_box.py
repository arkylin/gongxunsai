import cv2
import numpy as np
import platform
import time

system = platform.system()

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

def box(hsv_frame):
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask = cv2.bitwise_or(red_mask,cv2.bitwise_or(green_mask,blue_mask))
    origin_result = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    result = cv2.cvtColor(origin_result,cv2.COLOR_HSV2RGB)
    result = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
    result = cv2.GaussianBlur(result, (5,5), 0)
    result = cv2.medianBlur(result, 5)
    #霍夫圆检测
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=25, maxRadius=60)
    circles_data = []

    # 如果检测到圆
    if circles is not None:
        # 将检测到的圆转换为整数坐标
        circles = np.round(circles[0, :]).astype("int")

        # 遍历检测到的圆并绘制
        for (x, y, r) in circles:
            if y >= 50 and y<=156:
                circle_color = 0
                circle_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                pure_circle_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
                cv2.circle(circle_mask, (x, y), int(r/3*2), (0, 0, 0), -1)

                pure_circle = cv2.bitwise_and(origin_result,origin_result,mask=circle_mask)
                circle_gray = cv2.cvtColor(pure_circle, cv2.COLOR_HSV2BGR)
                circle_gray = cv2.cvtColor(circle_gray, cv2.COLOR_BGR2GRAY)
                # 对灰度图像进行阈值化
                _, heibai_img = cv2.threshold(circle_gray, 127, 255, cv2.THRESH_BINARY)
                
                if system == 'Windows':
                    # cv2.imshow("Test",cv2.bitwise_and(hsv_frame,hsv_frame,mask=circle_mask))
                    # cv2.imshow("Test2",cv2.cvtColor(pure_circle_mask,cv2.COLOR_HSV2BGR))
                    cv2.imshow("Test2",heibai_img)

                mean_color = cv2.mean(hsv_frame, mask=heibai_img)[:3]
                # print(mean_color)
                if check_color_range(mean_color, lower_red, upper_red):
                    circle_color = 1
                elif check_color_range(mean_color, lower_green, upper_green):
                    circle_color = 2
                elif check_color_range(mean_color, lower_blue, upper_blue):
                    circle_color = 3
                circles_data.append([circle_color,x,y])
    # print(circles_data)
    if len(circles_data) !=3:
        while(len(circles_data) < 3):
            circles_data.append([0,400,300])
        while(len(circles_data) > 3):
            circles_data.pop()
    circles_data = sorted(circles_data, key=lambda x: x[1])
    return circles_data

def check_color_range(color, lower_bound, upper_bound):
    return np.all(np.logical_and(color >= lower_bound, color <= upper_bound))

if __name__ == '__main__':
    # 初始化摄像头
    if system == 'Windows':
        # 初始化摄像头
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif system == 'Linux':
        # 初始化摄像头
        cap = cv2.VideoCapture("/dev/block_video0")
    else:
        print(system, flush=True)
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        frame = cv2.resize(frame, frame_wh)
        # cv2.imshow("T",frame)
        # 转换为HSV颜色空间
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("T",hsv_frame)
        print(box(hsv_frame))
        # 按下ESC键退出循环
        if cv2.waitKey(1) == 27:
            break