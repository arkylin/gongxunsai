import os
import cv2
import platform
import time

system = platform.system()

# 红色范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# 绿色范围
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# 蓝色范围
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

frame_wh = (400,300)

def vision_block(conn):    
    # 初始化摄像头
    if system == 'Windows':
        # 初始化摄像头
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    elif system == 'Linux':
        # 初始化摄像头
        cap = cv2.VideoCapture("/dev/block_video0")
    else:
        print(system)
        
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    if cap.isOpened():
        while True:
            # start_time = time.perf_counter()
            # 读取摄像头图像
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
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("Test", red_mask)

            img = mask
            # # 查找轮廓
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #初始化变量
            if len(contours) > 0:
                # 找到最大面积的轮廓
                max_area = 0
                max_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        max_contour = contour
                    # print(max_area,max_contour)
                    if isinstance(max_contour, np.ndarray) and max_area > frame_wh[0]*frame_wh[1]*0.02:
                        # 获取面积最大轮廓的凸包
                        hull = cv2.convexHull(max_contour)
                        conn.send("Hello")

                        # Debug
                        # 创建一个与原始图像相同大小的空白图像
                        # mask = np.zeros_like(frame)

                        # 在空白图像上绘制最大面积轮廓
                        # print(max_contour)
                        # cv2.drawContours(mask, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)

                        # 将原始图像和掩码图像进行位运算，将最大面积区域涂色
                        # image = cv2.bitwise_and(frame, mask)

                        # cv2.imshow("Test", image)

                        # 计算外接矩形
                        # x, y, w, h = cv2.boundingRect(hull)
                        # block_x = int(x+w/2)
                        # block_y = int(y+h/2)
                        # cv2.circle(frame, (block_x, block_y), 5, (0, 255, 0), -1)
                        # # cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
                        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        print("Block摄像头无法打开")