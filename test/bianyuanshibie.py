import cv2
import os
import numpy as np
import time
import math
import serial
import platform
from pyzbar import pyzbar

system = platform.system()

# 定义黄色的颜色范围
lower_yellow = np.array([20, 30, 30])
upper_yellow = np.array([40, 255, 255])

frame_wh = (400,300)

claw_xy = (200, 150)

GUAIJIAO = 0
# jiaodianshu_last = 0

qrcode_data = "qrcode.txt"

X = np.asarray([0])  # 最优估计状态
P = np.asarray([1])  # 最优状态协方差矩阵
Q = np.asarray([0.0025])  # 状态转移方差  (预测模型的方差)
F = np.asarray([1])  # 状态转移矩阵
H = np.asarray([1])  # 观测矩阵
R = 0.1  # 观测噪声方差
B = np.asarray([1])  # 控制矩阵
EYE = np.asarray([1])

def kalman(val):
    global X, P, Q, F, H, R, B, EYE
    X_ = F * X
    P_ = F * P * F.T + Q
    K = (P_ * H.T) / (H * P_ * H.T + R)
    X = X_ + K * (val - H * X_)
    P = (EYE - K * H) * P_
    return X[0]

# filtered_values = []

def guaijiaoshibie(jiaodiancanshu):
    jiaodianshu = len(jiaodiancanshu)
    # [[[639   0]]

    # [[639 339]]

    # [[  0 355]]

    # [[  0  66]]

    # [[122   0]]]
    global GUAIJIAO

    # if jiaodianshu != GUAIJIAO_last:
    if jiaodianshu == 3:
        # zhongxindianx = int((jiaodiancanshu[0][0][0] + jiaodiancanshu[0][1][0] + jiaodiancanshu[0][2][0])/3)
        # zhongxindiany = int((jiaodiancanshu[0][0][1] + jiaodiancanshu[0][1][1] + jiaodiancanshu[0][2][1])/3)
        center = np.mean(jiaodiancanshu, axis=0)  # 计算多边形的中心点
        center = center.astype(int)  # 将坐标转换为整数
        if center[0][0] < int(frame_wh[0]/2) and center[0][1] < int(frame_wh[1]/2):
            GUAIJIAO = 1
        else:
            GUAIJIAO = 0
    else:
        GUAIJIAO = 0

def send_serial_data(ser,frame_data):
    # 发送串口数据
    # 将帧数据转换为16进制字符串
    print("串口已发送：", frame_data[:], flush=True)
    hex_frame = bytes(frame_data)
    # print(hex_frame)
    # 将数据发送到串口
    ser.write(hex_frame)


def decode_qr_code(image, iswin=False):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用pyzbar解码二维码
    data = []
    barcodes = pyzbar.decode(gray)
    # print(barcodes)

    # 遍历解码结果
    for barcode in barcodes:
        # 提取二维码数据
        data.append(barcode.data.decode("utf-8"))

        if iswin:
            # 绘制边界框
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if iswin:
        # 显示图像
        cv2.imshow("Image", image)
    return data

def update_screen_by_qrcode(image,ser="",action=1):
    data = decode_qr_code(image)
    if action == 1:
        if len(data) > 0 and len(data[0]) == 7:
            with open(qrcode_data, "w") as file:
                file.write(data[0])
                file.close()
            if ser != "":
                # print(type(data[0]))
                ser.write(("t0.txt=\""+ data[0] +"\"").encode())
                ser.write(bytes.fromhex('ff ff ff'))      
            return data[0]
    elif action == 0:
        with open(qrcode_data, "r") as file:
            data = file.read()
            file.close()
    return 0

def convert_to_hex_te(lst):
    hex_str = '0x{:02x}'.format(int(''.join(map(str, lst))))
    return hex_str
    # [1,6] 0x10

def convert_to_hex(lst):
    hex_str = '0x' + ''.join([hex(num)[2:] for num in lst])
    return hex_str
    # [1,6] 0x16

def convert_to_need_numbers(lst):
    if len(lst) == 2:
        return lst[0]*10+lst[1]
    else:
        return 0

def parsing_scanned_qrcode_data(data):
    if "+" in data and len(data) == 7 :
        data = data.split("+")
        datalist = [123,132,213,231,312,321]
        return_data = []
        return_data.append(datalist.index(int(data[0]))+1)
        return_data.append(datalist.index(int(data[1]))+1)
        return return_data
    else:
        return False

def vision_left(conn1,conn2):
    #初始化变量
    old_value_x = 0
    old_value_y = 0
    serial_available = 0
    serial0_available = 0
    if os.path.exists(qrcode_data):
        os.remove(qrcode_data)
    # 创建串口对象
    if system == 'Windows':
        port='COM1'
        # 初始化摄像头
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    elif system == 'Linux':
        port='/dev/ttyUSB0'
        # 初始化摄像头
        cap = cv2.VideoCapture("/dev/left_video0")
    else:
        print(system, flush=True)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    if cap.isOpened():
        print('Video Left启动成功！', flush=True)
        try:
            ser = serial.Serial(
                port=port,
                baudrate=115200,  # 波特率，根据实际情况修改
                timeout=1  # 超时时间，根据实际情况修改
            )
            serial_available = 1
            print("检测到STM32串口", flush=True)
        except:
            serial_available = 0
            print("没有检测到STM32串口", flush=True)
        try:
            ser0 = serial.Serial(
                port="/dev/serial0",
                baudrate=115200,  # 波特率，根据实际情况修改
                timeout=1  # 超时时间，根据实际情况修改
            )
            serial0_available = 1
            print("检测到屏幕串口", flush=True)
        except:
            serial0_available = 0
            print("没有检测到屏幕串口", flush=True)
        
        # ser0.write(("t2.txt=\""+ "Get Ready!" +"\"").encode())
        # ser0.write(bytes.fromhex('ff ff ff'))

        while True:
            # start_time = time.perf_counter()
            # 读取摄像头图像
            ret, frame = cap.read()
            frame = cv2.resize(frame, frame_wh)

            if system == 'Linux':
                if os.path.exists(qrcode_data):
                    # print("识别到二维码", flush=True)
                    pass
                else:
                    # print("正在识别二维码")
                    qrcode_number = update_screen_by_qrcode(frame,ser0,1)
                    continue
            
            # if qrcode_number != '':
            #     send_serial_data(ser0,[0,convert_to_hex(parsing_scanned_qrcode_data("123+123")),13])

            # 将图像转换为HSV颜色空间
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 创建黄色和灰色的掩码
            yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

            # 对掩码进行形态学操作，以去除噪声
            kernel = np.ones((5, 5), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

            img = yellow_mask
            # 查找轮廓
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = []

            if len(contours) > 0:
                # 找到最大面积的轮廓
                max_area = 0
                max_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        max_contour = contour

                if isinstance(max_contour, np.ndarray):
                    # 获取面积最大轮廓的凸包
                    hull = cv2.convexHull(max_contour)

                    # 创建一个与原始图像相同大小的空白图像
                    mask = np.zeros_like(frame)

                    # 在空白图像上绘制最大面积轮廓
                    # print(max_contour)
                    cv2.drawContours(mask, [hull], -1, (0, 255, 0), thickness=cv2.FILLED)

                    # 将原始图像和掩码图像进行位运算，将最大面积区域涂色
                    image = cv2.bitwise_and(frame, mask)

                    # 进行多边形逼近，迭代逼近直到达到目标边数
                    epsilon = 0.01  # 初始逼近精度
                    max_iterations = 100  # 最大迭代次数
                    target_num_sides = [3,4,5]
                    for _ in range(max_iterations):
                        approx = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), closed=True)
                        num_sides = len(approx)
                        
                        # 如果逼近得到了目标边数的多边形，绘制并退出循环
                        if num_sides in target_num_sides:
                            # if num_sides == 5:
                            #     cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)  # 绘制红色多边形

                            # else:
                            #     cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  # 绘制绿色多边形
                            break
                        
                        epsilon += 0.01  # 增加逼近精度

                    # print(approx)
                    # 找到直线上最下方的两个点
                    if len(approx) in target_num_sides:
                        frame_data = [10]
                        # frame_data.append(10)
                        bottom_points = sorted(approx, key=lambda point: point[0][1], reverse=True)[:2]

                        # 计算直线的斜率
                        dx = bottom_points[1][0][0] - bottom_points[0][0][0]
                        dy = bottom_points[1][0][1] - bottom_points[0][0][1]
                        # Todo dx非零
                        if dx != 0:
                            slope = dy / dx

                            # 计算夹角
                            angle_rad = math.atan(slope)
                            angle_deg = math.degrees(angle_rad)
                        else:
                            angle_deg = 90

                        # 计算直线的斜率
                        # dlx = bottom_points[1][0][0] + bottom_points[0][0][0]
                        dly = bottom_points[1][0][1] + bottom_points[0][0][1]
                        dheight = int((frame_wh[1] - dly / 2) /frame_wh[1]*255)

                        # print("夹角（弧度）：", angle_rad)
                        # print("夹角（度数）：", f"{angle_deg:.2f}")
                        filtered_value = kalman(angle_deg)
                        # print("夹角（度数）：", f"{filtered_value:.2f}")
                        # print("长度：", dheight)
                        # filtered_values.append(filtered_value)

                        # 将整数转换为4位16进制
                        multiplied_value = int(filtered_value * 100)
                        if multiplied_value >= 0:
                            hex_representation = format(multiplied_value, '04X')
                        else:
                            hex_representation = format((1 << 16) + multiplied_value, '04X')

                        frame_data.append(int(hex_representation[:2],16))
                        frame_data.append(int(hex_representation[2:],16))
                        frame_data.append(dheight)

                        # 拐角识别
                        # guaijiaoshibie(approx)
                        circle_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                        cv2.circle(circle_mask, (int(frame_wh[0]/2-55),int(frame_wh[1]/6)), 40, (255, 255, 255), -1)
                        mean_color = cv2.mean(yellow_mask, mask=circle_mask)[:3][0]
                        guaijiao_check = 0
                        if mean_color == 0:
                            guaijiao_check = 1
                        # frame_data.append(GUAIJIAO)
                        frame_data.append(guaijiao_check)

                        frame_data.append(0)
                        frame_data.append(0)
                        frame_data.append(0)
                        # 接收到来自Block程序的数据
                        # if conn1.poll():
                        # block_data = conn1.recv()
                        # if len(block_data) > 0:
                        #     max_y = max(block_data, key=lambda x: x[2])
                        #     max_y_index = block_data.index(max_y)
                        #     frame_data[5] = block_data[max_y_index][0]
                        #     delta_block_x = int((block_data[max_y_index][1] - frame_wh[0]/2)/frame_wh[0]*127)
                        #     delta_block_y = int((block_data[max_y_index][2] - frame_wh[1]/2)/frame_wh[1]*127)
                        #     if delta_block_x < 0:
                        #         delta_block_x = 255 + delta_block_x
                        #     if delta_block_y < 0:
                        #         delta_block_y = 255 + delta_block_y
                        #     frame_data[6] = delta_block_x
                        #     frame_data[7] = delta_block_y

                        # 发送串口数据
                        frame_data.append(0)
                        # if qrcode_number != '' and len(qrcode_number) == 7:
                        #     frame_data[8] = convert_to_need_numbers(parsing_scanned_qrcode_data(qrcode_number))
                        #     # 为0BUG
                        # other_circles_data = conn2.recv()
                        # for i in range(3):
                        #     for j in range(3):
                        #         if j == 1:
                        #             item_block_circle_data = int(other_circles_data[i][j]/frame_wh[0]*255)
                        #         elif j == 2:
                        #             item_block_circle_data = int(other_circles_data[i][j]/frame_wh[1]*255)
                        #         else:
                        #             item_block_circle_data = other_circles_data[i][j]
                        #         frame_data.append(item_block_circle_data)
                        frame_data.append(13)
                        # print(frame_data[:])
                        if serial_available == 1:
                            send_serial_data(ser,frame_data)
                        else:
                            print(frame_data[:], flush=True)
                        # end_time = time.perf_counter()
                        # print("Left: " + str((end_time-start_time)*1000) + "ms")
    else:
        print("Left摄像头无法打开", flush=True)

if __name__ == '__main__':
    vision_left(False,False)
    # print(convert_to_hex(parsing_scanned_qrcode_data("123+123")))
    # 创建串口对象
    # if system == 'Windows':
    #     port='COM1'
    #     # 初始化摄像头
    #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # elif system == 'Linux':
    #     # port='/dev/ttyUSB0'
    #     # 初始化摄像头
    #     cap = cv2.VideoCapture("/dev/left_video0")
    # else:
    #     print(system, flush=True)
    # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    # ser = serial.Serial(
    #     port="/dev/serial0",
    #     baudrate=115200,  # 波特率，根据实际情况修改
    #     timeout=1  # 超时时间，根据实际情况修改
    # )
    # while True:
    #     ret, frame = cap.read()
    #     update_screen_by_qrcode(frame,ser,1)