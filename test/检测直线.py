import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(1)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("1",edges)

    # 使用霍夫直线变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=180)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # 避免除以零错误
            if (x2 - x1) != 0:
                slope = (y2 - y1) / (x2 - x1)
                print("Point 1: ({}, {}), Point 2: ({}, {}), Slope: {}".format(x1, y1, x2, y2, slope))
            else:
                slope = float('inf')  # 代表垂直线

            # 在原始图像上绘制直线
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('Frame', frame)

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
