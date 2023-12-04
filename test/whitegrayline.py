import cv2
import numpy as np

# 白色范围
lower_white = np.array([0, 0, 221])
upper_white = np.array([180, 30, 255])

# 灰色范围
lower_gray = np.array([0, 0, 46])
upper_gray = np.array([180, 43, 255])


# 初始化摄像头
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_frame,lower_white,upper_white)
    gray_mask = cv2.inRange(hsv_frame,lower_gray,upper_gray)

    cv2.imshow("1",white_mask)
    cv2.imshow("2",gray_mask)


    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
