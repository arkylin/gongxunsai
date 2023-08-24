import cv2

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    cv2.imshow("123", frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
