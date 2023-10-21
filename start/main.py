import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

while True:
    ret, frame = cap.read()
    cv2.imshow("123", frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
