import cv2
from pyzbar import pyzbar

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

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    print(decode_qr_code(frame))
    
    # cv2.imshow("123", frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()

