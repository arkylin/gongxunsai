import cv2
import numpy as np

def on_trackbar_change(pos):
    pass  # 滑块回调函数，此处暂时不执行任何操作

# 创建一幅空白图像作为背景
image = np.zeros((300, 300, 3), dtype=np.uint8)

# 创建窗口
cv2.namedWindow('HSV Color Slider')

# 创建H、S和V的滑块
cv2.createTrackbar('H', 'HSV Color Slider', 18, 179, on_trackbar_change)
cv2.createTrackbar('S', 'HSV Color Slider', 23, 255, on_trackbar_change)
cv2.createTrackbar('V', 'HSV Color Slider', 219, 255, on_trackbar_change)

while True:
    # 获取滑块的当前值
    h = cv2.getTrackbarPos('H', 'HSV Color Slider')
    s = cv2.getTrackbarPos('S', 'HSV Color Slider')
    v = cv2.getTrackbarPos('V', 'HSV Color Slider')
    
    # 创建一个HSV颜色
    hsv_color = np.array([h, s, v], dtype=np.uint8)
    bgr_color = cv2.cvtColor(np.array([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # 更新图像的颜色
    image[:] = bgr_color
    
    # 显示图像
    cv2.imshow('HSV Color Slider', image)
    
    # 检查退出条件
    if cv2.waitKey(10) & 0xFF == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()
