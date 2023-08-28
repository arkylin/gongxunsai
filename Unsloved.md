## 1
Traceback (most recent call last):
  File "C:\Users\x\miniconda3\envs\gongxunsai\lib\multiprocessing\process.py", line 315, in _bootstrap
    self.run()
  File "C:\Users\x\miniconda3\envs\gongxunsai\lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "c:\Users\x\Code\lunaaaaaaa\main\vision_left\main.py", line 219, in vision_left
    if abs(old_value_y-max_value_y) < 20 and abs(old_value_x-last_x_data) < 20:
TypeError: unsupported operand type(s) for -: 'list' and 'int'

## 2 (Sloved)
dx / dy

## 3
>>> cap = cv2.VideoCapture("/dev/block_video")
[ERROR:0@3.260] global cap.cpp:164 open VIDEOIO(CV_IMAGES): raised OpenCV exception:

OpenCV(4.8.0) /io/opencv/modules/videoio/src/cap_images.cpp:253: error: (-5:Bad argument) CAP_IMAGES: can't find starting number (in the name of file): /dev/block_video in function 'icvExtractPattern'

## 4 (Sloved)
KERNEL=="video*", ATTRS{idVendor}=="05a3", ATTRS{idProduct}=="9230", ATTR{index}=="0", MODE:="0777", SYMLINK+="left_video0"
KERNEL=="video*", ATTRS{idVendor}=="2993", ATTRS{idProduct}=="0858", ATTR{index}=="0", MODE:="0777", SYMLINK+="block_video0"
sudo udevadm control --reload-rules

## 5 (Sloved)
Traceback (most recent call last):
  File "/usr/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/pi/gongxunsai/vision_left/main.py", line 229, in vision_left
    delta_block_x = int((block_data[0][1] - frame_wh[0]/2)/frame_wh[0]*127)
TypeError: unsupported operand type(s) for -: 'tuple' and 'float'

## 5
比赛的时候二维码挡板脚为红色、绿色、蓝色

## 6
帮我编写一个python程序，使用flask创建一个网页，点击开始按钮实现运行“python main.py”（这个程序是一直循环运行的不会结束），并将这个循环运行的程序的输出（日志）显示在网页上（要求日志显示在一个固定的框里面不要出来），具有实时刷新的功能，刷新的时候保留原有的日志只刷新新的行。点击结束按钮需要停止运行程序，并且清空显示的日志，还有一个按钮可以直接清除日志。