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