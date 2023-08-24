## 1
Traceback (most recent call last):
  File "C:\Users\x\miniconda3\envs\gongxunsai\lib\multiprocessing\process.py", line 315, in _bootstrap
    self.run()
  File "C:\Users\x\miniconda3\envs\gongxunsai\lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "c:\Users\x\Code\lunaaaaaaa\main\vision_left\main.py", line 219, in vision_left
    if abs(old_value_y-max_value_y) < 20 and abs(old_value_x-last_x_data) < 20:
TypeError: unsupported operand type(s) for -: 'list' and 'int'

## 2
dx / dy