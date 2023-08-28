import os
import cv2
import platform
import time

import torch
from .utils.tool import *
from .module.detector import Detector

system = platform.system()

category_num = 6
input_width = 352
input_height = 352
LABEL_NAMES = ["green","red","blue","green_box","red_box","blue_box"]
weight = "vision_block/770-1.pth"
thresh = 0.95
frame_wh = (400,300)

def vision_block(conn):
    # 选择推理后端  
    device = torch.device("cpu") 
    # 模型加载
    # print("load weight from:%s"%weight)
    model = Detector(category_num, True).to(device)
    model.load_state_dict(torch.load(weight, map_location=device))
    #sets the module in eval node
    model.eval()
    print("开始视觉识别")
    
    # 初始化摄像头
    if system == 'Windows':
        # 初始化摄像头
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif system == 'Linux':
        # 初始化摄像头
        cap = cv2.VideoCapture("/dev/block_video0")
    else:
        print(system)
        
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    if cap.isOpened():
        while True:
            # start_time = time.perf_counter()
            # 读取摄像头图像
            ret, frame = cap.read()
            send_data = []
            # 数据预处理
            res_img = cv2.resize(frame, (input_width, input_height), interpolation = cv2.INTER_LINEAR) 
            img = res_img.reshape(1, input_height, input_width, 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(device).float() / 255.0

            # 模型推理
            # start = time.perf_counter()
            preds = model(img)
            # end = time.perf_counter()
            # time = (end - start) * 1000.
            # print("forward time:%fms"%time)

            # 特征图后处理
            output = handle_preds(preds, device, thresh)
            
            H, W = frame_wh
            scale_h, scale_w = H / input_height, W / input_width

            # 绘制预测框
            for box in output[0]:
                # print(box)
                # end_time = time.perf_counter()
                # print("block" + str((end_time-start_time)*1000) + "ms")
                box = box.tolist()
            
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                x1, y1 = int(box[0] * W), int(box[1] * H)
                x2, y2 = int(box[2] * W), int(box[3] * H)

                # print([category, int(obj_score*100), x1, y1, x2, y2])
                # conn.send("123")
                # int(obj_score*100) # 置信度
                send_data.append([category, int((x1+x2)/2), int((y1+y2)/2)])
            if len(send_data) !=0:
                # time.sleep(1)
                conn.send(send_data)
    else:
        print("Block摄像头无法打开")