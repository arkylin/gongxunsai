import os
import cv2
import time

import torch
from .utils.tool import *
from .module.detector import Detector

category_num = 6
input_width = 352
input_height = 352
LABEL_NAMES = ["green","red","blue","green_box","red_box","blue_box"]
weight = "770-1.pth"
thresh = 0.95

def vision_block(conn):
    # 选择推理后端
    # if opt.cpu:
    #     print("run on cpu...")
    #     device = torch.device("cpu")
    # else:
    #     if torch.cuda.is_available():
    #         print("run on gpu...")
    #         device = torch.device("cuda")
    #     else:
    #         print("run on cpu...")
    #         device = torch.device("cpu")    
    device = torch.device("cpu") 

    # 模型加载
    print("load weight from:%s"%weight)
    model = Detector(category_num, True).to(device)
    model.load_state_dict(torch.load(weight, map_location=device))
    #sets the module in eval node
    model.eval()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        start_time = time.perf_counter()
        # 读取摄像头图像
        ret, frame = cap.read()
        # 数据预处理
        ori_img = frame.copy()
        res_img = cv2.resize(ori_img, (input_width, input_height), interpolation = cv2.INTER_LINEAR) 
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
        
        H, W, _ = ori_img.shape
        scale_h, scale_w = H / input_height, W / input_width

        # 绘制预测框
        for box in output[0]:
            print(box)
            end_time = time.perf_counter()
            print("block" + str((end_time-start_time)*1000) + "ms")
    #         box = box.tolist()
        
    #         obj_score = box[4]
    #         category = LABEL_NAMES[int(box[5])]

    #         x1, y1 = int(box[0] * W), int(box[1] * H)
    #         x2, y2 = int(box[2] * W), int(box[3] * H)

    #         cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    #         cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
    #         cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    #     cv2.imshow("R", ori_img)
    #     # 按下 'q' 键退出循环
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # 释放摄像头资源
    # cap.release()
    # cv2.destroyAllWindows()