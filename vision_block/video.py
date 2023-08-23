import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector

category_num = 6
input_width = 352
input_height = 352
LABEL_NAMES = ["green","red","blue","green_box","red_box","blue_box"]

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="770-1.pth", help='.weight config')
    parser.add_argument('--thresh', type=float, default=0.95, help='The path of test image')
    parser.add_argument('--cpu', action="store_true", default=False, help='Run on cpu')

    opt = parser.parse_args()
    assert os.path.exists(opt.weight), "请指定正确的模型路径"

    # 选择推理后端
    if opt.cpu:
        print("run on cpu...")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("run on gpu...")
            device = torch.device("cuda")
        else:
            print("run on cpu...")
            device = torch.device("cpu")     

    # 模型加载
    print("load weight from:%s"%opt.weight)
    model = Detector(category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    #sets the module in eval node
    model.eval()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
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
        output = handle_preds(preds, device, opt.thresh)
        
        H, W, _ = ori_img.shape
        scale_h, scale_w = H / input_height, W / input_width

        # 绘制预测框
        for box in output[0]:
            print(box)
            box = box.tolist()
        
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        cv2.imshow("R", ori_img)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()