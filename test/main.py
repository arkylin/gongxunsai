# -*- coding: UTF-8 -*-
import multiprocessing
import psutil

import serial_com.main.send_cd

if __name__ == '__main__':
    # 创建多个进程，并将它们绑定到指定的CPU核心
    processes = []

    p_angle = multiprocessing.Process(target=serial_com.main.send_cd)
    # os.sched_setaffinity(p.pid,[i]) # 绑定到第i个CPU核心
    p_angle.start()
    psutil.Process(p_angle.pid).cpu_affinity([1])
    processes.append(p_angle)

    # 等待所有进程完成
    for p in processes:
        p.join()
