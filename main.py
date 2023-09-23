# -*- coding: UTF-8 -*-
import multiprocessing
import psutil
import os

import vision_left.main
import vision_block.main

if __name__ == '__main__':
    with open('main.pid', 'w') as pid_file:
        pid_file.write(str(os.getpid()))

    # 创建多个进程，并将它们绑定到指定的CPU核心
    processes = []

    conn1, conn2 = multiprocessing.Pipe()
    conn3, conn4 = multiprocessing.Pipe()

    p_angle = multiprocessing.Process(target=vision_left.main.vision_left,args=(conn1,conn3,))
    # os.sched_setaffinity(p.pid,[i]) # 绑定到第i个CPU核心
    p_angle.daemon = True
    p_angle.start()
    psutil.Process(p_angle.pid).cpu_affinity([0])
    processes.append(p_angle)

    p_block = multiprocessing.Process(target=vision_block.main.vision_block,args=(conn2,conn4,))
    # os.sched_setaffinity(p.pid,[i]) # 绑定到第i个CPU核心
    p_block.daemon = True
    p_block.start()
    psutil.Process(p_block.pid).cpu_affinity([1])
    processes.append(p_block)

    # 等待所有进程完成
    for p in processes:
        p.join()
