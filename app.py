from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import queue
import time
import psutil
import os

app = Flask(__name__)
log_queue = queue.Queue()
running_process = None
running_event = 0

def run_process():
    global running_process, running_event, log_queue
    if running_event == 0:
        cmd = "python main.py"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            bufsize=1,
        )
        running_event = 1
        running_process = process
        # 添加子进程
        for line in iter(process.stdout.readline, ''):
            log_queue.put(line.strip())
        process.stdout.close()
        process.wait()

@app.route('/')
def index():
    return render_template('bootstrap.html')

@app.route('/start')
def start():
    global running_event, log_queue
    log_queue.queue.clear()
    if running_event == 0:
        run_process()
        return jsonify({'message': 'Process started.'})
    else:
        stop()
        start()

@app.route('/stop')
def stop():
    global running_process, running_event, log_queue
    if running_event == 1:
        if os.path.exists("main.pid"):
            with open("main.pid", "r") as file:
                pid = file.read()
                file.close()
            parent = psutil.Process(int(pid))
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait()
            running_process = None
            running_event = 0
            log_queue.queue.clear()
            return jsonify({'message': 'Process stopped.'})
        else:
            return jsonify({'message': 'Process stop failed.'})
    else:
        return jsonify({'message': 'Process not run.'})

@app.route('/clear')
def clear():
    global log_queue
    log_queue.queue.clear()
    return jsonify({'message': 'Log cleared.'})

@app.route('/clear_qrcode')
def clear_qrcode():
    # 检查文件是否存在
    if os.path.exists("qrcode.txt"):
        # 删除文件
        try:
            os.remove("qrcode.txt")
            print("文件删除成功")
        except OSError as e:
            print(f"文件删除失败: {e}")
    else:
        print("文件不存在")
    return jsonify({'message': 'Qrcode cleared.'})

@app.route('/create_qrcode')
def create_qrcode():
    # 检查文件是否存在
    if os.path.exists("qrcode.txt"):
        pass
    else:
        with open("qrcode.txt", "w") as file:
            file.write("123+321")
            file.close()
    return jsonify({'message': 'Qrcode created.'})

@app.route('/get_log')
def get_log():
    log_lines = []
    while not log_queue.empty():
        log_lines.append(log_queue.get_nowait())
    return jsonify({'log': log_lines})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)