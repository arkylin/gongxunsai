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
stop_event = threading.Event()

def run_process():
    global running_process
    while not stop_event.is_set():
        cmd = "python main.py"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            bufsize=1,
        )
        running_process = process
        for line in iter(process.stdout.readline, ''):
            log_queue.put(line.strip())
        process.stdout.close()
        process.wait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    global stop_event
    stop_event.clear()
    thread = threading.Thread(target=run_process)
    thread.daemon = True
    thread.start()
    return jsonify({'message': 'Process started.'})

@app.route('/stop')
def stop():
    global stop_event, running_process
    stop_event.set()
    if running_process:
        parent = psutil.Process(running_process.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        parent.wait()
        running_process = None
    return jsonify({'message': 'Process stopped.'})

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

@app.route('/get_log')
def get_log():
    log_lines = []
    while not log_queue.empty():
        log_lines.append(log_queue.get_nowait())
    return jsonify({'log': log_lines})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)