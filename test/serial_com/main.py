import serial

def send_cmd():
    try:
        ser = serial.Serial(
            # port='/dev/ttyUSB0',  # 串口设备号，根据实际情况修改
            port='COM1',
            baudrate=115200,  # 波特率，根据实际情况修改
            timeout=1  # 超时时间，根据实际情况修改
        )
        ser.write("Hello".encode())
    except:
        print("Error Hello")
ser = serial.Serial(
        # port='/dev/ttyUSB0',  # 串口设备号，根据实际情况修改
        port='COM1',
        baudrate=115200,  # 波特率，根据实际情况修改
        timeout=1  # 超时时间，根据实际情况修改
    )
def send_cd():

    ser.write("Hello".encode())