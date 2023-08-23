import serial

def send_command(conn):
    # while True:
    #     echo = conn.recv()
    #     if echo != "":
    #         print(conn.recv()+"1")
    # 创建串口对象
    ser = serial.Serial(
        # port='/dev/ttyUSB0',  # 串口设备号，根据实际情况修改
        port='COM1',
        baudrate=115200,  # 波特率，根据实际情况修改
        timeout=1  # 超时时间，根据实际情况修改
    )
    while True:
        # print(conn.recv())
        ser.write(conn.recv().encode())
        # ser.close()
        # conn.close()

# if __name__ == '__main__':
    # ser.write("Hello".encode())