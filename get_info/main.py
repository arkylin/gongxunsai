import socket

def getNum(ip="127.0.0.1", remote_port=8888, local_port=7777, tu="udp"):
    if tu == "tcp":
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    elif tu == 'udp':
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.bind(("0.0.0.0", local_port))

    # 设置超时时间为 3 秒
    # client_socket.settimeout(3.0)

    try:
        # 绑定服务器地址和端口
        server_address = (ip, remote_port)
        # print(server_address)
        # 连接服务器
        client_socket.connect(server_address)
        if tu == "tcp":
            response = client_socket.recv(1024)
        elif tu == 'udp':
            response = client_socket.recvfrom(1024)[0]
        response = response.decode()
        if len(response) == 7:
            return response
        else:
            return 0
    except socket.timeout:
        print("连接超时，自动停止运行")
        return 1
    finally:
        # 关闭套接字
        client_socket.close()
if __name__ == '__main__':
    print(getNum(ip="192.168.31.49",remote_port=8080,local_port=7777,tu="udp"))