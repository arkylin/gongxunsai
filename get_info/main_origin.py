import socket

def getNum(ip="127.0.0.1",remote_port=8888,local_port=7777,tu="udp"):
    if tu == "tcp":
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    elif tu == 'udp':
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.bind(("127.0.0.1",local_port))
    
    # 绑定服务器地址和端口
    server_address = (ip, remote_port)
    # 连接服务器
    client_socket.connect(server_address)
    if tu == "tcp":
        response = client_socket.recv(1024)
    elif tu == 'udp':
        response = client_socket.recvfrom(1024)[0]
    response = response.decode()
    # print(type(response),response)
    if len(response) == 7:
        return response
    else:
        return 0

if __name__ == '__main__':
    getNum(ip="127.0.0.1",remote_port=8888,local_port=7777,tu="udp")