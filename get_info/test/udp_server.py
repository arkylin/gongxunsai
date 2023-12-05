import socket

# 创建一个TCP/IP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定服务器地址和端口
server_address = ('0.0.0.0', 8888)
server_socket.bind(server_address)

print('服务器正在运行，等待连接...')

while True:
    # 等待客户端连接
    # client_socket, client_address = server_socket.accept()
    # print('客户端已连接:', client_address)
    response = "213+213"
    server_socket.sendto(response.encode(), ('192.168.1.103',7777))