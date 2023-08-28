import serial #导入模块
try:
  #端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
    portx="COM3"
  #波特率，标准值：2400 4800 9600 19200 38400 57600 115200 230400 256000 512000 921600
    bps=115200
  #超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
    timex=0
  # 打开串口，并得到串口对象
    ser=serial.Serial(portx,bps,timeout=timex)
    print("串口详情参数：", ser)
  # 写数据
    #注意：编码应与工程的编码一致（设备-字符编码，如果usarthmi中设置的为utf-8，则这里也填写utf8，如果usarthmi中设置的为GB2312，则这里也填写GB2312也可以填写GBK）
    # result=ser.write("main".encode("utf-8"))
    #result=ser.write("page page2".encode("GB2312"))
    result=ser.write("t0.txt=\"312+132\"".encode())
  # 发送结束符
    ser.write(bytes.fromhex('ff ff ff'))

    ser.close()#关闭串口

except Exception as e:
    print("---异常---：",e)