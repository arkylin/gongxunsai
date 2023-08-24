# 2023 工训赛树莓派源码
## 安装
### 克隆代码仓库
```
git clone http://192.168.1.109:3000/x/gongxunsai
```
### pip安装依赖
```
pip install -r requirements.txt
```
### 拷贝摄像头命令规则（可选）
运行命令
```
lsusb
```
找到所需摄像头的id，例如``Bus 001 Device 005: ID 2993:0858 RYS USB4K CAMERA``
修改gongxunsai.rules文件
```
sudo cp gongxunsai.rules /etc/udev/rules.d
```
## 运行
```
python main.py
```