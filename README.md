# 2023 工训赛树莓派源码
git remote set-url origin http://:3000/x/gongxunsai
git clone http://:3000/x/gongxunsai
## 安装
### 克隆代码仓库
```
git clone http://192.168.1.109:3000/x/gongxunsai
```
### 安装扫描二维码依赖
```
sudo apt install libzbar0
```
### 切换软件源
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
### pip安装依赖
```
pip install -r requirements.txt
```
### 设置开机自启动程序
```
sudo cp gongxunsai.service /etc/systemd/system/
sudo systemctl start gongxunsai.service
sudo systemctl enable gongxunsai.service
```
### 修正工训赛文件
sudo cp gongxunsai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart gongxunsai.service
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
