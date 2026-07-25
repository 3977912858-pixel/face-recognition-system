安装步骤（用 Anaconda）
第一步：安装 Anaconda
打开 https://www.anaconda.com/download
下载 Windows 版，安装
安装时勾选 "Add Anaconda to PATH"（虽然它不推荐，但勾上更方便）
安装完成后打开 Anaconda Prompt（开始菜单搜索 "Anaconda"）
第二步：创建虚拟环境
在 Anaconda Prompt 中依次输入：


conda create -n face python=3.11 -y
conda activate face
第三步：安装依赖

conda install -c conda-forge dlib -y
pip install opencv-python numpy Pillow
第四步：运行

cd 你解压文件的路径
python face_recognize_desktop.py
例如文件放在桌面：


cd C:\Users\你的用户名\Desktop\作业
python face_recognize_desktop.py
安装验证
如果不确定装好了没有，在 Anaconda Prompt 中输入：


python -c "import dlib; print('dlib OK')"
python -c "import cv2; print('opencv OK')"
都显示 OK 就能跑了。