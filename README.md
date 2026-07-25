# 🎯 多人实时人脸识别考勤系统

基于 Python + OpenCV + dlib 的多人实时人脸识别系统，支持桌面版和 Web 版两种形态。

## ✨ 功能特性

- **多人实时识别**：支持同时识别多张人脸
- **68 特征点定位**：dlib 68 个人脸关键点精准定位
- **128 维特征提取**：ResNet 模型提取人脸特征向量
- **IoU 人脸跟踪**：跳帧识别时框不跳动
- **动态跳帧优化**：根据人脸数量自适应调整识别频率
- **人脸对齐**：根据双眼关键点做仿射变换，消除头部倾斜影响
- **批量导入**：支持 Excel 学生名单批量导入
- **考勤记录**：自动记录打卡时间，支持 CSV 导出

## 📁 项目结构

```
face-recognition-system/
├── desktop_version/          # 桌面版（tkinter GUI）
│   ├── face_recognize_desktop.py
│   ├── face_features/
│   └── README.md
├── web_version/              # Web 版（Flask）
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── face_features/
├── models/                   # dlib 模型文件
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
├── data/                     # 数据目录
├── docs/                     # 文档
├── requirements.txt          # Python 依赖
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Windows 10/11

### 安装依赖

```bash
# 创建虚拟环境
conda create -n face python=3.11 -y
conda activate face

# 安装依赖
pip install -r requirements.txt
```

### 运行桌面版

```bash
cd desktop_version
python face_recognize_desktop.py
```

### 运行 Web 版

```bash
cd web_version
python app.py
```

浏览器访问 http://localhost:5000

## 📊 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.11 | 主要编程语言 |
| dlib | 人脸检测、特征点定位、特征提取 |
| OpenCV | 图像处理、摄像头采集 |
| Flask | Web 服务器 |
| NumPy | 数值计算 |
| Pandas | 数据处理、Excel 导入 |
| Pillow | 中文文字渲染 |
| tkinter | 桌面 GUI |

## 🔧 核心算法

### 1. 人脸检测
```python
detector = dlib.get_frontal_face_detector()
faces = detector(rgb_frame)
```

### 2. 特征点定位
```python
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
landmarks = predictor(rgb_frame, face)
```

### 3. 特征提取
```python
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
face_descriptor = face_rec.compute_face_descriptor(rgb_frame, landmarks)
```

### 4. 特征匹配
```python
def find_best_match(target_feature, normalized_features, threshold=0.65):
    # 计算余弦相似度
    similarities = feat_matrix @ target_normalized
    max_sim = float(np.max(similarities))
    return best_match, best_similarity
```

## 📈 性能优化

- **多线程架构**：CameraStream 独立线程采集，Flask 主线程处理请求
- **双缓冲读帧**：只保留最新帧，旧帧自动丢弃
- **动态跳帧**：根据人脸数量自适应调整识别频率
- **图像缩放**：处理时缩放到 320px，降低计算量
- **IoU 跟踪**：跳帧识别时框不跳动

## 🎓 学习成果

- 掌握 dlib 人脸检测和特征提取
- 理解多线程编程和线程同步
- 学会 Flask Web 开发
- 了解图像处理和计算机视觉基础

## 📝 更新日志

### v1.0.0 (2025-07)
- 初始版本发布
- 桌面版和 Web 版
- 支持多人实时识别
- 支持批量导入和考勤导出

## 📄 License

MIT License

## 👨‍💻 作者

张琪 - AI 应用开发工程师

---

> 💡 这是我的第一个完整的 AI 项目，从模型调用到 Web 部署的完整链路。
