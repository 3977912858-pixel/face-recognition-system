# -*- coding: utf-8 -*-
import os
import sys

# ============================================================
# 启动前置校验：模型文件 + 目录
# ============================================================
REQUIRED_MODELS = [
    'shape_predictor_68_face_landmarks.dat',# 68个人脸关键点预测模型（用于定位眼睛、五官）
    'dlib_face_recognition_resnet_model_v1.dat',# 人脸特征提取模型（生成128维人脸向量做相似度比对）
]
_missing = [f for f in REQUIRED_MODELS if not os.path.isfile(f)]
if _missing:
    print("=" * 55)
    print("  [启动失败] 缺少以下 dlib 模型文件：")
    for f in _missing:
        print(f"    ✗ {f}")
    print("  请将模型文件放到 app.py 同级目录后重试。")
    print("=" * 55)
    sys.exit(1)

REQUIRED_DIRS = [
    'face_features',# 存储所有人脸特征向量txt文件，每个学生一个独立文件夹
    'static/uploads', # 临时存放用户上传的人脸照片
    'static/user_images',# 永久保存录入的学生原始人脸照片
    'templates',# 存放前端网页html模板
]
for _d in REQUIRED_DIRS:
    os.makedirs(_d, exist_ok=True)
if not os.path.isfile('templates/index.html'):
    print("=" * 55)
    print("  [启动失败] 缺少前端页面文件：templates/index.html")
    print("=" * 55)
    sys.exit(1)

# ============================================================
# 依赖导入
# ============================================================
import cv2
import dlib
import numpy as np
import shutil
import threading
import time
import logging
import pandas as pd  # 读取Excel学生名单
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

# 日志配置：同时输出到控制台+本地日志文件face_recognition.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceRecognition")

# 初始化Flask网页服务对象
app = Flask(__name__)
# 配置上传、用户图片存储路径
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['USER_IMAGES_FOLDER'] = 'static/user_images'
# 允许上传的图片格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# 确保上传文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['USER_IMAGES_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 全局变量
is_recognizing = False# 全局开关：True开启人脸识别+考勤，False仅预览画面
attendance_records = {}# 考勤记录存储字典 {学生姓名:{日期:打卡时间}}
system_status = {
    "user_count": 0,
    "attendance_count": 0,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "recognition_status": "待机",
    "camera_status": "未连接",
    "system_uptime": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "frame_rate": 0
}


# ============================================================
# 人脸识别器类
# ============================================================
class FaceRecognizer:
    def __init__(self):
        try:
            self.detector = dlib.get_frontal_face_detector()# 加载 dlib 人脸检测器（替代 Haar 级联分类器）
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')# 加载 68 个人脸关键点定位模型
            self.face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')# 加载 128 维人脸特征提取模型（ResNet）
            logger.info("人脸识别模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

        self.process_width = 320
        self.features_dict = self.load_face_features()
        self.normalized_features = self._build_normalized_features()  # 预计算L2归一化特征
        self.font = None
        self.small_font = None
        self.init_font()

    def init_font(self):
         # 遍历系统常用中文字体路径，自动匹配可用字体
        self.chinese_available = False
        font_paths = [
            'simhei.ttf', 'simsun.ttc', 'msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simsun.ttc',
            'C:/Windows/Fonts/msyh.ttc',
        ]
        for font_path in font_paths:
            try:
                self.font = ImageFont.truetype(font_path, 32)
                self.chinese_available = True
                logger.info(f"字体加载成功: {font_path}")
                break
            except:
                continue
        if self.font is None:
            self.font = ImageFont.load_default()
            logger.warning("未找到中文字体，状态栏自动切换为英文显示")

        for font_path in font_paths:
            try:
                self.small_font = ImageFont.truetype(font_path, 22)
                break
            except:
                continue
        if self.small_font is None:
            self.small_font = self.font

    def _build_normalized_features(self):
        """预计算所有特征的L2归一化矩阵，用于向量化匹配"""
        normalized = {}
        for person_name, feat_list in self.features_dict.items():
            if not feat_list:
                continue
            matrix = np.array(feat_list, dtype=np.float64)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized[person_name] = matrix / norms
        return normalized

    def align_face(self, rgb_frame, landmarks, desired_size=150):
        """
        人脸对齐：根据双眼位置做仿射变换，将人脸旋转到水平标准位置
        消除头部倾斜对特征提取的影响
        """
        # 左眼中心（关键点 36-41 的平均值）
        left_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(36, 42)]
        left_eye_center = (
            int(np.mean([p[0] for p in left_eye_pts])),
            int(np.mean([p[1] for p in left_eye_pts]))
        )
        # 右眼中心（关键点 42-47 的平均值）
        right_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(42, 48)]
        right_eye_center = (
            int(np.mean([p[0] for p in right_eye_pts])),
            int(np.mean([p[1] for p in right_eye_pts]))
        )
        # 计算两眼连线与水平线的夹角
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        # 计算缩放比例
        eye_distance = np.sqrt(dX**2 + dY**2)
        desired_eye_distance = desired_size * 0.35
        scale = desired_eye_distance / eye_distance if eye_distance > 0 else 1.0
        # 以两眼中心的中点为旋转中心
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) // 2,
            (left_eye_center[1] + right_eye_center[1]) // 2
        )
        # 构建仿射变换矩阵：旋转 + 缩放
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        # 调整平移，使两眼中心移到输出图像的固定位置
        tX = desired_size * 0.5
        tY = desired_size * 0.35
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]
        # 执行仿射变换
        aligned = cv2.warpAffine(rgb_frame, M, (desired_size, desired_size),
                                  flags=cv2.INTER_CUBIC)
        return aligned

    def load_face_features(self):
        # 读取face_features文件夹下所有学生人脸特征，加载到内存字典
        # 兼容两种格式：原有feature_*.txt（手动录入）和 姓名_学号.npy（批量导入）
        features_dict = {}
        base_dir = 'face_features'
        try:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                return features_dict
            # 遍历每个学生文件夹（文件夹名称=学生姓名）
            for person_name in os.listdir(base_dir):
                person_dir = os.path.join(base_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue
                features_dict[person_name] = []
                # 读取该学生所有特征文件
                for feature_file in os.listdir(person_dir):
                    feature_path = os.path.join(person_dir, feature_file)
                    if not os.path.isfile(feature_path):
                        continue
                    try:
                        # 【兼容】支持原有txt格式和新增npy格式
                        if feature_file.endswith('.txt') and feature_file.startswith('feature_'):
                            feature_vector = np.loadtxt(feature_path)
                            features_dict[person_name].append(feature_vector)
                        elif feature_file.endswith('.npy'):
                            feature_data = np.load(feature_path)
                            if feature_data.ndim == 1:
                                # 单个128维向量，直接存入
                                features_dict[person_name].append(feature_data)
                            elif feature_data.ndim == 2 and feature_data.shape[1] == 128:
                                # (N, 128) 矩阵：拆分为 N 个独立向量逐个存入
                                for i in range(feature_data.shape[0]):
                                    features_dict[person_name].append(feature_data[i])
                            else:
                                logger.warning(f"异常特征形状{feature_data.shape}: {feature_path}")
                    except Exception as e:
                        logger.error(f"加载特征文件错误: {feature_path} - {str(e)}")
            system_status["user_count"] = len(features_dict)# 更新系统学生总数
            logger.info(f"特征库加载完成，用户数量: {len(features_dict)}")
            return features_dict
        except Exception as e:
            logger.error(f"加载特征库失败: {str(e)}")
            return {}

    def process_frame(self, frame):
        # 核心帧处理函数：接收摄像头原图，完成人脸检测、68关键点、人眼坐标、人脸匹配
        # 返回结果列表：[(人脸框,匹配姓名,相似度,左眼矩形,右眼矩形)]
        try:
            height, width = frame.shape[:2]
            # 计算缩放比例，缩小图像降低运算量
            scale = self.process_width / width
            if scale < 1:
                small_frame = cv2.resize(frame, (self.process_width, int(height * scale)))
            else:
                small_frame = frame
                scale = 1.0
            # dlib仅支持RGB图像，opencv读取默认BGR，需要转换通道
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # 检测画面中所有人脸
            faces = self.detector(rgb_frame)
            results = []
            # 缩放因子：坐标从小图还原回原图时使用
            inv_scale = 1.0 / scale if scale < 1 else 1.0
            # 循环处理每一张检测到的人脸
            for face in faces:
                try:
                    landmarks = self.predictor(rgb_frame, face)# 获取68个面部关键点
                    # 人脸对齐：用双眼中心做仿射变换，消除头部倾斜影响
                    aligned_face = self.align_face(rgb_frame, landmarks, desired_size=150)
                    aligned_faces = self.detector(aligned_face)
                    if aligned_faces:
                        aligned_landmarks = self.predictor(aligned_face, aligned_faces[0])
                        face_descriptor = self.face_rec.compute_face_descriptor(aligned_face, aligned_landmarks)
                    else:
                        # 对齐后检测失败，回退到原始方法
                        face_descriptor = self.face_rec.compute_face_descriptor(rgb_frame, landmarks)
                    target_feature = np.array(face_descriptor)
                    best_match, similarity = self.find_best_match(target_feature) # 和特征库比对，匹配学生

                    # 截取68点中眼睛区域坐标：左眼36-41，右眼42-47
                    left_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(36, 42)]
                    right_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(42, 48)]

                    # 计算左眼外接矩形最小/最大坐标
                    le_x = [p[0] for p in left_eye_pts]
                    le_y = [p[1] for p in left_eye_pts]
                    left_eye_rect = (min(le_x), min(le_y), max(le_x), max(le_y))

                    re_x = [p[0] for p in right_eye_pts]
                    re_y = [p[1] for p in right_eye_pts]
                    right_eye_rect = (min(re_x), min(re_y), max(re_x), max(re_y))
                    # 如果图像做了缩小，把人脸、眼睛坐标还原回原图尺寸
                    if scale < 1:
                        face = dlib.rectangle(
                            round(face.left() * inv_scale), round(face.top() * inv_scale),
                            round(face.right() * inv_scale), round(face.bottom() * inv_scale)
                        )
                        left_eye_rect = tuple(round(v * inv_scale) for v in left_eye_rect)
                        right_eye_rect = tuple(round(v * inv_scale) for v in right_eye_rect)
                        # 存入结果列表，统一返回给画面绘制函数
                    results.append((face, best_match, similarity, left_eye_rect, right_eye_rect))
                except Exception as e:
                    logger.warning(f"处理人脸时出错: {str(e)}")
                    continue
            return results
        except Exception as e:
            logger.error(f"帧处理失败: {str(e)}")
            return []

    def compute_similarity(self, feature1, feature2):
        # 余弦相似度计算：对比两个人脸128维向量相似程度，数值越高长得越像
        try:
            return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        except:
            return 0.0

    def find_best_match(self, target_feature, threshold=0.95):
        """
        向量化匹配：预计算L2归一化 + 矩阵乘法，一次完成所有特征的余弦相似度
        阈值选择依据（本系统基准测试结果）：
          同一人相似度范围：0.77 ~ 1.00
          不同人相似度范围：0.75 ~ 0.98（学籍照背景相似导致重叠严重）
          当前取 0.95：严格匹配，只有高度相似才通过，宁可全部显示陌生人也不认错人
          可调用 benchmark_threshold() 实测调优
        """
        best_match = None
        best_similarity = -1
        try:
            if not self.normalized_features:
                return "陌生人", 0.0
            # L2归一化目标向量
            norm = np.linalg.norm(target_feature)
            if norm == 0:
                return "陌生人", 0.0
            target_normalized = target_feature / norm
            # 一次矩阵乘法完成所有匹配：(N,128) @ (128,) -> (N,) 余弦相似度
            for person_name, feat_matrix in self.normalized_features.items():
                similarities = feat_matrix @ target_normalized
                max_sim = float(np.max(similarities))
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_match = person_name
            # 最高相似度低于阈值，判定陌生人
            if best_similarity < threshold:
                return "陌生人", best_similarity
        except Exception as e:
            logger.error(f"匹配失败: {str(e)}")
        return best_match, best_similarity

    def benchmark_threshold(self):
        """
        阈值基准测试：用自身特征库做交叉验证
        计算不同阈值下的 FAR(误识率) 和 FRR(拒识率)，输出推荐阈值
        在命令行运行：python -c "from app import recognizer; recognizer.benchmark_threshold()"
        """
        print("=" * 55)
        print("  阈值基准测试")
        print("=" * 55)

        # 收集所有特征，标注所属人员
        all_features = []
        for person_name, feat_list in self.features_dict.items():
            for feat in feat_list:
                all_features.append((person_name, feat))

        if len(all_features) < 2:
            print("  特征数量不足，无法测试")
            return 0.65

        # 生成同人对和异人对的相似度
        same_pairs = []
        diff_pairs = []
        print(f"  正在计算 {len(all_features)} 个特征的交叉相似度...")
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                sim = self.compute_similarity(all_features[i][1], all_features[j][1])
                if all_features[i][0] == all_features[j][0]:
                    same_pairs.append(sim)
                else:
                    diff_pairs.append(sim)

        if not same_pairs or not diff_pairs:
            print("  同人对或异人对为空，无法测试")
            return 0.65

        print(f"  同人对数量: {len(same_pairs)}")
        print(f"  异人对数量: {len(diff_pairs)}")
        print(f"  同人相似度范围: {min(same_pairs):.4f} ~ {max(same_pairs):.4f}")
        print(f"  异人相似度范围: {min(diff_pairs):.4f} ~ {max(diff_pairs):.4f}")
        print()
        print(f"  {'阈值':>6}  {'FAR(误识)':>10}  {'FRR(拒识)':>10}  {'准确率':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

        best_threshold = 0.65
        best_acc = 0

        for t in [x * 0.01 for x in range(50, 80)]:
            far = sum(1 for s in diff_pairs if s >= t) / len(diff_pairs)
            frr = sum(1 for s in same_pairs if s < t) / len(same_pairs)
            acc = 1 - (far + frr) / 2
            marker = " <-- 当前" if abs(t - 0.65) < 0.005 else ""
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
            print(f"  {t:>6.2f}  {far:>10.2%}  {frr:>10.2%}  {acc:>8.2%}{marker}")

        print()
        print(f"  >>> 推荐阈值: {best_threshold:.2f} (准确率 {best_acc:.2%})")
        print("=" * 55)
        return best_threshold

    def save_multiple_face_features(self, images, name):
        # 批量保存人脸特征：新增学生/批量导入时调用，存储原图、人脸截图、特征txt
        try:
            person_dir = os.path.join('face_features', name)
            os.makedirs(person_dir, exist_ok=True)
            user_image_dir = os.path.join(app.config['USER_IMAGES_FOLDER'], name)
            os.makedirs(user_image_dir, exist_ok=True)
            # 循环处理该学生每一张上传图片
            for i, img_data in enumerate(images):
                # 保存原始上传照片
                cv2.imwrite(os.path.join(user_image_dir, f'original_{i + 1}.jpg'), img_data)
                rgb_frame = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                faces = self.detector(rgb_frame)
                if not faces:
                    logger.warning(f"图片 {i + 1} 中未检测到人脸")
                    continue
                # 一张图多个人脸，取面积最大人脸作为目标人脸
                face = max(faces, key=lambda r: r.width() * r.height()) if len(faces) > 1 else faces[0]
                landmarks = self.predictor(rgb_frame, face)
                face_descriptor = self.face_rec.compute_face_descriptor(rgb_frame, landmarks)
                feature_vector = np.array(face_descriptor)
                # 特征向量写入txt本地持久化
                feature_file = os.path.join(person_dir, f'feature_{time.strftime("%Y%m%d_%H%M%S")}_{i}.txt')
                np.savetxt(feature_file, feature_vector)
                # 截取人脸区域保存截图
                face_img = img_data[face.top():face.bottom(), face.left():face.right()]
                cv2.imwrite(os.path.join(user_image_dir, f'face_sample_{i + 1}.jpg'), face_img)
                # 内存特征库同步更新
                if name not in self.features_dict:
                    self.features_dict[name] = []
                self.features_dict[name].append(feature_vector)
            # 同步更新归一化特征矩阵
            self.normalized_features = self._build_normalized_features()
            # 更新系统学生数量
            system_status["user_count"] = len(self.features_dict)
            system_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            return True
        except Exception as e:
            logger.error(f"批量保存特征失败: {str(e)}")
            return False

    def delete_person(self, name):
        # 删除单个学生：删除本地文件夹+内存特征
        try:
            person_dir = os.path.join('face_features', name)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
            user_image_dir = os.path.join(app.config['USER_IMAGES_FOLDER'], name)
            if os.path.exists(user_image_dir):
                shutil.rmtree(user_image_dir)
            if name in self.features_dict:
                del self.features_dict[name]
            self.normalized_features = self._build_normalized_features()
            system_status["user_count"] = len(self.features_dict)
            system_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            return True
        except Exception as e:
            logger.error(f"删除用户失败: {name} - {str(e)}")
            return False

    def clear_all_persons(self):
        # 一键清空所有学生数据
        try:
            base_dir = 'face_features'
            if os.path.exists(base_dir):
                for name in os.listdir(base_dir):
                    p = os.path.join(base_dir, name)
                    if os.path.isdir(p):
                        shutil.rmtree(p)
            user_img_dir = app.config['USER_IMAGES_FOLDER']
            if os.path.exists(user_img_dir):
                for name in os.listdir(user_img_dir):
                    p = os.path.join(user_img_dir, name)
                    if os.path.isdir(p):
                        shutil.rmtree(p)
            self.features_dict.clear()
            self.normalized_features.clear()
            system_status["user_count"] = 0
            system_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            logger.info("已清空全部人员数据")
            return True
        except Exception as e:
            logger.error(f"清空全部人员失败: {str(e)}")
            return False

    def draw_all_text(self, image, overlay_texts, label_texts):
        # 使用PIL绘制中文文字（opencv原生不支持中文），增加黑色半透明底色提升可读性
        try:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
             # 绘制顶部状态栏小字：时间、FPS、摄像头状态
            for text, position, color in overlay_texts:
                draw.text(position, text, font=self.small_font, fill=color[::-1])
            # 绘制人脸下方姓名+相似度大字，带黑色背景框
            for text, position, color in label_texts:
                bbox = draw.textbbox((0, 0), text, font=self.font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bg = (position[0], position[1] - th - 5, position[0] + tw + 10, position[1] + 5)
                draw.rectangle(bg, fill=(0, 0, 0, 128))
                draw.text((position[0] + 5, position[1] - th), text, font=self.font, fill=color[::-1])
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return image

    def record_attendance(self, name):
         # 识别到在册学生，自动记录打卡，同一天仅记录一次
        if name == "陌生人":
            return False
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            now_str = datetime.now().strftime("%H:%M:%S")
            if name not in attendance_records:
                attendance_records[name] = {}
            # 当天无打卡记录才新增
            if today not in attendance_records[name]:
                attendance_records[name][today] = now_str
                system_status["attendance_count"] += 1
                logger.info(f"考勤记录: {name} 于 {now_str} 签到")
                return True
        except Exception as e:
            logger.error(f"记录考勤失败: {str(e)}")
        return False


# 初始化人脸识别器
try:
    recognizer = FaceRecognizer()
    logger.info("人脸识别器初始化成功")
except Exception as e:
    logger.critical(f"人脸识别器初始化失败: {str(e)}")
    recognizer = None


def allowed_file(filename):
    # 工具函数：判断上传文件后缀是否为允许的图片格式
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class CameraStream:
    """摄像头双缓冲：独立线程持续读帧，处理慢时不会阻塞画面"""

    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.cap = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._connect()
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.connected = False
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue
            self.connected = True
            with self.lock:
                self.frame = frame  # 只保留最新帧，旧帧自动丢弃

    def _connect(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.connected = True
                system_status["camera_status"] = "已连接"
                logger.info("摄像头连接成功")
            else:
                self.cap.release()
                self.cap = None
                self.connected = False
                system_status["camera_status"] = "无摄像设备"
                time.sleep(1)
        except Exception as e:
            logger.error(f"摄像头连接异常: {e}")
            self.connected = False
            time.sleep(1)

    def read(self):
        """返回最新帧的副本，不会阻塞"""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def is_opened(self):
        return self.connected

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


# 初始化摄像头（双缓冲模式）
camera_stream = CameraStream(camera_id=0, width=640, height=480)


class FaceTracker:
    """基于 IoU 的轻量级人脸跟踪器，用于跳帧识别时平滑框的位置"""

    def __init__(self, max_lost=5):
        self.tracks = {}       # {track_id: {'bbox': ..., 'name': ..., 'lost': 0, ...}}
        self.next_id = 0
        self.max_lost = max_lost

    def iou(self, boxA, boxB):
        """计算两个矩形框的 IoU (交并比)"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        """
        用新的检测结果更新轨迹
        detections: [(face_rect, name, similarity, left_eye, right_eye), ...]
        """
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
            return

        det_boxes = []
        for face, name, sim, le, re in detections:
            det_boxes.append((face.left(), face.top(), face.right(), face.bottom()))

        matched_dets = set()
        matched_tracks = set()

        for tid, track in self.tracks.items():
            best_iou = 0.3
            best_det_idx = -1
            for i, box in enumerate(det_boxes):
                if i in matched_dets:
                    continue
                iou_val = self.iou(track['bbox'], box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i
            if best_det_idx >= 0:
                face, name, sim, le, re = detections[best_det_idx]
                self.tracks[tid]['bbox'] = det_boxes[best_det_idx]
                self.tracks[tid]['name'] = name
                self.tracks[tid]['similarity'] = sim
                self.tracks[tid]['left_eye'] = le
                self.tracks[tid]['right_eye'] = re
                self.tracks[tid]['lost'] = 0
                matched_dets.add(best_det_idx)
                matched_tracks.add(tid)
            else:
                self.tracks[tid]['lost'] += 1

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]

        for i, (face, name, sim, le, re) in enumerate(detections):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det_boxes[i],
                    'name': name,
                    'similarity': sim,
                    'left_eye': le,
                    'right_eye': re,
                    'lost': 0,
                }

    def get_current_results(self):
        """获取当前所有活跃轨迹，用于绘制"""
        results = []
        for tid, track in self.tracks.items():
            if track['lost'] == 0:
                box = track['bbox']
                face = dlib.rectangle(box[0], box[1], box[2], box[3])
                results.append((face, track['name'], track['similarity'],
                              track['left_eye'], track['right_eye']))
        return results


def generate_frames():
    # MJPEG视频流生成器：持续读取摄像头帧，绘制人脸/人眼框，编码后推送给前端网页
    global is_recognizing, system_status

    last_frame_time = time.time()
    frame_counter = 0
    fps_counter = 0
    fps_last_time = time.time()
    # 动态跳帧：根据识别耗时自动调整间隔
    recognize_interval = 5
    cached_results = []# 缓存上一次识别结果，中间帧复用不用重新计算
    face_tracker = FaceTracker(max_lost=8)# 帧间跟踪器

    while True:
        try:
            # 双缓冲读取最新帧，不会阻塞
            success, frame = camera_stream.read()

            # 摄像头读取失败，输出空白提示画面
            if not success:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No Camera", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(blank_frame, "Waiting...", (50, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            frame_counter += 1
            # 真实FPS统计
            fps_counter += 1
            now = time.time()
            if now - fps_last_time >= 1.0:
                system_status["frame_rate"] = fps_counter
                fps_counter = 0
                fps_last_time = now

            processed_frame = frame.copy()# 复制原图用于绘制标注，不破坏原始帧
            label_texts = []

            # 如果开启识别开关，执行人脸检测逻辑
            if is_recognizing and recognizer:
                # 动态跳帧：根据上一次识别耗时自动调整间隔
                if frame_counter % recognize_interval == 1 or not cached_results:
                    t0 = time.time()
                    cached_results = recognizer.process_frame(frame.copy())
                    elapsed = time.time() - t0
                    face_tracker.update(cached_results)
                    # 自适应调整识别间隔：快则多识别，慢则少识别
                    if elapsed < 0.15:
                        recognize_interval = 3
                    elif elapsed > 0.5:
                        recognize_interval = 8
                    else:
                        recognize_interval = 5
                else:
                    # 跟踪帧：用上一帧的轨迹继续显示，框不会跳动
                    cached_results = face_tracker.get_current_results()

                # 循环绘制每个人脸框、人眼框、姓名文字
                for face, name, similarity, left_eye, right_eye in cached_results:
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()

                    # 人脸矩形框（绿色=已知，红色=陌生人）
                    color = (0, 255, 0) if name != "陌生人" else (0, 0, 255)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                    # 人眼矩形框（蓝色，区分于人脸框）
                    le_x1, le_y1, le_x2, le_y2 = left_eye
                    re_x1, re_y1, re_x2, re_y2 = right_eye
                    cv2.rectangle(processed_frame, (le_x1, le_y1), (le_x2, le_y2), (255, 0, 0), 1)
                    cv2.rectangle(processed_frame, (re_x1, re_y1), (re_x2, re_y2), (255, 0, 0), 1)

                    # 组装画面底部姓名文字
                    if name == "陌生人":
                        label_texts.append((f"陌生人 ({similarity:.2%})", (x1, y1), (0, 0, 255)))
                    else:
                        label_texts.append((f"{name} ({similarity:.2%})", (x1, y1), (0, 255, 0)))
                        recognizer.record_attendance(name)

            if is_recognizing and not recognizer:
                cv2.putText(processed_frame, "System Error: Model not loaded", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 绘制顶部状态栏文字（时间、模式、FPS、摄像头状态）
            _cn = recognizer.chinese_available if recognizer else False
            if label_texts and recognizer:
                overlay_texts = [
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 8), (0, 255, 255)),
                    ("考勤模式" if _cn else "Attendance Mode", (10, 36), (255, 255, 0)),
                    (f"FPS: {system_status['frame_rate']}", (10, 64), (255, 255, 0)),
                    (f"摄像头: {system_status['camera_status']}" if _cn else f"Camera: {system_status['camera_status']}", (10, 92), (255, 255, 0)),
                    (f"当前人脸：{len(cached_results)}人" if _cn else f"Faces: {len(cached_results)}", (10, 120), (0, 255, 0)),
                ]#显示人脸
                processed_frame = recognizer.draw_all_text(processed_frame, overlay_texts, label_texts)
            else:
                # 未开启识别仅显示基础状态文字
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if _cn and recognizer:
                    status_text = "待机模式" if not is_recognizing else "识别中"
                    overlay_texts = [
                        (time_str, (10, 8), (0, 255, 255)),
                        (status_text, (10, 36), (255, 255, 0)),
                        (f"FPS: {system_status['frame_rate']}", (10, 64), (255, 255, 0)),
                        (f"当前人脸：{len(cached_results)}人", (10, 92), (0, 255, 0)),
                    ]
                    processed_frame = recognizer.draw_all_text(processed_frame, overlay_texts, [])
                else:
                    status_text = "Monitoring" if not is_recognizing else "Recognizing"
                    cv2.putText(processed_frame, time_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cv2.putText(processed_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(processed_frame, f"FPS: {system_status['frame_rate']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(processed_frame, f"Faces: {len(cached_results)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # 图像压缩为jpg，降低传输体积
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 1])

            # 限制帧间隔，最高30帧，防止CPU满载
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)
            last_frame_time = current_time

             # 持续推送视频流分片给前端img标签
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logger.error(f"生成视频帧失败: {str(e)}")
            time.sleep(0.1)


# ============================================================
# Flask 路由
# ============================================================
@app.route('/')
def index():
    try:
        person_list = []
        base_dir = 'face_features'
        if os.path.exists(base_dir):
            person_list = [name for name in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, name))]

        attendance_list = []
        for name, records in attendance_records.items():
            for date, time_str in records.items():
                attendance_list.append({'name': name, 'date': date, 'time': time_str})
        attendance_list.sort(key=lambda x: x['date'] + x['time'], reverse=True)
        recent_attendance = attendance_list[:10]

        start_time = datetime.strptime(system_status["system_uptime"], "%Y-%m-%d %H:%M")
        uptime = datetime.now() - start_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        system_status["uptime"] = f"{int(hours)}小时{int(minutes)}分钟"

        return render_template('index.html',
                               person_list=person_list,
                               attendance_records=recent_attendance,
                               system_status=system_status,
                               is_recognizing=is_recognizing)
    except Exception as e:
        logger.error(f"渲染主页面失败: {str(e)}")
        return "系统错误，请查看日志", 500


@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"视频流路由错误: {str(e)}")
        return "视频流错误", 500


@app.route('/toggle_recognize', methods=['POST'])
def toggle_recognize():
    global is_recognizing
    try:
        is_recognizing = not is_recognizing
        system_status["recognition_status"] = "运行中" if is_recognizing else "待机"
        logger.info(f"识别状态切换为: {system_status['recognition_status']}")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"切换识别状态失败: {str(e)}")
        return redirect(url_for('index'))


@app.route('/add_person', methods=['POST'])
def add_person():
    try:
        name = request.form.get('name')
        if not name:
            return redirect(url_for('index'))
        if 'file' not in request.files:
            return redirect(url_for('index'))
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return redirect(url_for('index'))
        images = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                if image is not None:
                    images.append(image)
         # 至少上传3张照片才能录入
        if len(images) < 3:
            return redirect(url_for('index'))
        if recognizer and recognizer.save_multiple_face_features(images, name):
            logger.info(f"添加人员成功: {name}")
        else:
            logger.error(f"添加人员失败: {name}")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"添加人员异常: {str(e)}")
        return redirect(url_for('index'))


@app.route('/delete_person', methods=['POST'])
def delete_person():
    # 删除单个学生接口
    try:
        name = request.form.get('name')
        if name:
            if recognizer and recognizer.delete_person(name):
                logger.info(f"删除人员成功: {name}")
            else:
                logger.error(f"删除人员失败: {name}")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"删除人员异常: {str(e)}")
        return redirect(url_for('index'))


@app.route('/clear_all_persons', methods=['POST'])
def clear_all_persons():
    # 清空全部学生接口，返回json提示
    try:
        if recognizer and recognizer.clear_all_persons():
            logger.info("清空全部人员成功")
            return jsonify({"status": "success", "message": "已清空全部人员数据"})
        else:
            logger.error("清空全部人员失败")
            return jsonify({"status": "error", "message": "清空失败"}), 500
    except Exception as e:
        logger.error(f"清空全部人员异常: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user_images/<name>/<filename>')
def get_user_image(name, filename):
     # 前端展示学生人脸样本图片的静态资源接口
    try:
        user_dir = os.path.join(app.config['USER_IMAGES_FOLDER'], name)
        return send_from_directory(user_dir, filename)
    except:
        return "图片未找到", 404


@app.route('/export_attendance', methods=['POST'])
def export_attendance():
    # 导出考勤记录CSV表格，浏览器自动下载文件
    try:
        if not attendance_records:
            return "没有考勤记录可导出", 400
        csv_content = "姓名,日期,时间\n"
        for name, records in attendance_records.items():
            for date, time_str in records.items():
                csv_content += f"{name},{date},{time_str}\n"
        response = Response(
            csv_content, mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=attendance_records.csv"}
        )
        logger.info("考勤记录导出成功")
        return response
    except Exception as e:
        logger.error(f"导出考勤记录失败: {str(e)}")
        return "导出失败", 500


@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    # 清空所有打卡记录
    try:
        attendance_records.clear()
        system_status["attendance_count"] = 0
        logger.info("考勤记录已清除")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"清除考勤记录失败: {str(e)}")
        return redirect(url_for('index'))


@app.route('/restart_camera', methods=['POST'])
def restart_camera():
     # 重启摄像头接口：释放资源，后台线程自动重连
    global camera_stream
    try:
        camera_stream.release()
        camera_stream = CameraStream(camera_id=0, width=640, height=480)
        logger.info("摄像头重启请求")# 调用摄像头，设备编号 0 = 本机默认摄像头
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"重启摄像头失败: {str(e)}")
        return redirect(url_for('index'))


@app.route('/batch_import', methods=['POST'])
def batch_import():
    """批量导入全班照片
    接收按姓名文件夹组织的照片，自动提取特征入库
    目录结构要求：每个子文件夹名=人员姓名，子文件夹内放该人的照片（≥3张）
    """
    try:
        if 'files' not in request.files:
            return jsonify({"status": "error", "message": "未选择文件"}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({"status": "error", "message": "未选择文件"}), 400

        # 按文件夹名分组：folder_name -> [file1, file2, ...]
        folder_groups = {}
        for file in files:
            # webkitRelativePath 格式: "文件夹名/文件名.jpg"
            relative_path = file.filename
            if '/' in relative_path:
                folder_name = relative_path.split('/')[0]
                actual_filename = relative_path.split('/')[-1]
            else:
                continue  # 跳过根目录文件

            if folder_name not in folder_groups:
                folder_groups[folder_name] = []
            folder_groups[folder_name].append((actual_filename, file))

        if not folder_groups:
            return jsonify({"status": "error", "message": "未检测到有效的文件夹结构"}), 400

        success_count = 0
        fail_count = 0
        details = []

        for person_name, file_list in folder_groups.items():
            images = []
            for actual_filename, file in file_list:
                if not allowed_file(actual_filename):
                    continue
                # 保存到临时目录
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{actual_filename}")
                file.save(temp_path)
                image = cv2.imread(temp_path)
                if image is not None:
                    images.append(image)

            if len(images) < 3:
                details.append(f"{person_name}: 照片不足3张({len(images)}张)，跳过")
                fail_count += 1
                continue

            if recognizer and recognizer.save_multiple_face_features(images, person_name):
                success_count += 1
                details.append(f"{person_name}: 成功录入{len(images)}张")
            else:
                fail_count += 1
                details.append(f"{person_name}: 特征提取失败")

        return jsonify({
            "status": "success",
            "message": f"批量导入完成：成功{success_count}人，失败{fail_count}人",
            "details": details
        })
    except Exception as e:
        logger.error(f"批量导入异常: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/batch_import_excel', methods=['POST'])
def batch_import_excel():
    """【新增】基于Excel学生名单 + xuehao_pic图片目录的精准批量导入
    核心逻辑：Excel学号→姓名映射 + 图片文件夹名截取学号 → 精准匹配，杜绝张冠李戴
    """
    try:
        # ============================================================
        # 第一步：读取Excel学生名单，生成 {学号(字符串): 姓名} 映射字典
        # ============================================================
        excel_path = '243人工智能1班学生名单_136877420.xlsx'
        if not os.path.exists(excel_path):
            return jsonify({"status": "error", "message": f"Excel文件不存在: {excel_path}"}), 400

        # 读取Excel，A列=学号，B列=姓名，强制学号为字符串防止int/str错位
        df = pd.read_excel(excel_path, dtype={0: str})
        # 兼容列名：取前两列，重命名为学号和姓名
        df = df.iloc[:, :2]
        df.columns = ['学号', '姓名']
        df['学号'] = df['学号'].astype(str).str.strip()
        df['姓名'] = df['姓名'].astype(str).str.strip()

        # 生成映射字典：{学号: 姓名}
        student_map = dict(zip(df['学号'], df['姓名']))
        excel_ids = set(student_map.keys())
        logger.info(f"Excel读取完成，共 {len(student_map)} 名学生")

        # ============================================================
        # 第二步：遍历xuehao_pic/xuehao_pic目录，匹配学号提取特征
        # ============================================================
        pic_base = os.path.join('xuehao_pic', 'xuehao_pic')
        if not os.path.exists(pic_base):
            return jsonify({"status": "error", "message": f"图片目录不存在: {pic_base}"}), 400

        success_count = 0       # 成功录入人数
        skip_exist_count = 0    # 已存在跳过人数
        skip_no_match_count = 0 # 学号无匹配跳过人数
        fail_count = 0          # 特征提取失败人数
        details = []            # 详细日志
        matched_ids = set()     # 成功匹配到的学号集合
        no_photo_ids = set()    # Excel有学号但图片文件夹缺失
        invalid_folders = []    # 有文件夹但Excel无对应学号

        # 遍历所有子文件夹
        for folder_name in os.listdir(pic_base):
            folder_path = os.path.join(pic_base, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 截取文件夹名中"_pic"前的数字作为学号
            # 文件夹命名规则：2024325103_pic
            if '_pic' not in folder_name:
                invalid_folders.append(folder_name)
                continue

            xuehao = folder_name.split('_pic')[0].strip()

            # 到student_map字典精准匹配姓名
            if xuehao not in student_map:
                # Excel无该学号，打印警告，跳过，绝不随意绑定其他学生
                skip_no_match_count += 1
                invalid_folders.append(folder_name)
                logger.warning(f"文件夹 {folder_name} 学号 {xuehao} 在Excel中无对应记录，跳过")
                continue

            person_name = student_map[xuehao]
            matched_ids.add(xuehao)

            # 检查是否已存在该学生的特征文件（姓名_学号.npy），避免重复录入
            person_dir = os.path.join('face_features', person_name)
            npy_file = os.path.join(person_dir, f'{person_name}_{xuehao}.npy')
            if os.path.exists(npy_file):
                skip_exist_count += 1
                details.append(f"{person_name}({xuehao}): 特征已存在，跳过")
                logger.info(f"{person_name}({xuehao}): 特征已存在，跳过")
                continue

            # 读取该文件夹下全部.jpg图片
            jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
            if not jpg_files:
                fail_count += 1
                details.append(f"{person_name}({xuehao}): 文件夹内无jpg图片")
                logger.warning(f"{person_name}({xuehao}): 文件夹内无jpg图片")
                continue

            # 对每张图片提取人脸特征
            os.makedirs(person_dir, exist_ok=True)
            all_features = []  # 该学生所有图片的特征向量
            img_count = 0
            skip_count = 0

            for jpg_file in jpg_files:
                img_path = os.path.join(folder_path, jpg_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        skip_count += 1
                        continue
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = recognizer.detector(rgb_img)
                    if not faces:
                        skip_count += 1
                        logger.warning(f"  {jpg_file}: 未检测到人脸，跳过")
                        continue
                    # 多个人脸取最大的
                    face = max(faces, key=lambda r: r.width() * r.height()) if len(faces) > 1 else faces[0]
                    landmarks = recognizer.predictor(rgb_img, face)
                    face_descriptor = recognizer.face_rec.compute_face_descriptor(rgb_img, landmarks)
                    feature_vector = np.array(face_descriptor)
                    all_features.append(feature_vector)
                    img_count += 1
                except Exception as e:
                    skip_count += 1
                    logger.warning(f"  {jpg_file}: 处理失败({e})，跳过")
                    continue

            if not all_features:
                fail_count += 1
                details.append(f"{person_name}({xuehao}): 所有图片均未检测到人脸")
                logger.warning(f"{person_name}({xuehao}): 所有图片均未检测到人脸")
                continue

            # 将该学生所有特征合并保存为 姓名_学号.npy
            features_array = np.array(all_features)
            np.save(npy_file, features_array)

            # 同步更新内存特征库：每张图片的特征单独存入列表
            if person_name not in recognizer.features_dict:
                recognizer.features_dict[person_name] = []
            for fv in all_features:
                recognizer.features_dict[person_name].append(fv)

            success_count += 1
            details.append(f"{person_name}({xuehao}): 成功提取{img_count}张，跳过{skip_count}张")
            logger.info(f"{person_name}({xuehao}): 成功提取{img_count}张，跳过{skip_count}张")

        # ============================================================
        # 第三步：自动筛查缺失学生，输出核对清单
        # ============================================================
        # Excel有学号但图片文件夹缺失 = 缺照片学生
        no_photo_ids = excel_ids - matched_ids
        # 有文件夹但Excel无对应学号 = 无效文件夹（已在上面处理）

        # 更新系统状态
        recognizer.normalized_features = recognizer._build_normalized_features()
        system_status["user_count"] = len(recognizer.features_dict)
        system_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 打印核对清单到控制台
        print("\n" + "=" * 55)
        print("  批量导入完成核对清单")
        print("=" * 55)
        print(f"  成功录入: {success_count} 人")
        print(f"  已存在跳过: {skip_exist_count} 人")
        print(f"  学号无匹配跳过: {skip_no_match_count} 个文件夹")
        if no_photo_ids:
            print(f"\n  缺照片学生（Excel有但图片文件夹缺失）共 {len(no_photo_ids)} 人:")
            for sid in sorted(no_photo_ids):
                print(f"    ✗ {sid} - {student_map[sid]}")
        if invalid_folders:
            print(f"\n  无效文件夹（图片存在但Excel无对应学号）共 {len(invalid_folders)} 个:")
            for f in sorted(invalid_folders):
                print(f"    ✗ {f}")
        print("=" * 55 + "\n")

        return jsonify({
            "status": "success",
            "message": f"批量导入完成：成功{success_count}人，已存在跳过{skip_exist_count}人，无匹配跳过{skip_no_match_count}个文件夹",
            "details": details,
            "no_photo_students": [f"{sid}-{student_map[sid]}" for sid in sorted(no_photo_ids)],
            "invalid_folders": sorted(invalid_folders)
        })

    except Exception as e:
        logger.error(f"Excel批量导入异常: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    try:
        logger.info("=" * 40)
        logger.info("  人脸识别考勤系统启动")
        logger.info(f"  访问地址: http://127.0.0.1:5000")
        logger.info("=" * 40)

        # 直接运行 Flask
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

    except OSError as e:
        if 'address already in use' in str(e).lower() or 10048 in str(getattr(e, 'winerror', [])):
            logger.critical("端口 5000 已被占用，请关闭占用进程或更换端口")
            logger.critical("排查命令: netstat -ano | findstr :5000")
        else:
            logger.critical(f"应用程序崩溃: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Ctrl+C 中断，正在关闭系统...")
    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}")
    finally:
        # 释放摄像头硬件资源
        try:
            camera_stream.release()
            logger.info("摄像头资源已释放")
        except Exception:
            pass
        logger.info("系统已退出")
