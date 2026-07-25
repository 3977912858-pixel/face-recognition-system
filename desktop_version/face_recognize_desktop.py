# -*- coding: utf-8 -*-
"""
基于 Python+OpenCV+dlib 的多人实时人脸识别系统（桌面版）
功能：调用摄像头，检测人脸和人眼，识别同学身份并显示姓名，
      绘制彩色矩形框标注，按 q 键退出。
"""

import os
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1. 加载模型文件
# ============================================================
# dlib 人脸检测器
detector = dlib.get_frontal_face_detector()
# 68 个人脸关键点模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 128 维人脸特征提取模型
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# ============================================================
# 2. 加载特征库（所有已录入同学的人脸特征）
# ============================================================
def load_face_features():
    """读取 face_features 文件夹下所有同学的特征向量"""
    features_dict = {}
    base_dir = 'face_features'
    if not os.path.exists(base_dir):
        print(f"[警告] 特征库目录 {base_dir} 不存在")
        return features_dict

    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        features_dict[person_name] = []
        for feature_file in os.listdir(person_dir):
            feature_path = os.path.join(person_dir, feature_file)
            if not os.path.isfile(feature_path):
                continue
            try:
                if feature_file.endswith('.txt') and feature_file.startswith('feature_'):
                    feature_vector = np.loadtxt(feature_path)
                    features_dict[person_name].append(feature_vector)
                elif feature_file.endswith('.npy'):
                    feature_data = np.load(feature_path)
                    if feature_data.ndim == 1:
                        features_dict[person_name].append(feature_data)
                    elif feature_data.ndim == 2 and feature_data.shape[1] == 128:
                        for i in range(feature_data.shape[0]):
                            features_dict[person_name].append(feature_data[i])
            except Exception as e:
                print(f"[错误] 加载特征失败: {feature_path} - {e}")

    print(f"特征库加载完成，共 {len(features_dict)} 人")
    return features_dict


def build_normalized_features(features_dict):
    """预计算 L2 归一化特征矩阵，用于快速匹配"""
    normalized = {}
    for person_name, feat_list in features_dict.items():
        if not feat_list:
            continue
        matrix = np.array(feat_list, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized[person_name] = matrix / norms
    return normalized


def find_best_match(target_feature, normalized_features, threshold=0.65):
    """在特征库中找到最匹配的同学"""
    if not normalized_features:
        return "陌生人", 0.0

    norm = np.linalg.norm(target_feature)
    if norm == 0:
        return "陌生人", 0.0
    target_normalized = target_feature / norm

    best_match = None
    best_similarity = -1
    for person_name, feat_matrix in normalized_features.items():
        similarities = feat_matrix @ target_normalized
        max_sim = float(np.max(similarities))
        if max_sim > best_similarity:
            best_similarity = max_sim
            best_match = person_name

    if best_similarity < threshold:
        return "陌生人", best_similarity
    return best_match, best_similarity


# ============================================================
# 3. 加载中文字体（用于显示姓名）
# ============================================================
def load_font():
    """加载系统中文字体"""
    font_paths = [
        'simhei.ttf', 'simsun.ttc', 'msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 28)
            return font
        except:
            continue
    return ImageFont.load_default()


def draw_text_with_chinese(frame, texts):
    """用 PIL 绘制中文文字（OpenCV 原生不支持中文）"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = load_font()
    for text, position, color in texts:
        draw.text(position, text, font=font, fill=color[::-1])  # BGR -> RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# 4. 主程序
# ============================================================
def main():
    # 加载特征库
    features_dict = load_face_features()
    normalized_features = build_normalized_features(features_dict)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    print("=" * 40)
    print("  人脸识别系统已启动")
    print(f"  已录入 {len(features_dict)} 人")
    print("  按 q 键退出")
    print("=" * 40)

    process_width = 320  # 缩放宽度，降低计算量

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 图像预处理：缩放 + 转 RGB ----
        height, width = frame.shape[:2]
        scale = process_width / width
        if scale < 1:
            small_frame = cv2.resize(frame, (process_width, int(height * scale)))
        else:
            small_frame = frame
            scale = 1.0
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # ---- 人脸检测 ----
        faces = detector(rgb_frame)

        label_texts = []
        inv_scale = 1.0 / scale if scale < 1 else 1.0

        for face in faces:
            try:
                # ---- 68 关键点定位 ----
                landmarks = predictor(rgb_frame, face)

                # ---- 提取 128 维特征 ----
                face_descriptor = face_rec.compute_face_descriptor(rgb_frame, landmarks)
                target_feature = np.array(face_descriptor)

                # ---- 特征匹配，识别身份 ----
                name, similarity = find_best_match(target_feature, normalized_features)

                # ---- 计算眼睛坐标 ----
                left_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(36, 42)]
                right_eye_pts = [(landmarks.part(j).x, landmarks.part(j).y) for j in range(42, 48)]

                le_x = [p[0] for p in left_eye_pts]
                le_y = [p[1] for p in left_eye_pts]
                left_eye_rect = (min(le_x), min(le_y), max(le_x), max(le_y))

                re_x = [p[0] for p in right_eye_pts]
                re_y = [p[1] for p in right_eye_pts]
                right_eye_rect = (min(re_x), min(re_y), max(re_x), max(re_y))

                # ---- 还原坐标到原图尺寸 ----
                if scale < 1:
                    face = dlib.rectangle(
                        round(face.left() * inv_scale), round(face.top() * inv_scale),
                        round(face.right() * inv_scale), round(face.bottom() * inv_scale)
                    )
                    left_eye_rect = tuple(round(v * inv_scale) for v in left_eye_rect)
                    right_eye_rect = tuple(round(v * inv_scale) for v in right_eye_rect)

                # ---- 绘制矩形框 ----
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()

                # 人脸框：绿色 = 已识别，红色 = 陌生人
                color = (0, 255, 0) if name != "陌生人" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 人眼框：蓝色
                le_x1, le_y1, le_x2, le_y2 = left_eye_rect
                re_x1, re_y1, re_x2, re_y2 = right_eye_rect
                cv2.rectangle(frame, (le_x1, le_y1), (le_x2, le_y2), (255, 0, 0), 1)
                cv2.rectangle(frame, (re_x1, re_y1), (re_x2, re_y2), (255, 0, 0), 1)

                # ---- 准备姓名文字 ----
                if name == "陌生人":
                    label_texts.append((f"陌生人", (x1, y1 - 30), (0, 0, 255)))
                else:
                    label_texts.append((f"{name}", (x1, y1 - 30), (0, 255, 0)))

            except Exception as e:
                continue

        # ---- 绘制人脸数量 ----
        cv2.putText(
            frame, f"Faces: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )

        # ---- 绘制中文姓名 ----
        if label_texts:
            frame = draw_text_with_chinese(frame, label_texts)

        # ---- 显示画面 ----
        cv2.imshow('Face Recognition - Press q to quit', frame)

        # ---- 按 q 退出 ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到 q 键，正在退出...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("系统已退出")


if __name__ == '__main__':
    main()
