import json
import os
import cv2
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)

def load_annotations(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and 'samples' in data:
            data = data['samples']
        return data
    except Exception as e:
        logging.error(f"加载标注失败: {e}")
        return []

def load_image_cv2(path):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("cv2.imread 返回 None")
        return img
    except Exception as e:
        logging.error(f"cv2.imread 在加载图片时发生错误: {e}")
        return None

def load_image_pillow(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = np.array(img)
        return img
    except Exception as e:
        logging.error(f"Pillow 在加载图片时发生错误: {e}")
        return None

def preprocess_data(image_paths):
    data = []
    for path in image_paths:
        logging.info(f"加载图片：{path}")
        img = load_image_cv2(path)
        if img is None:
            logging.info("尝试使用 Pillow 加载图片")
            img = load_image_pillow(path)
        if img is None:
            logging.warning(f"最终无法加载图片: {path}")
            continue
        data.append(img)
    return data

if __name__ == "__main__":
    # 举例：构造图片路径列表
    images_dir = r"D:\ProgramData\PyCharm Community Edition 2024.3.5\PycharmProjects\PythonProject2\OpenDataLab___CrowdHuman\dsdl\dataset_root\Images"
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    loaded_images = preprocess_data(image_files)
    logging.info(f"成功加载 {len(loaded_images)} 张图片")