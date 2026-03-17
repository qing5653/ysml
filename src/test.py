import os
import sys

# ========== 第 1 步：设置 Qt 插件路径（必须在导入任何其他库之前） ==========
try:
    import PyQt5
    qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
    os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
    print(f"✅ Qt 插件路径已设置为: {qt_plugin_path}")
except ImportError:
    print("❌ PyQt5 未安装，请先安装 PyQt5")
    sys.exit(1)

# ========== 第 2 步：立即创建 QApplication（锁定 Qt 环境） ==========
from PyQt5.QtWidgets import QApplication
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# ========== 第 3 步：再导入其他可能干扰的库 ==========
import torch
import cv2
import numpy as np
import skimage
import pandas as pd
import matplotlib
import seaborn as sns
import tqdm
import comet_ml
from PIL import Image

print("所有基础库导入成功，开始功能测试...\n")

# ========== 第 4 步：功能测试函数 ==========
def check_cuda():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
            return True, f"CUDA 可用 (设备数: {device_count}, 当前设备: {device_name})"
        else:
            return False, "CUDA 不可用，将使用 CPU"
    except Exception as e:
        return False, f"CUDA 检查失败: {e}"

def check_yolo_inference():
    try:
        from ultralytics import YOLO
        model_name = 'yolo11n.pt'
        print(f"正在下载/加载 {model_name} ...")
        model = YOLO(model_name)
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        if results and len(results) > 0:
            return True, "YOLOv11 推理成功"
        else:
            return False, "YOLOv11 推理返回空结果"
    except Exception as e:
        return False, f"YOLOv11 测试失败: {e}"

def check_qt():
    """测试 PyQt5 能否创建简单窗口（1秒后自动关闭）"""
    try:
        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtCore import QTimer
        label = QLabel("Qt 测试窗口 - 1秒后自动关闭")
        label.show()
        timer = QTimer()
        timer.timeout.connect(app.quit)
        timer.start(1000)
        app.exec_()
        return True, "Qt 窗口创建并关闭成功"
    except Exception as e:
        return False, f"Qt 测试失败: {e}"

def check_opencv():
    try:
        img = np.zeros((100,100,3), dtype=np.uint8)
        ret, buf = cv2.imencode('.jpg', img)
        if ret:
            return True, "OpenCV 编码成功"
        else:
            return False, "OpenCV 编码失败"
    except Exception as e:
        return False, f"OpenCV 测试失败: {e}"

def check_slic():
    try:
        from skimage import data, segmentation
        img = data.coffee()
        segments = segmentation.slic(img, n_segments=100, compactness=10, start_label=1)
        if segments.shape == img.shape[:2]:
            return True, "SLIC 超像素分割成功"
        else:
            return False, "SLIC 输出形状不符"
    except Exception as e:
        return False, f"SLIC 测试失败: {e}"

def check_glcm():
    try:
        from skimage import data, feature
        img = data.camera()
        glcm = feature.graycomatrix(img, distances=[1], angles=[0], levels=256)
        if glcm.shape == (256, 256, 1, 1):
            return True, "GLCM 计算成功"
        else:
            return False, "GLCM 输出形状不符"
    except Exception as e:
        return False, f"GLCM 测试失败: {e}"

# ========== 第 5 步：执行测试 ==========
results = []

# CUDA
ok, msg = check_cuda()
results.append(('CUDA', ok, msg))
print(f"[{'✅' if ok else '⚠️'}] {msg}")

# OpenCV
ok, msg = check_opencv()
results.append(('OpenCV', ok, msg))
print(f"[{'✅' if ok else '❌'}] {msg}")

# YOLOv11 推理
ok, msg = check_yolo_inference()
results.append(('YOLOv11推理', ok, msg))
print(f"[{'✅' if ok else '❌'}] {msg}")

# SLIC
ok, msg = check_slic()
results.append(('SLIC', ok, msg))
print(f"[{'✅' if ok else '❌'}] {msg}")

# GLCM
ok, msg = check_glcm()
results.append(('GLCM', ok, msg))
print(f"[{'✅' if ok else '❌'}] {msg}")

# Qt GUI
print("\n注意：接下来会短暂显示一个 Qt 窗口（1秒后自动关闭）...")
ok, msg = check_qt()
results.append(('PyQt5 GUI', ok, msg))
print(f"[{'✅' if ok else '❌'}] {msg}")

print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
failed = [name for name, ok, _ in results if not ok]
if failed:
    print(f"❌ 以下组件测试失败: {', '.join(failed)}")
    sys.exit(1)
else:
    print("✅ 所有组件测试通过，环境就绪！")
    sys.exit(0)