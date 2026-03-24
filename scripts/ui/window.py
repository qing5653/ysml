from __future__ import annotations

from pathlib import Path
import time
import sys

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11_project.spot_guided import SpotGuidedConfig, apply_spot_guided_attention


def _to_qpixmap(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _resolve_default_input_dir(root: Path) -> Path:
    train_cfg = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))
    data_key = "prepared_dataset_yaml" if train_cfg.get("use_prepared_dataset", False) else "data"
    data_cfg = yaml.safe_load((root / train_cfg[data_key]).read_text(encoding="utf-8"))
    return root / data_cfg["path"] / data_cfg["val"]


class RealtimeWorker(QThread):
    frame_signal = pyqtSignal(object, float)
    error_signal = pyqtSignal(str)

    def __init__(self, model_path: Path, conf: float, use_guided: bool, sg_cfg: SpotGuidedConfig):
        super().__init__()
        self.model_path = model_path
        self.conf = conf
        self.use_guided = use_guided
        self.sg_cfg = sg_cfg
        self.running = True

    def run(self) -> None:
        if not self.model_path.exists():
            self.error_signal.emit(f"模型不存在: {self.model_path}")
            return

        model = YOLO(str(self.model_path))
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("无法打开摄像头。")
            return

        last_time = time.perf_counter()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue

            if self.use_guided:
                frame, _ = apply_spot_guided_attention(frame, self.sg_cfg)

            result = model.predict(frame, conf=self.conf, verbose=False)[0]
            vis = result.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            self.frame_signal.emit(vis, fps)

        cap.release()

    def stop(self) -> None:
        self.running = False


class BatchWorker(QThread):
    progress_signal = pyqtSignal(int)
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, model_path: Path, input_dir: Path, output_dir: Path, conf: float, use_guided: bool, sg_cfg: SpotGuidedConfig):
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.conf = conf
        self.use_guided = use_guided
        self.sg_cfg = sg_cfg

    def run(self) -> None:
        if not self.model_path.exists():
            self.error_signal.emit(f"模型不存在: {self.model_path}")
            return
        if not self.input_dir.exists():
            self.error_signal.emit(f"输入目录不存在: {self.input_dir}")
            return

        images = [p for p in sorted(self.input_dir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not images:
            self.error_signal.emit("输入目录没有可处理图像。")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self.model_path))

        start = time.perf_counter()
        for idx, image_path in enumerate(images, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            if self.use_guided:
                image, _ = apply_spot_guided_attention(image, self.sg_cfg)

            result = model.predict(image, conf=self.conf, verbose=False)[0]
            cv2.imwrite(str(self.output_dir / image_path.name), result.plot())
            self.progress_signal.emit(int(idx * 100 / len(images)))

        elapsed = max(time.perf_counter() - start, 1e-6)
        fps = len(images) / elapsed
        self.done_signal.emit(f"批处理完成: {self.output_dir} | 样本数={len(images)} | 平均FPS={fps:.2f}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("轴承缺陷检测系统 (YOLO11)")
        self.resize(1280, 760)

        self.rt_worker: RealtimeWorker | None = None
        self.batch_worker: BatchWorker | None = None

        root = Path(__file__).resolve().parents[2]
        default_model = root / "experiments" / "yolo11_spot_bifpn" / "weights" / "best.pt"
        if not default_model.exists():
            default_model = root / "src" / "yolo11n.pt"

        cfg = yaml.safe_load((root / "configs" / "prepare.yaml").read_text(encoding="utf-8"))
        sg = cfg.get("spot_guided", {})
        self.sg_cfg = SpotGuidedConfig(
            slic_segments=int(sg.get("slic_segments", 200)),
            slic_compactness=float(sg.get("slic_compactness", 12.0)),
            glcm_distances=tuple(int(v) for v in sg.get("glcm_distances", [1, 2])),
            glcm_angles=tuple(float(v) for v in sg.get("glcm_angles", [0.0, 0.785398, 1.570796])),
            entropy_threshold_quantile=float(sg.get("entropy_threshold_quantile", 0.75)),
            blend_alpha=float(sg.get("blend_alpha", 0.45)),
        )

        default_input_dir = _resolve_default_input_dir(root)
        self.model_edit = QLineEdit(str(default_model))
        self.conf_edit = QLineEdit("0.25")
        self.input_dir_edit = QLineEdit(str(default_input_dir))
        self.use_guided_checkbox = QCheckBox("启用 Spot-Guided 候选区域增强")
        self.use_guided_checkbox.setChecked(True)

        browse_model_btn = QPushButton("选择模型")
        browse_model_btn.clicked.connect(self._pick_model)
        browse_input_btn = QPushButton("选择批处理目录")
        browse_input_btn.clicked.connect(self._pick_input_dir)

        self.realtime_btn = QPushButton("启动实时检测")
        self.realtime_btn.clicked.connect(self._toggle_realtime)
        self.batch_btn = QPushButton("启动批量处理")
        self.batch_btn.clicked.connect(self._run_batch)

        self.fps_label = QLabel("实时FPS: -")
        self.target_label = QLabel("目标: >= 25 FPS")
        self.progress = QProgressBar()
        self.progress.setValue(0)

        form = QFormLayout()
        form.addRow("模型路径", self.model_edit)
        form.addRow("置信度", self.conf_edit)
        form.addRow("批处理输入目录", self.input_dir_edit)
        form.addRow("增强", self.use_guided_checkbox)

        top_buttons = QHBoxLayout()
        top_buttons.addWidget(browse_model_btn)
        top_buttons.addWidget(browse_input_btn)
        top_buttons.addWidget(self.realtime_btn)
        top_buttons.addWidget(self.batch_btn)

        self.video_label = QLabel("视频窗口")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(900, 560)
        self.video_label.setStyleSheet("background: #111; color: #ddd;")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        right_col = QVBoxLayout()
        right_col.addLayout(form)
        right_col.addLayout(top_buttons)
        right_col.addWidget(self.fps_label)
        right_col.addWidget(self.target_label)
        right_col.addWidget(self.progress)
        right_col.addWidget(self.log_text)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addLayout(right_col, stretch=2)

        wrapper = QWidget()
        wrapper.setLayout(main_layout)
        self.setCentralWidget(wrapper)

    def _append_log(self, text: str) -> None:
        self.log_text.append(text)

    def _pick_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PyTorch Model (*.pt)")
        if file_path:
            self.model_edit.setText(file_path)

    def _pick_input_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_dir_edit.setText(dir_path)

    def _toggle_realtime(self) -> None:
        if self.rt_worker and self.rt_worker.isRunning():
            self.rt_worker.stop()
            self.rt_worker.wait()
            self.realtime_btn.setText("启动实时检测")
            self._append_log("实时检测已停止。")
            return

        model_path = Path(self.model_edit.text().strip())
        conf = float(self.conf_edit.text().strip())
        use_guided = self.use_guided_checkbox.isChecked()

        self.rt_worker = RealtimeWorker(model_path, conf, use_guided, self.sg_cfg)
        self.rt_worker.frame_signal.connect(self._on_frame)
        self.rt_worker.error_signal.connect(self._on_error)
        self.rt_worker.start()
        self.realtime_btn.setText("停止实时检测")
        self._append_log("实时检测已启动。")

    def _run_batch(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.information(self, "提示", "批处理正在进行中。")
            return

        root = Path(__file__).resolve().parents[2]
        output_dir = root / "experiments" / "ui_batch_results"

        model_path = Path(self.model_edit.text().strip())
        input_dir = Path(self.input_dir_edit.text().strip())
        conf = float(self.conf_edit.text().strip())
        use_guided = self.use_guided_checkbox.isChecked()

        self.batch_worker = BatchWorker(model_path, input_dir, output_dir, conf, use_guided, self.sg_cfg)
        self.batch_worker.progress_signal.connect(self.progress.setValue)
        self.batch_worker.done_signal.connect(self._on_batch_done)
        self.batch_worker.error_signal.connect(self._on_error)
        self.progress.setValue(0)
        self.batch_worker.start()
        self._append_log("批处理已启动。")

    def _on_frame(self, frame_bgr, fps: float) -> None:
        pix = _to_qpixmap(frame_bgr)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.fps_label.setText(f"实时FPS: {fps:.2f}")
        if fps >= 25.0:
            self.target_label.setText("目标: >= 25 FPS (已达成)")
        else:
            self.target_label.setText("目标: >= 25 FPS (未达成，建议减小imgsz/更换GPU)")

    def _on_error(self, msg: str) -> None:
        self._append_log(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _on_batch_done(self, msg: str) -> None:
        self._append_log(msg)
        QMessageBox.information(self, "完成", msg)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.rt_worker and self.rt_worker.isRunning():
            self.rt_worker.stop()
            self.rt_worker.wait()
        event.accept()
