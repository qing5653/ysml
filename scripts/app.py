from __future__ import annotations

import os
from pathlib import Path
import sys

import PyQt5
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ui.window import MainWindow


def _configure_qt_environment() -> None:
    """Force Qt to use PyQt5 plugin path to avoid OpenCV Qt plugin conflicts."""
    plugins_path = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    if plugins_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path
        os.environ["QT_PLUGIN_PATH"] = plugins_path

    # If OpenCV injected its own qt plugin dir into env, strip it to prevent xcb mismatch.
    cv2_qt_hint = "cv2/qt/plugins"
    for key in ["QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"]:
        value = os.environ.get(key)
        if value and cv2_qt_hint in value:
            os.environ[key] = plugins_path

    # Keep xcb as default when a display server is present.
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _configure_app_font(app: QApplication) -> None:
    """Ensure Chinese UI text can render by loading a bundled CJK font."""
    font_candidates = [
        ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf",
    ]

    loaded_family = None
    for font_path in font_candidates:
        if not font_path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                loaded_family = families[0]
                break

    if loaded_family is None:
        # Fall back to generic family; if system has no CJK fonts, glyphs may still be missing.
        app.setFont(QFont("Sans Serif", 10))
        print("[WARN] 未加载到内置中文字体，界面可能出现方块字。", file=sys.stderr)
        return

    app_font = QFont(loaded_family, 10)
    app.setFont(app_font)


def main() -> None:
    _configure_qt_environment()
    app = QApplication([])
    _configure_app_font(app)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
