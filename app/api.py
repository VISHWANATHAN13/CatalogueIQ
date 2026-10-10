from __future__ import annotations

import os

try:
    from .gradio_ui import demo
except ImportError:  # pragma: no cover
    from app.gradio_ui import demo


if __name__ == "__main__":
    demo.launch()
