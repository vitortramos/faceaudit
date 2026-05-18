"""Stub of forensicface.app — used only for headless screenshot capture.

Allows the Dataset Manager UI to initialize without requiring the real
ONNX models (which are large and not part of this repository). Methods
return None / no-ops; calling code that would actually run inference
will not produce meaningful results. Do NOT use this stub for real
dataset creation — only for capturing UI screenshots.
"""


class ForensicFace:
    def __init__(self, *args, **kwargs):
        self.models = kwargs.get("models", [])
        self.det_size = kwargs.get("det_size", 320)
        self.use_gpu = kwargs.get("use_gpu", False)

    def process_image(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop
