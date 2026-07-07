"""Sistema de tema (claro/escuro) da UI do FaceAudit.

Ponto de entrada: `ThemeManager`. Tokens semânticos em `tokens`.
"""

from .theme_manager import ThemeManager, build_palette, build_global_qss
from .tokens import TOKENS, color, normalize_mode, VALID_MODES, DEFAULT_MODE
from .themed_widget import apply_themed_stylesheet, current_token_fn

__all__ = [
    "ThemeManager",
    "build_palette",
    "build_global_qss",
    "apply_themed_stylesheet",
    "current_token_fn",
    "TOKENS",
    "color",
    "normalize_mode",
    "VALID_MODES",
    "DEFAULT_MODE",
]
