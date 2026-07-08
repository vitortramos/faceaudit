"""Sistema de tema (claro/escuro) da UI do FaceAudit.

Ponto de entrada: `ThemeManager`. Tokens semânticos em `tokens`.
"""

from .theme_manager import ThemeManager, build_palette, build_global_qss
from .tokens import TOKENS, color, normalize_mode, VALID_MODES, DEFAULT_MODE
from .themed_widget import apply_themed_stylesheet, current_token_fn
from .native_titlebar import apply_titlebar_theme, apply_titlebar_theme_to_all

__all__ = [
    "ThemeManager",
    "build_palette",
    "build_global_qss",
    "apply_themed_stylesheet",
    "current_token_fn",
    "apply_titlebar_theme",
    "apply_titlebar_theme_to_all",
    "TOKENS",
    "color",
    "normalize_mode",
    "VALID_MODES",
    "DEFAULT_MODE",
]
