"""Tokens semânticos de tema do FaceAudit (claro/escuro).

Fonte única de verdade das cores da UI. NÃO cravar hex inline em widgets —
sempre referenciar um token via `ThemeManager.instance().color("TOKEN")`
ou os dicts abaixo.

Dois modos: ``light`` e ``dark``. O modo claro reproduz EXATAMENTE a paleta
histórica (`interface/run.py::_apply_light_palette`), garantindo que ativar
o sistema de tema não altere em nada a aparência clara existente.

Módulo puro (sem PyQt) — importável em testes sem QApplication.
"""

from __future__ import annotations

from typing import Dict

VALID_MODES = ("light", "dark")
DEFAULT_MODE = "light"


def normalize_mode(mode: str) -> str:
    """Normaliza para um modo válido; qualquer coisa inválida vira o default."""
    m = (mode or "").strip().lower()
    return m if m in VALID_MODES else DEFAULT_MODE


# ── Modo claro ────────────────────────────────────────────────────────────
# Os valores de paleta (WINDOW..DISABLED_TEXT) são idênticos aos hardcoded
# em _apply_light_palette — não alterar sem revisar aquele baseline.
LIGHT: Dict[str, str] = {
    # Papéis de QPalette (base do tema — Fusion respeita tudo isto)
    "WINDOW":         "#f0f0f0",  # QPalette.Window       (240,240,240)
    "WINDOW_TEXT":    "#212121",  # QPalette.WindowText   (33,33,33)
    "BASE":           "#ffffff",  # QPalette.Base
    "ALT_BASE":       "#f0f0f0",  # QPalette.AlternateBase
    "TEXT":           "#212121",  # QPalette.Text
    "BUTTON":         "#dcdcdc",  # QPalette.Button       (220,220,220)
    "BUTTON_TEXT":    "#212121",  # QPalette.ButtonText
    "TOOLTIP_BASE":   "#ffffff",  # QPalette.ToolTipBase
    "TOOLTIP_TEXT":   "#212121",  # QPalette.ToolTipText
    "PLACEHOLDER":    "#787878",  # QPalette.PlaceholderText (120,120,120)
    "HIGHLIGHT":      "#0078d7",  # QPalette.Highlight    (0,120,215)
    "HIGHLIGHT_TEXT": "#ffffff",  # QPalette.HighlightedText
    "LINK":           "#0078d7",  # QPalette.Link
    "BRIGHT_TEXT":    "#000000",  # QPalette.BrightText
    "DISABLED_TEXT":  "#8c8c8c",  # grupo Disabled        (140,140,140)

    # Tokens semânticos de UI (para migração dos setStyleSheet cravados)
    "SURFACE":        "#ffffff",  # fundo de conteúdo/inputs
    "SURFACE_ALT":    "#f0f0f0",  # fundo secundário/janela
    "SURFACE_RAISED": "#f5f5f5",  # cards/painéis elevados
    "BORDER":         "#c8c8c8",  # divisórias/bordas
    "TEXT_MUTED":     "#757575",  # texto secundário/placeholder visual

    "ACCENT":         "#E8A020",  # âmbar forense (validação/instruções)
    "ACCENT_BG":      "#FFF8E8",  # fundo âmbar suave
    "ACCENT_TEXT":    "#8B5E00",  # texto sobre fundo âmbar

    "PRIMARY":        "#1976d2",  # azul primário (info/ação)
    "PRIMARY_STRONG": "#1a3d6e",  # navy de acento em TEXTO/BORDA (tabs)
    "PRIMARY_BG":     "#e8f0fe",  # fundo de info/seleção suave

    # Cabeçalho de fundo escuro com texto claro (column headers, painéis).
    # Distinto de PRIMARY_STRONG, que é acento em texto/borda.
    "HEADER_BG":      "#2c3e50",  # fundo navy do cabeçalho
    "HEADER_TEXT":    "#ffffff",  # texto sobre o cabeçalho

    "SUCCESS":        "#2e7d32",  # genuíno/OK
    "SUCCESS_BG":     "#e8f5e9",
    "DANGER":         "#c62828",  # impostor/erro
    "DANGER_BG":      "#ffebee",
    "WARNING":        "#ff6d00",  # atenção
    "WARNING_BG":     "#fff3cd",
}

# ── Modo escuro ───────────────────────────────────────────────────────────
DARK: Dict[str, str] = {
    "WINDOW":         "#2b2b2b",
    "WINDOW_TEXT":    "#e6e6e6",
    "BASE":           "#1e1e1e",
    "ALT_BASE":       "#262626",
    "TEXT":           "#e6e6e6",
    "BUTTON":         "#3a3a3a",
    "BUTTON_TEXT":    "#e6e6e6",
    "TOOLTIP_BASE":   "#2a2a2a",
    "TOOLTIP_TEXT":   "#e6e6e6",
    "PLACEHOLDER":    "#8a8a8a",
    "HIGHLIGHT":      "#2d5a8c",
    "HIGHLIGHT_TEXT": "#ffffff",
    "LINK":           "#4a9eff",
    "BRIGHT_TEXT":    "#ffffff",
    "DISABLED_TEXT":  "#6a6a6a",

    "SURFACE":        "#1e1e1e",
    "SURFACE_ALT":    "#2b2b2b",
    "SURFACE_RAISED": "#252525",
    "BORDER":         "#3a3a3a",
    "TEXT_MUTED":     "#9e9e9e",

    "ACCENT":         "#E8A020",  # âmbar preservado (identidade forense)
    "ACCENT_BG":      "#3a3320",
    "ACCENT_TEXT":    "#E8C87A",

    "PRIMARY":        "#4a9eff",  # azul clareado p/ contraste em fundo escuro
    "PRIMARY_STRONG": "#5a9ee8",
    "PRIMARY_BG":     "#14324f",

    "HEADER_BG":      "#30363e",  # cabeçalho: banda distinta, ainda escura
    "HEADER_TEXT":    "#e6e6e6",

    "SUCCESS":        "#66bb6a",
    "SUCCESS_BG":     "#1e3a20",
    "DANGER":         "#ef5350",
    "DANGER_BG":      "#3a1f1f",
    "WARNING":        "#ffa726",
    "WARNING_BG":     "#3a3320",
}

TOKENS: Dict[str, Dict[str, str]] = {
    "light": LIGHT,
    "dark": DARK,
}

# Sanidade: os dois modos devem declarar exatamente o mesmo conjunto de tokens.
assert set(LIGHT) == set(DARK), (
    "LIGHT e DARK devem ter os mesmos tokens: "
    f"{set(LIGHT) ^ set(DARK)}"
)


def color(token_name: str, mode: str = DEFAULT_MODE) -> str:
    """Retorna o hex de um token no modo dado. Token desconhecido → preto."""
    return TOKENS[normalize_mode(mode)].get(token_name, "#000000")
