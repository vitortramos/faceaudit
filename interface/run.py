import sys
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore

# ─── Tuning anti-"Não respondendo" no Windows ─────────────────────────────
#
# Quando workers Python (ex: pickle.dump de modelos grandes) seguram a GIL
# por muitos segundos, a main thread Qt fica starved e não drena a fila de
# mensagens do Windows. O OS então substitui a janela pelo "ghost" com
# título "(Não respondendo)" e overlay branco.
#
# Duas mitigações combinadas, aplicadas ANTES de qualquer outro import:
#
# 1) sys.setswitchinterval(0.001) — reduz o intervalo de troca da GIL de 5ms
#    (default) para 1ms. A main thread ganha 5× mais turnos para processar
#    eventos do Qt durante operações Python-puras concorrentes.
#
# 2) user32.DisableProcessWindowsGhosting() — desabilita o ghost window do
#    Windows. A janela continua reagindo assim que a GIL libera, mas SEM o
#    overlay "Não respondendo" cosmético no título. Sem efeito colateral.
sys.setswitchinterval(0.001)
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.user32.DisableProcessWindowsGhosting()
    except Exception:
        # Falha aqui é puramente cosmética — não impede a app de subir.
        pass

# --- AJUSTE DE PATHS ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# 1. DEIXAMOS APENAS A RAIZ DO PROJETO NO SYS.PATH
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Imports da Aplicação
# 2. AGORA OS IMPORTS LOCAIS PRECISAM DO PREFIXO 'interface.'
from interface.main_gui_view import MainGuiView
from interface.main_gui_controller import MainGuiController
from faceaudit.containers import ApplicationContainer
from faceaudit.models.system_config import AppConfig

# Caminhos de recursos
UI_MAIN = str(CURRENT_DIR / "resources" / "main_window.ui")
APP_CONFIG_JSON = str(ROOT_DIR / "config.json")

def bootstrap_configuration(config_path: str) -> None:
    """Garante que o config.json existe E está em formato novo antes do
    container ler. O ConfigLoaderService.__init__ detecta formato legado
    (paths.datasets/pipelines/strategies separados, ff.model/det_size,
    serfiq.*, plda.*) e regrava em formato novo (paths.library +
    paths.dir_templates) — vital para o container conseguir resolver
    `config.paths.library` no `_datasets_dir` provider.
    """
    from faceaudit.services.config_loader_service import ConfigLoaderService
    ConfigLoaderService(config_file=config_path)
    print(f"✅ Configuração pronta: {Path(config_path).resolve()}")

def _apply_light_palette(app: QtWidgets.QApplication) -> None:
    """
    Força palette clara independente do modo dark/light do SO.

    A UI mistura widgets cujo CSS tem cores fixas (ex: '#fff', '#fafdf6',
    'color: #2e7d32') com widgets que usam a palette padrão do Qt. Quando o
    macOS entra em modo escuro, só a segunda metade muda — o resultado é
    contraste quebrado (texto branco em fundo branco, etc.). Travar a
    palette em cores claras evita esse desencontro até o team auditar todos
    os stylesheets pra serem theme-aware.
    """
    palette = QtGui.QPalette()
    white = QtGui.QColor(255, 255, 255)
    black = QtGui.QColor(0, 0, 0)
    almost_black = QtGui.QColor(33, 33, 33)
    light_gray = QtGui.QColor(240, 240, 240)
    mid_gray = QtGui.QColor(220, 220, 220)
    blue_highlight = QtGui.QColor(0, 120, 215)

    palette.setColor(QtGui.QPalette.ColorRole.Window, light_gray)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, almost_black)
    palette.setColor(QtGui.QPalette.ColorRole.Base, white)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, light_gray)
    palette.setColor(QtGui.QPalette.ColorRole.Text, almost_black)
    palette.setColor(QtGui.QPalette.ColorRole.Button, mid_gray)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, almost_black)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, white)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, almost_black)
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(120, 120, 120))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, blue_highlight)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, white)
    palette.setColor(QtGui.QPalette.ColorRole.Link, blue_highlight)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, black)

    # Disabled: cinza-médio em fundo claro — legível mas claramente inativo.
    disabled_text = QtGui.QColor(140, 140, 140)
    for role in (
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QPalette.ColorRole.Text,
        QtGui.QPalette.ColorRole.ButtonText,
        QtGui.QPalette.ColorRole.HighlightedText,
    ):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, disabled_text)

    app.setPalette(palette)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_light_palette(app)

    # 1) Garante que o JSON existe
    bootstrap_configuration(APP_CONFIG_JSON)

    # 2) Cria o container
    container = ApplicationContainer()

    # 3) Carrega o conteúdo do JSON no container
    container.config.from_json(APP_CONFIG_JSON)

    # 4) Define explicitamente o caminho do config.json
    container.config.system.config_file.from_value(APP_CONFIG_JSON)

    # Wiring do dependency-injector: a nova arch não usa @inject (recursos
    # vão pelo construtor do StrategyExecutorFactory), então não há módulos
    # a registrar. O bloco antigo apontava pra plda_manager/kde_manager,
    # ambos movidos pra `Código Velho/`.

    try:
        view = MainGuiView(UI_MAIN)
        controller = MainGuiController(view, container, APP_CONFIG_JSON)
        view.showMaximized()
        sys.exit(app.exec())
    except Exception:
        import traceback
        traceback.print_exc()
        raise