import sys
from pathlib import Path
from PyQt6 import QtWidgets

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
    """Garante que o config.json exista antes do container ler."""
    path_obj = Path(config_path)

    if not path_obj.exists():
        print(f"⚠️ Configuração não encontrada em: {path_obj}")
        print("⚙️ Gerando arquivo padrão usando AppConfig...")
        default_config = AppConfig()
        default_config.save_to_file(str(path_obj))
    else:
        print(f"✅ Configuração encontrada: {path_obj}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # 1) Garante que o JSON existe
    bootstrap_configuration(APP_CONFIG_JSON)

    # 2) Cria o container
    container = ApplicationContainer()

    # 3) Carrega o conteúdo do JSON no container
    container.config.from_json(APP_CONFIG_JSON)

    # 4) Define explicitamente o caminho do config.json
    container.config.system.config_file.from_value(APP_CONFIG_JSON)

    # 5) WIRING (VITAL PARA O @inject FUNCIONAR NO BACKEND)
    import faceaudit.manager.plda_manager
    import faceaudit.manager.kde_manager
    
    container.wire(modules=[
        faceaudit.manager.plda_manager,
        faceaudit.manager.kde_manager,
    ])

    try:
        view = MainGuiView(UI_MAIN)
        controller = MainGuiController(view, container, APP_CONFIG_JSON)
        view.showMaximized()
        sys.exit(app.exec())
    except Exception:
        import traceback
        traceback.print_exc()
        raise