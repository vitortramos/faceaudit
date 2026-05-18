"""Captura headless dos diálogos do Gerenciador de Datasets para a documentação.

Roda o app em modo headless via Xvfb, com a fonte Inter e resolução 2560x1440,
abre cada diálogo programaticamente e salva PNGs em ../assets/.

Pré-requisitos:
  - Xvfb (`apt-get install xvfb`)
  - Fonte Inter (`apt-get install fonts-inter`)
  - Um parquet de teste + pasta de imagens correspondente

Variáveis de ambiente (com defaults):
  FACEAUDIT_TEST_PARQUET   = /tmp/faceaudit_doc_assets/Dataset_teste.parquet
  FACEAUDIT_TEST_IMAGES    = /tmp/faceaudit_doc_assets/teste_alinhado
  FACEAUDIT_DOC_ASSETS_DIR = <dir do script>/../assets

Uso:
  xvfb-run -a -s "-screen 0 2560x1440x24" python capture_dataset_manager.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]

# Setup sys.path: repo root + stub forensicface
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR / "fake_forensicface"))
os.chdir(str(_REPO_ROOT))

sys.setswitchinterval(0.001)

# Stub serviços que exigem binários externos (OFIQ, SERFIQ).
import faceaudit.services.ofiq_service as _ofiq_mod
def _stub_ofiq_init(self, ofiq_exe_path=None, ofiq_data_path=None):
    self.exe_path = ofiq_exe_path or "/stub/OFIQSampleApp"
    self.config_path = ofiq_data_path or "/stub/data"
_ofiq_mod.OFIQService.__init__ = _stub_ofiq_init

try:
    import faceaudit.services.serfiq_service as _serfiq_mod
    def _stub_serfiq(self, *args, **kwargs):
        pass
    _serfiq_mod.SERFIQService.__init__ = _stub_serfiq
except Exception:
    pass

from PyQt6 import QtWidgets, QtGui, QtCore

from interface.run import (
    _apply_light_palette,
    bootstrap_configuration,
    APP_CONFIG_JSON,
    UI_MAIN,
)
from faceaudit.containers import ApplicationContainer
from interface.main_gui_view import MainGuiView
from interface.main_gui_controller import MainGuiController
from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy


PARQUET_PATH = os.environ.get(
    "FACEAUDIT_TEST_PARQUET",
    "/tmp/faceaudit_doc_assets/Dataset_teste.parquet",
)
IMAGES_DIR = os.environ.get(
    "FACEAUDIT_TEST_IMAGES",
    "/tmp/faceaudit_doc_assets/teste_alinhado",
)
OUT_DIR = Path(os.environ.get(
    "FACEAUDIT_DOC_ASSETS_DIR",
    str(_SCRIPT_DIR.parent / "assets"),
))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dialog intercept ────────────────────────────────────────────────────────
_dialog_label = {"name": None}
_capture_log: list[tuple[str, int, int]] = []


def install_dialog_intercept() -> None:
    """Replace QDialog.exec to show + grab + close (returns Rejected)."""
    def fake_exec(self):
        label = _dialog_label["name"] or f"dialog_{self.__class__.__name__}"
        try:
            self.show()
            self.raise_()
            for _ in range(8):
                QtCore.QCoreApplication.processEvents()
            pix = self.grab()
            out = OUT_DIR / f"{label}.png"
            pix.save(str(out))
            _capture_log.append((label, pix.width(), pix.height()))
            print(f"  ✓ {label}.png ({pix.width()}x{pix.height()})")
        except Exception as e:
            print(f"  ✗ {label} failed: {e}")
        finally:
            try:
                self.close()
            except Exception:
                pass
        return QtWidgets.QDialog.DialogCode.Rejected

    QtWidgets.QDialog.exec = fake_exec

    # Neutraliza diálogos de arquivo (alguns fluxos abrem antes do próprio
    # diálogo principal; deixá-los travar bloquearia a captura).
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))


def grab_window(view, name: str) -> None:
    for _ in range(5):
        QtCore.QCoreApplication.processEvents()
    pix = view.grab()
    out = OUT_DIR / f"{name}.png"
    pix.save(str(out))
    print(f"  ✓ {name}.png ({pix.width()}x{pix.height()})")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Inter", 10))
    _apply_light_palette(app)
    install_dialog_intercept()

    bootstrap_configuration(APP_CONFIG_JSON)
    container = ApplicationContainer()
    container.config.from_json(APP_CONFIG_JSON)
    container.config.system.config_file.from_value(APP_CONFIG_JSON)

    view = MainGuiView(UI_MAIN)
    controller = MainGuiController(view, container, APP_CONFIG_JSON)
    view.resize(2560, 1440)
    view.showMaximized()

    state = {"step": 0}

    def step():
        s = state["step"]
        try:
            if s == 0:
                grab_window(view, "00_main_strategy_manager_default")

            elif s == 1:
                view.main_tabs.setCurrentIndex(0)
                grab_window(view, "10_dataset_manager_empty")

            elif s == 2:
                dm = controller.manager_controller
                dm.open_dataset_from_path(PARQUET_PATH)

            elif s == 3:
                # Rebase paths so images render
                from faceaudit.services.dataset_localization_service import (
                    DatasetLocalizationService,
                )
                dm = controller.manager_controller
                service = DatasetLocalizationService()
                detected = service.detect_path_prefix(dm.dataset)
                result = service.localize(
                    dataset=dm.dataset,
                    target_dir=IMAGES_DIR,
                    mode=LocalizationMode.REBASE_PREFIX,
                    naming=NamingStrategy.PRESERVE,
                    skip_existing=True,
                    old_prefix=detected,
                )
                dm.carregar_dataset_na_tela(result.dataset, PARQUET_PATH)
                tbl = dm.table_ctrl.view
                if tbl.model() and tbl.model().rowCount() > 0:
                    tbl.selectRow(0)
                    tbl.setCurrentIndex(tbl.model().index(0, 0))
                grab_window(view, "11_dataset_loaded_with_image")

            elif s == 4:
                tb = controller.manager_controller.toolbar_ctrl.view
                for i, name in enumerate(["arquivo", "dataset", "exibir", "imagens"]):
                    tb.tabs.setCurrentIndex(i)
                    for _ in range(5):
                        QtCore.QCoreApplication.processEvents()
                    grab_window(view, f"20_ribbon_{i}_{name}")
                tb.tabs.setCurrentIndex(0)

            elif s == 5:
                dp = controller.manager_controller.detail_ctrl.view
                dp.tabs.setCurrentIndex(1)
                grab_window(view, "12_detail_quality_iso")
                dp.tabs.setCurrentIndex(0)
                grab_window(view, "12_detail_geral_bio")

            # ── Dialogs ──────────────────────────────────────────────────
            elif s == 6:
                _dialog_label["name"] = "30_novo_dataset"
                controller.manager_controller.fluxo_novo_dataset()
            elif s == 7:
                _dialog_label["name"] = "31_localizar"
                controller.manager_controller.fluxo_localizar()
            elif s == 8:
                _dialog_label["name"] = "32_enriquecer"
                controller.manager_controller.fluxo_enriquecer()
            elif s == 9:
                _dialog_label["name"] = "33_filtrar"
                controller.manager_controller.fluxo_filtrar()
            elif s == 10:
                _dialog_label["name"] = "34_estatisticas"
                controller.manager_controller.fluxo_estatisticas()
            elif s == 11:
                _dialog_label["name"] = "35_schema_report"
                controller.manager_controller.fluxo_schema_report()
            elif s == 12:
                _dialog_label["name"] = "36_dividir"
                controller.manager_controller.fluxo_split()
            elif s == 13:
                _dialog_label["name"] = "37_balancear"
                controller.manager_controller.fluxo_balancear()
            elif s == 14:
                _dialog_label["name"] = "38_add_column"
                controller.manager_controller.fluxo_adicionar_coluna()
            elif s == 15:
                _dialog_label["name"] = "39_metadados"
                controller.manager_controller.fluxo_editar_metadados()
            elif s == 16:
                _dialog_label["name"] = "40_auditar"
                controller.manager_controller.fluxo_analise_dataset()
            elif s == 17:
                _dialog_label["name"] = "41_image_pipeline_editor"
                controller.manager_controller.fluxo_image_pipeline_editor()
            elif s == 18:
                _dialog_label["name"] = "42_aplicar_image_pipeline"
                controller.manager_controller.fluxo_aplicar_image_pipeline()
            elif s == 19:
                _dialog_label["name"] = "43_audit_log"
                controller.manager_controller.fluxo_audit_log()

            elif s == 20:
                print("\n--- CAPTURED ---")
                for label, w, h in _capture_log:
                    print(f"  {label}: {w}x{h}")
                app.quit()
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Step {s} error: {e}")

        state["step"] += 1
        QtCore.QTimer.singleShot(800, step)

    QtCore.QTimer.singleShot(2000, step)
    app.exec()


if __name__ == "__main__":
    main()
