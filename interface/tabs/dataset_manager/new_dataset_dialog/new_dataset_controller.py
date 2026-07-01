from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QDialogButtonBox

from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.dir_template import DirTemplate
from faceaudit.enums.sepael_model import SepaelModel

from .new_dataset_view import NewDatasetView
from .template_creator_dialog.template_creator_controller import TemplateCreatorController


class NewDatasetController(QtCore.QObject):
    """Controlador do diálogo de criação de novo dataset (redesenho — Etapa 1).

    A view cuida da disposição + visibilidade condicional; aqui ficam browse,
    criador de template, validação reativa (OK + rodapé) e o get_config.
    """

    def __init__(
        self,
        parent_widget=None,
        templates_path: Path = None,
        ff_available: bool = True,
        ofiq_available: bool = True,
        default_dataset_dir: str = "",
    ):
        super().__init__()
        self.view = NewDatasetView(parent_widget, default_dataset_dir=default_dataset_dir)

        self.templates_path = templates_path if templates_path else Path("data/templates")
        if not self.templates_path.exists():
            try:
                self.templates_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        self.templates_map = {}

        self._setup_connections()
        self._load_templates()
        self.view.set_stage_availability(ff_available=ff_available, ofiq_available=ofiq_available)
        self._revalidate()

    def exec(self) -> bool:
        return self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted

    # ───────────────────────── conexões ─────────────────────────
    def _setup_connections(self):
        self.view.btn_browse.clicked.connect(self._browse_folder)
        self.view.btn_browse_dest.clicked.connect(self._browse_dest)
        self.view.btn_new_template.clicked.connect(self._abrir_criador_template)
        self.view.buttons.accepted.connect(self._on_accept)
        self.view.buttons.rejected.connect(self.view.reject)
        # Validação reativa.
        self.view.changed.connect(self._revalidate)

    # ───────────────────────── browse ─────────────────────────
    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self.view, "Selecionar pasta de imagens")
        if path:
            self.view.txt_path.setText(path)
            if not self.view.txt_name.text():
                self.view.txt_name.setText(Path(path).name)

    def _browse_dest(self):
        start = self.view.get_dataset_dir() or ""
        path = QFileDialog.getExistingDirectory(self.view, "Diretório destino do dataset", start)
        if path:
            self.view.txt_dest.setText(path)

    # ───────────────────────── templates ─────────────────────────
    def _abrir_criador_template(self):
        ctrl = TemplateCreatorController(parent_view=self.view, save_directory=self.templates_path)
        if ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._load_templates()
            count = self.view.cmb_template.count()
            if count > 0:
                self.view.cmb_template.setCurrentIndex(count - 1)

    def _load_templates(self):
        self.view.cmb_template.clear()
        self.view.cmb_template.addItem("Nenhum (usar identidade base)", None)
        if not self.templates_path.exists():
            return
        self.templates_map = {}
        for file_path in sorted(self.templates_path.glob("*.json")):
            try:
                template = DirTemplate.from_json(file_path)
                self.templates_map[template.name] = template
                self.view.cmb_template.addItem(f"📄 {template.name}", template.name)
            except Exception as e:
                import traceback
                print(f"Erro ao carregar template {file_path.name}: {e}")
                traceback.print_exc()

    # ───────────────────────── validação reativa ─────────────────────────
    def _first_blocker(self) -> str:
        v = self.view
        if not v.txt_path.text().strip():
            return "Selecione a pasta de imagens."
        if not v.txt_name.text().strip():
            return "Informe o nome do dataset."
        if not v.get_dataset_dir():
            return "Informe o diretório destino."
        if v.use_template() and v.cmb_template.currentData() is None:
            return "Selecione um template (ou desmarque “Usar template de colunas”)."
        if v.is_single_identity() and not v.txt_identity_flat.text().strip():
            return "Informe a identidade (modo identidade única)."
        if v.chk_serfiq.isChecked() and not v.any_model_checked():
            return "SER-FIQ precisa de ao menos um modelo de embedding."
        return ""

    def _summary(self) -> str:
        v = self.view
        entrada = "já alinhadas" if v.is_input_aligned() else "originais"
        if v.get_self_contained():
            armaz = f"autocontido ({'alinhada' if v.get_image_type() == 'aligned' else 'original'})"
        else:
            armaz = "caminho absoluto"
        if v.use_template():
            ident = "colunas via template"
        elif v.is_single_identity():
            ident = "identidade única"
        else:
            ident = "identidade por subpasta"
        servs = []
        models = [SepaelModel.from_ff_name(n).label_pt if SepaelModel.from_ff_name(n) else n
                  for n in v.get_ff_models()]
        if models:
            servs.append("FF (" + ", ".join(models) + ")")
        if v.chk_ofiq.isChecked():
            servs.append("OFIQ")
        if v.chk_serfiq.isChecked():
            servs.append("SER-FIQ")
        servs_txt = " + ".join(servs) if servs else "sem embeddings/qualidade"
        custo = "⏱ custo alto" if (v.chk_ofiq.isChecked() or v.chk_serfiq.isChecked()) else "rápido"
        return f"{entrada} · {armaz} · {ident} · {servs_txt} · {custo}"

    def _revalidate(self):
        blocker = self._first_blocker()
        self.view.lbl_blocker.setText(("⚠ " + blocker) if blocker else "")
        self.view.lbl_summary.setText(self._summary())
        ok_btn = self.view.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setEnabled(not blocker)
        ok_btn.setToolTip(blocker or "")

    def _on_accept(self):
        # OK só fica habilitado quando válido; guarda final defensiva.
        if self._first_blocker():
            self._revalidate()
            return
        self.view.accept()

    # ───────────────────────── config ─────────────────────────
    def get_config(self) -> DatasetIngestionConfig:
        v = self.view
        single = v.is_single_identity()
        recursive = not single  # template ou "por subpasta" percorrem subpastas
        template_obj = (self.templates_map.get(v.cmb_template.currentData())
                        if v.use_template() else None)

        identity = v.txt_identity_flat.text().strip() if single else "Unknown"
        g_text = v.cmb_gender.currentText() if single else ""
        e_text = v.txt_ethnicity.text().strip() if single else ""

        # Mapeamento dos rádios de armazenamento → flags de ingestão existentes.
        # (Etapa 2 liga gravar-no-destino/cópia autocontida; aqui só o que já
        #  existe é honrado: is_aligned + align_during/aligned_target_dir.)
        input_aligned = v.is_input_aligned()
        align_now = (not input_aligned) and v.get_self_contained() and v.get_image_type() == "aligned"

        return DatasetIngestionConfig(
            input_path=v.txt_path.text().strip(),
            dataset_name=v.txt_name.text().strip(),
            description=v.txt_desc.toPlainText().strip(),
            recursive=recursive,
            template=template_obj,
            default_identity=identity,
            origin=v.txt_origin.text().strip(),
            default_gender=(g_text or None),
            default_ethnicity=(e_text or None),
            compute_ff=v.any_model_checked(),
            compute_ofiq=v.chk_ofiq.isChecked(),
            compute_serfiq=v.chk_serfiq.isChecked(),
            ff_models=(v.get_ff_models() or [SepaelModel.default().ff_name]),
            ff_det_size=v.get_ff_det_size(),
            ff_extended=v.get_ff_extended(),
            serfiq_t=v.get_serfiq_t(),
            serfiq_alpha=v.get_serfiq_alpha(),
            is_aligned=input_aligned,
            align_during_ingestion=align_now,
            aligned_target_dir=(Path(v.get_dataset_dir()) if align_now and v.get_dataset_dir() else None),
        )

    # ── Getters novos (consumidos pela Etapa 2 — gravar-no-destino/autocontido) ──
    def get_dataset_dir(self) -> str:
        return self.view.get_dataset_dir()

    def get_self_contained(self) -> bool:
        return self.view.get_self_contained()

    def get_image_type(self) -> str:
        return self.view.get_image_type()
