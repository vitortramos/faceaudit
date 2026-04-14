from PyQt6.QtWidgets import QFileDialog, QMessageBox

from faceaudit.models.filter_config import FilterConfig
from .filter_dialog_view import FilterDialogView


class FilterDialogController:
    """
    Controlador de Filtros.
    Responsável APENAS por:
    - Popular combos a partir do dataset
    - Carregar/Salvar FilterConfig (Agora em JSON)
    - Converter UI <-> FilterConfig
    """

    NUMERIC_FIELDS = {
        "fiqa_score", "ofiq_score", "serfiq_score",
        "ofiq_yaw", "ofiq_pitch", "ofiq_roll",
        "ff_yaw", "ff_pitch", "ff_roll",
        "age", "ipd",
    }

    def __init__(self, dataset, current_config: FilterConfig = None, parent=None):
        self.dataset = dataset
        self.config = current_config or FilterConfig()
        self.view = FilterDialogView(parent)

        self._populate_combos_from_dataset()
        self._connect_signals()
        self._load_ui_from_config()

    def exec(self):
        return self.view.exec()

    def get_config(self) -> FilterConfig:
        return self._build_config_from_ui()

    def _populate_combos_from_dataset(self):
        if not self.dataset:
            return

        df = self.dataset.to_pandas()
        if df is None or df.empty:
            return

        def get_unique(col_name: str):
            if col_name not in df.columns:
                return []
            series = df[col_name].dropna().astype(str)
            series = series[series.str.strip() != ""]
            return series.unique().tolist()

        self.view.populate_combo_options("identity", get_unique("identity"))
        self.view.populate_combo_options("gender", get_unique("gender"))
        self.view.populate_combo_options("ethnicity", get_unique("ethnicity"))

        if "origin" in df.columns:
            origins = get_unique("origin")
        else:
            origins = get_unique("source")
        self.view.populate_combo_options("source", origins)

    def _connect_signals(self):
        self.view.btn_load_cfg.clicked.connect(self._action_load_file)
        self.view.btn_save_cfg.clicked.connect(self._action_save_file)

    def _load_ui_from_config(self):
        data = {}
        for attr in self.view.widgets_map.keys():
            if hasattr(self.config, attr):
                data[attr] = getattr(self.config, attr)
        self.view.set_form_data(data)

    def _is_numeric_key(self, key: str) -> bool:
        base = key
        for prefix in ("min_", "max_", "abs_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
        return base in self.NUMERIC_FIELDS

    def _build_config_from_ui(self) -> FilterConfig:
        raw_data = self.view.get_raw_form_data()

        for key, raw_val in raw_data.items():
            # Checkboxes
            if isinstance(raw_val, bool):
                setattr(self.config, key, raw_val)
                continue

            # Combos: None já significa "sem filtro"
            if raw_val is None:
                setattr(self.config, key, None)
                continue

            # LineEdits / strings
            text = str(raw_val).strip().replace(",", ".")
            if not text:
                setattr(self.config, key, None)
                continue

            if self._is_numeric_key(key):
                try:
                    setattr(self.config, key, float(text))
                except ValueError:
                    setattr(self.config, key, None)
            else:
                setattr(self.config, key, text)

        return self.config

    # --- Persistência (ATUALIZADA PARA JSON) ---
    def _action_save_file(self):
        self._build_config_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self.view,
            "Salvar Filtro",
            "filtro_auditoria.json",
            "JSON (*.json)"
        )
        if path:
            try:
                self.config.to_json(path)
                QMessageBox.information(self.view, "Sucesso", "Configuração salva com sucesso.")
            except Exception as e:
                QMessageBox.critical(self.view, "Erro", f"Falha ao salvar: {e}")

    def _action_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Carregar Filtro",
            "",
            "JSON (*.json)"
        )
        if path:
            try:
                new_cfg = FilterConfig.from_json(path)
                self.config = new_cfg
                self._load_ui_from_config()
            except Exception as e:
                QMessageBox.critical(self.view, "Erro", f"Falha ao carregar: {e}")