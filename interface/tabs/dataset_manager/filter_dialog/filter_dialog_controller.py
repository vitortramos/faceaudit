"""Controller do diálogo de filtro — orquestra view + DTOs + I/O.

Reescrito pra substituir o controller legado que conhecia 50+ fields
hardcoded. Agora delega:
  - Lógica de matching → `DatasetFilterService`
  - Persistência → `GlobalLibraryStorage` (filtros salvos em
    `~/.faceaudit/filter_configs/<name>.json`)
  - Validação contra dataset corrente → `DatasetFilterService.validate`

Aceita um `library_root` opcional pro Salvar/Salvar Como/Abrir; se
None, cai em QFileDialog manual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QInputDialog, QMessageBox,
)

from faceaudit.enums.global_library_folder import GlobalLibraryFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.filter import (
    BooleanColumnFilter,
    ColumnFilter,
    ColumnFilterDataType,
    FilterConfig,
    NumericColumnFilter,
    StringColumnFilter,
)
from faceaudit.services.dataset_filter_service import DatasetFilterService

from .filter_dialog_view import FilterDialogView


# Roles que NÃO entram no dropdown de coluna — vector/list não têm
# representação escalar pra filtro.
_NON_FILTERABLE_DTYPES = frozenset({"vector", "list"})


class FilterDialogController:
    """Conecta `FilterDialogView` com modelos + service + I/O."""

    def __init__(
        self,
        dataset: FaceDataset,
        current_config: Optional[FilterConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        library_root: Optional[Union[str, Path]] = None,
    ):
        self.dataset = dataset
        self.config = current_config or FilterConfig()
        self._library_root = (
            Path(library_root) if library_root else None
        )
        # Nome do filtro corrente em disco — `Salvar` reusa este; vira
        # vazio em "Novo" e populado em "Abrir/Salvar Como".
        self._current_disk_id: Optional[str] = None

        self.view = FilterDialogView(parent)

        # Disponibiliza colunas + popula header
        cols = self._build_available_columns(dataset)
        self.view.set_available_columns(cols)
        # Pré-computa valores únicos por coluna string pra alimentar
        # o popup ▾ multi-select do filtro de string.
        values = self._build_available_values_per_column(dataset, cols)
        self.view.set_available_values_per_column(values)
        self.view.set_header(self.config.name, self.config.description)

        # Restaura rows do config existente (se houver)
        self._populate_rows_from_config(self.config)
        self._refresh_invalid_marks()

        # Wire signals
        self.view.add_filter_requested.connect(self._on_add_filter)
        self.view.clear_all_requested.connect(self._on_clear_all)
        self.view.open_filter_requested.connect(self._on_open)
        self.view.save_filter_requested.connect(self._on_save)
        self.view.save_as_filter_requested.connect(self._on_save_as)
        self.view.apply_requested.connect(self._on_apply)

    def exec(self):
        return self.view.exec()

    def get_config(self) -> FilterConfig:
        """Constrói FilterConfig a partir do estado atual da view."""
        return self._build_config_from_view()

    # ── Inferência de colunas filtráveis ─────────────────────────────

    @staticmethod
    def _build_available_values_per_column(
        dataset: FaceDataset,
        columns: List[Tuple[str, ColumnFilterDataType]],
    ) -> dict:
        """Pra cada coluna STRING, coleta os valores únicos.

        **Prioriza dados sobre vocabulário** — vocab é só uma lista
        canônica de "opções esperadas" (ex: gender = {Desconhecido, F,
        M, Outro}), mas o dataset pode ter valores legados ("Male"/
        "Female") que NÃO casam com o vocab. Filtrar só por vocab
        produziria zero matches.

        Estratégia: união (valores do dataframe ∪ vocab), com prioridade
        visual pros valores que efetivamente existem no dado. Quando o
        df está vazio nessa coluna, cai pra vocab sozinho.

        Cap de 500 valores pra colunas alta-cardinalidade (identity)
        — UI fica inviável e o picker desabilita; usuário cai no
        LineEdit livre.
        """
        out: dict = {}
        specs = getattr(dataset, "column_specs", None) or {}
        df = None  # lazy
        MAX_PICKER_VALUES = 500

        for col_name, dt in columns:
            if dt != ColumnFilterDataType.STRING:
                continue

            # Valores efetivamente presentes no dataframe (fonte
            # primária — são esses que dão match no filtro).
            data_values: List[str] = []
            if df is None:
                try:
                    df = dataset.to_pandas()
                except Exception:
                    df = None
            if df is not None and col_name in df.columns:
                try:
                    series = df[col_name].dropna().astype(str)
                    series = series[series.str.strip() != ""]
                    data_values = series.unique().tolist()
                except Exception:
                    data_values = []

            # Vocab canônico (do column_spec) — adicionado depois dos
            # valores reais, pra perito ver as opções "esperadas" e
            # poder pré-selecionar mesmo que ainda não tenham aparecido
            # no dado.
            spec = specs.get(col_name)
            vocab = list(getattr(spec, "vocabulary", None) or [])

            # União ordenada: dados primeiro (alfabético), depois vocab
            # que falta (mantém ordem do spec)
            data_set = set(data_values)
            uniques = sorted(data_set) + [
                str(v) for v in vocab if str(v) not in data_set
            ]

            if not uniques:
                continue
            if len(uniques) > MAX_PICKER_VALUES:
                continue
            out[col_name] = uniques
        return out

    @staticmethod
    def _build_available_columns(
        dataset: FaceDataset,
    ) -> List[Tuple[str, ColumnFilterDataType]]:
        """Lista colunas filtráveis (NUMERIC/STRING/BOOLEAN) a partir
        do `column_specs` do dataset. Filtra out vector/list (sem
        sentido escalar). Pra parquet legado sem specs, fallback
        com nomes canônicos."""
        out: List[Tuple[str, ColumnFilterDataType]] = []
        specs = getattr(dataset, "column_specs", None) or {}
        if specs:
            for name, spec in specs.items():
                dtype_value = getattr(getattr(spec, "dtype", None), "value", None)
                if dtype_value is None:
                    continue
                if dtype_value in _NON_FILTERABLE_DTYPES:
                    continue
                ft = ColumnFilterDataType.from_column_dtype(dtype_value)
                if ft is None:
                    continue
                out.append((name, ft))
        else:
            # Fallback legacy — colunas mais comuns dos parquets antigos
            legacy = [
                ("identity", ColumnFilterDataType.STRING),
                ("gender", ColumnFilterDataType.STRING),
                ("ethnicity", ColumnFilterDataType.STRING),
                ("origin", ColumnFilterDataType.STRING),
                ("source", ColumnFilterDataType.STRING),
                ("fiqa_score", ColumnFilterDataType.NUMERIC),
                ("ofiq_score", ColumnFilterDataType.NUMERIC),
                ("serfiq_score", ColumnFilterDataType.NUMERIC),
                ("yaw", ColumnFilterDataType.NUMERIC),
                ("pitch", ColumnFilterDataType.NUMERIC),
                ("roll", ColumnFilterDataType.NUMERIC),
            ]
            out.extend(legacy)
        return sorted(out, key=lambda x: x[0])

    # ── View → Config / Config → View ────────────────────────────────

    def _populate_rows_from_config(self, config: FilterConfig) -> None:
        """Limpa rows e cria uma row pra cada ColumnFilter no config."""
        self.view.clear_rows()
        for cf in config.filters:
            kwargs: dict = {}
            if isinstance(cf, NumericColumnFilter):
                kwargs["numeric_values"] = list(cf.values)
                kwargs["numeric_ranges"] = list(cf.ranges)
            elif isinstance(cf, StringColumnFilter):
                kwargs["string_values"] = list(cf.values)
            elif isinstance(cf, BooleanColumnFilter):
                kwargs["bool_value"] = bool(cf.value)
            self.view.add_row(
                column_name=cf.column_name,
                data_type=cf.data_type,
                **kwargs,
            )

    def _build_config_from_view(self) -> FilterConfig:
        name, desc = self.view.get_header()
        states = self.view.get_row_states()
        filters: List[ColumnFilter] = []
        for s in states:
            cn = s["column_name"]
            dt = s["data_type"]
            if dt == ColumnFilterDataType.NUMERIC:
                filters.append(NumericColumnFilter(
                    column_name=cn,
                    data_type=dt,
                    values=list(s.get("values") or []),
                    ranges=list(s.get("ranges") or []),
                ))
            elif dt == ColumnFilterDataType.STRING:
                filters.append(StringColumnFilter(
                    column_name=cn,
                    data_type=dt,
                    values=list(s.get("values") or []),
                ))
            elif dt == ColumnFilterDataType.BOOLEAN:
                filters.append(BooleanColumnFilter(
                    column_name=cn,
                    data_type=dt,
                    value=bool(s.get("value", False)),
                ))
        return FilterConfig(
            name=name, description=desc, filters=filters,
        )

    def _refresh_invalid_marks(self) -> None:
        """Re-valida os filtros contra o dataset corrente e marca as
        rows inválidas em vermelho."""
        cfg = self._build_config_from_view()
        invalids = DatasetFilterService.validate(self.dataset, cfg)
        marks = {inv.filter_index: inv.reason for inv in invalids}
        self.view.mark_invalid_rows(marks)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_add_filter(self) -> None:
        self.view.add_row()

    def _on_clear_all(self) -> None:
        res = QMessageBox.question(
            self.view, "Limpar tudo",
            "Remove todos os filtros da lista?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if res == QMessageBox.StandardButton.Yes:
            self.view.clear_rows()

    def _on_apply(self) -> None:
        # Build + validate podem ser lentos em datasets grandes
        # (varredura de N faces pra resolver atributos). WaitCursor
        # mantido até passar do warning ou aceitar.
        QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor),
        )
        try:
            cfg = self._build_config_from_view()
            invalids = DatasetFilterService.validate(self.dataset, cfg)
        finally:
            QApplication.restoreOverrideCursor()

        if invalids:
            # Re-marca rows pra atualizar feedback visual antes do dialog
            self.view.mark_invalid_rows(
                {inv.filter_index: inv.reason for inv in invalids}
            )
            n = len(invalids)
            res = QMessageBox.warning(
                self.view, "Filtros inválidos",
                f"<b>{n} filtro(s) inválido(s)</b> — coluna não existe no "
                f"dataset corrente. Eles serão <b>ignorados</b> ao aplicar.\n\n"
                f"Deseja continuar mesmo assim?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        self.config = cfg
        self.view.accept()

    # ── I/O — Salvar / Salvar Como / Abrir ───────────────────────────

    def _ensure_storage(self):
        """Lazy import + instancia GlobalLibraryStorage. Retorna None
        se library_root não foi configurado."""
        if self._library_root is None:
            return None
        from faceaudit.storage.global_library_storage import (
            GlobalLibraryStorage,
        )
        storage = GlobalLibraryStorage(library_root=self._library_root)
        storage.create_library_structure()
        return storage

    def _on_save(self) -> None:
        """Salva sob o nome corrente (current_disk_id). Se nunca salvou,
        cai em Save As."""
        if not self._current_disk_id:
            self._on_save_as()
            return
        cfg = self._build_config_from_view()
        try:
            self._save_to_library(cfg, self._current_disk_id)
            QMessageBox.information(
                self.view, "Salvo",
                f"Filtro salvo: {self._current_disk_id}.json",
            )
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro ao salvar", str(e),
            )

    def _on_save_as(self) -> None:
        """Pede um config_id (nome do arquivo) e salva."""
        cfg = self._build_config_from_view()
        default_id = self._sanitize_id(cfg.name)
        new_id, ok = QInputDialog.getText(
            self.view, "Salvar Como…",
            "Nome do arquivo (sem extensão):",
            text=default_id,
        )
        if not ok or not new_id.strip():
            return
        new_id = self._sanitize_id(new_id.strip())
        try:
            self._save_to_library(cfg, new_id)
            self._current_disk_id = new_id
            QMessageBox.information(
                self.view, "Salvo",
                f"Filtro salvo: {new_id}.json",
            )
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro ao salvar", str(e),
            )

    def _save_to_library(self, cfg: FilterConfig, config_id: str) -> None:
        """Persiste via GlobalLibraryStorage. Fallback pra
        QFileDialog se não houver library_root."""
        storage = self._ensure_storage()
        if storage is not None:
            storage.save_artifact(
                StrategyArtifactType.FILTER_CONFIG,
                cfg,
                config_id=config_id,
            )
            return
        # Fallback: QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Salvar Filtro",
            f"{config_id}.json", "JSON (*.json)",
        )
        if path:
            cfg.to_json(path)

    def _on_open(self) -> None:
        """Lista filtros salvos na biblioteca global; usuário escolhe um.
        Fallback pra QFileDialog se não houver library_root."""
        storage = self._ensure_storage()
        chosen_path: Optional[Path] = None

        if storage is not None:
            folder = (
                storage.library_root /
                GlobalLibraryFolder.FILTER_CONFIGS.value
            )
            if folder.exists():
                json_files = sorted(folder.glob("*.json"))
                if not json_files:
                    QMessageBox.information(
                        self.view, "Sem filtros salvos",
                        f"Nenhum filtro salvo em {folder}.",
                    )
                    return
                names = [p.stem for p in json_files]
                choice, ok = QInputDialog.getItem(
                    self.view, "Abrir Filtro",
                    "Selecione um filtro salvo:",
                    names, 0, editable=False,
                )
                if not ok:
                    return
                chosen_path = folder / f"{choice}.json"
            else:
                # Pasta ainda não existe — biblioteca vazia
                QMessageBox.information(
                    self.view, "Sem filtros salvos",
                    "A biblioteca global ainda não tem nenhum filtro.",
                )
                return
        else:
            path, _ = QFileDialog.getOpenFileName(
                self.view, "Abrir Filtro", "", "JSON (*.json)",
            )
            if not path:
                return
            chosen_path = Path(path)

        if not chosen_path or not chosen_path.exists():
            return

        # WaitCursor cobre o parse + populate + validação
        QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor),
        )
        try:
            try:
                cfg = FilterConfig.from_json(chosen_path)
            except Exception as e:
                QMessageBox.critical(
                    self.view, "Erro ao abrir",
                    f"Falha ao ler {chosen_path.name}: {e}",
                )
                return
            if cfg is None:
                return

            # Aplica na view + valida
            self.config = cfg
            self._current_disk_id = chosen_path.stem
            self.view.set_header(cfg.name, cfg.description)
            self._populate_rows_from_config(cfg)
            self._refresh_invalid_marks()
        finally:
            QApplication.restoreOverrideCursor()

        # Avisa se há inválidos
        if self.view.has_invalid_rows():
            QMessageBox.warning(
                self.view, "Filtros inválidos no arquivo",
                "Pelo menos um filtro referencia uma coluna que não "
                "existe no dataset corrente. Eles aparecem em vermelho "
                "— remova antes de aplicar (ou Aplicar mesmo assim e "
                "eles serão ignorados).",
            )

    @staticmethod
    def _sanitize_id(text: str) -> str:
        """Normaliza nome pra um ID seguro de filename."""
        out = "".join(
            c if (c.isalnum() or c in "-_.") else "_"
            for c in text.strip()
        )
        return out or "filtro"
