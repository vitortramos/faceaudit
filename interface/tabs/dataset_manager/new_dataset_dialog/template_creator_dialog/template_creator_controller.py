from pathlib import Path
from typing import Dict, List, Tuple

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from faceaudit.models.dir_template import BUILTIN_COLUMNS, IGNORE_TOKEN, DirTemplate

# IMPORTAÇÃO RELATIVA LOCAL
from .template_creator_view import TemplateCreatorView


class TemplateCreatorController(QtCore.QObject):
    """Controlador do Criador de Templates.

    Mantém estado intermediário das três abas (Colunas / Pastas /
    Arquivos), garante consistência entre elas (combos de atributo
    refletem o universo de colunas declaradas) e produz um `DirTemplate`
    serializado em JSON no save.
    """

    def __init__(self, parent_view=None, save_directory: Path = None):
        super().__init__()
        self.save_directory = save_directory if save_directory else Path("data/templates")
        self._ensure_dir()

        self.view = TemplateCreatorView(parent_view)
        self._connect_signals()
        # Popula os combos com built-ins desde o início
        self._refresh_attribute_combos()

    def exec(self):
        return self.view.exec()

    def _ensure_dir(self):
        if not self.save_directory.exists():
            try:
                self.save_directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Erro real (permissão, disco cheio, path inválido) — não
                # silenciar: o save vai falhar depois com mensagem genérica.
                print(f"⚠ Falha ao criar diretório de templates {self.save_directory}: {e}")

    def _connect_signals(self):
        # Aba Colunas
        self.view.btn_add_column.clicked.connect(self._on_add_custom_column)
        self.view.btn_remove_column.clicked.connect(self._on_remove_custom_column)

        # Aba Pastas
        self.view.btn_add_f.clicked.connect(self._on_add_folder)
        self.view.btn_rem_f.clicked.connect(self.view.remove_folder_item)

        # Aba Arquivos
        self.view.btn_add_file.clicked.connect(self._on_add_file)
        self.view.btn_rem_file.clicked.connect(self.view.remove_file_item)

        # Interceptação do botão Save: desconecta o handler default do QDialogButtonBox
        # antes de injetar o nosso. Qt levanta TypeError quando não há slot conectado —
        # esse caso é benigno e silenciado; demais erros propagam.
        try:
            self.view.buttons.accepted.disconnect()
        except TypeError:
            pass
        self.view.buttons.accepted.connect(self._on_save_attempt)

    # ───────────────────────── Aba Colunas ──────────────────────────

    def _on_add_custom_column(self) -> None:
        name, ok = QInputDialog.getText(
            self.view,
            "Nova coluna custom",
            "Nome (minúsculas, sem espaços):",
        )
        if not ok:
            return
        key = (name or "").strip().lower().replace(" ", "_")
        if not key:
            QMessageBox.warning(self.view, "Nome inválido", "O nome não pode ser vazio.")
            return
        if key in BUILTIN_COLUMNS:
            QMessageBox.warning(
                self.view, "Nome conflitante",
                f"'{key}' já existe como coluna built-in.",
            )
            return
        if key in self.view.get_custom_column_keys():
            QMessageBox.warning(
                self.view, "Nome duplicado",
                f"A coluna '{key}' já foi adicionada.",
            )
            return
        if key == IGNORE_TOKEN:
            QMessageBox.warning(
                self.view, "Nome reservado",
                f"'{IGNORE_TOKEN}' é uma palavra reservada — escolha outro nome.",
            )
            return

        self.view.add_custom_column_item(key)
        self._refresh_attribute_combos()

    def _on_remove_custom_column(self) -> None:
        key = self.view.selected_column_key()
        if not key or key in BUILTIN_COLUMNS:
            # Botão deveria estar desabilitado; double-check.
            return

        # Antes de remover, verifica se essa coluna está em algum mapeamento.
        # Bloqueia com mensagem explicando onde está em uso.
        folder_uses, file_uses = self._find_usages_in_lists(key)
        if folder_uses or file_uses:
            lines = []
            if folder_uses:
                lines.append("📁 Estrutura de Pastas:")
                lines.extend(f"  • {u}" for u in folder_uses)
            if file_uses:
                lines.append("📄 Nome do Arquivo:")
                lines.extend(f"  • {u}" for u in file_uses)
            QMessageBox.information(
                self.view,
                "Coluna em uso",
                f"A coluna '{key}' está mapeada em:\n\n"
                + "\n".join(lines)
                + "\n\nRemova os mapeamentos correspondentes antes de excluir a coluna.",
            )
            return

        self.view.remove_selected_custom_column()
        self._refresh_attribute_combos()

    def _find_usages_in_lists(self, column_key: str) -> Tuple[List[str], List[str]]:
        """Varre as ListWidgets de mapeamento procurando pela chave.
        Retorna (folder_descriptions, file_descriptions).
        """
        folder_uses: List[str] = []
        file_uses: List[str] = []

        for i in range(self.view.list_folders.count()):
            text = self.view.list_folders.item(i).text()
            level, attr = self._parse_folder_item(text)
            if attr == column_key and level is not None:
                folder_uses.append(f"Nível {level}")

        for i in range(self.view.list_files.count()):
            text = self.view.list_files.item(i).text()
            idx, attr = self._parse_file_item(text)
            if attr == column_key and idx is not None:
                file_uses.append(f"Parte {idx}")

        return folder_uses, file_uses

    def _refresh_attribute_combos(self) -> None:
        """Repopula os combos das abas 2 e 3 com (built-ins + customs)."""
        keys = list(BUILTIN_COLUMNS.keys()) + self.view.get_custom_column_keys()
        self.view.populate_attribute_combos(keys)

    # ───────────────────────── Aba Pastas / Arquivos ────────────────

    def _on_add_folder(self):
        level = self.view.spin_folder_level.value()
        # Lemos a CHAVE canônica do combo (userData), não o label exibido.
        attr_key = self.view.cmb_folder_attr.currentData() or IGNORE_TOKEN
        self.view.add_folder_item(level, attr_key)

    def _on_add_file(self):
        idx = self.view.spin_file_idx.value()
        attr_key = self.view.cmb_file_attr.currentData() or IGNORE_TOKEN
        self.view.add_file_item(idx, attr_key)

    # ───────────────────────── Save ─────────────────────────────────

    def _on_save_attempt(self):
        """Valida e salva o template em formato JSON."""
        name = self.view.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self.view, "Erro", "Defina um nome para o template.")
            return

        f_map = self._parse_folder_list()
        fn_map = self._parse_file_list()
        custom_columns = self.view.get_custom_column_keys()

        try:
            template = DirTemplate(
                name=name,
                folder_separator=self.view.txt_folder_separator.text() or "_",
                use_folder_separator=self.view.chk_use_folder_sep.isChecked(),
                filename_separator=self.view.txt_filename_separator.text() or "_",
                use_filename_separator=self.view.chk_use_filename_sep.isChecked(),
                custom_columns=custom_columns,
                folder_map=f_map,
                filename_map=fn_map,
                identity_from_parent=self.view.chk_parent.isChecked(),
            )

            path = self.save_directory / f"{name}.json"
            template.to_json(path)

            QMessageBox.information(self.view, "Sucesso", f"Template salvo em:\n{path}")
            self.view.accept()

        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao salvar: {e}")

    # ───────────────────────── Parsers internos ─────────────────────

    @staticmethod
    def _parse_folder_item(text: str) -> Tuple:
        """Tenta extrair (level:int, attr_key:str) de uma linha visual.
        Retorna (None, None) se mal formada."""
        try:
            part1 = text.split("] ->")[0]
            level = int(part1.split("Nível ")[1])
            attr = text.split("-> ")[1].strip()
            return level, attr
        except (IndexError, ValueError):
            return None, None

    @staticmethod
    def _parse_file_item(text: str) -> Tuple:
        try:
            part1 = text.split("] ->")[0]
            idx = int(part1.split("Parte ")[1])
            attr = text.split("-> ")[1].strip()
            return idx, attr
        except (IndexError, ValueError):
            return None, None

    def _parse_folder_list(self) -> Dict[int, List[str]]:
        f_map: Dict[int, List[str]] = {}
        for i in range(self.view.list_folders.count()):
            text = self.view.list_folders.item(i).text()
            level, attr = self._parse_folder_item(text)
            if level is None:
                print(f"⚠ Item de pasta mal formado, ignorado: {text!r}")
                continue
            f_map.setdefault(level, []).append(attr)
        return f_map

    def _parse_file_list(self) -> Dict[int, str]:
        fn_map: Dict[int, str] = {}
        for i in range(self.view.list_files.count()):
            text = self.view.list_files.item(i).text()
            idx, attr = self._parse_file_item(text)
            if idx is None:
                print(f"⚠ Item de arquivo mal formado, ignorado: {text!r}")
                continue
            fn_map[idx] = attr
        return fn_map
