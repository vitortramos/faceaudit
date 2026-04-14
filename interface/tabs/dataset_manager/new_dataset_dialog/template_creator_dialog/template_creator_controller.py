from pathlib import Path
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QMessageBox
from faceaudit.models.dir_template import DirTemplate

# IMPORTAÇÃO RELATIVA LOCAL
from .template_creator_view import TemplateCreatorView

class TemplateCreatorController(QtCore.QObject):
    """
    Controlador do Criador de Templates.
    Processa as listas visuais, gera o objeto DirTemplate e salva em disco.
    """
    def __init__(self, parent_view=None, save_directory: Path = None):
        super().__init__()
        # Define onde salvar
        self.save_directory = save_directory if save_directory else Path("data/templates")
        self._ensure_dir()

        self.view = TemplateCreatorView(parent_view)
        self._connect_signals()

    def exec(self):
        return self.view.exec()

    def _ensure_dir(self):
        if not self.save_directory.exists():
            try:
                self.save_directory.mkdir(parents=True, exist_ok=True)
            except: pass

    def _connect_signals(self):
        # Ações de Botões Locais (Add/Rem)
        self.view.btn_add_f.clicked.connect(self._on_add_folder)
        self.view.btn_rem_f.clicked.connect(self.view.remove_folder_item)
        
        self.view.btn_add_file.clicked.connect(self._on_add_file)
        self.view.btn_rem_file.clicked.connect(self.view.remove_file_item)
        
        # Interceptação do botão Save
        try:
            self.view.buttons.accepted.disconnect()
        except: pass
        self.view.buttons.accepted.connect(self._on_save_attempt)

    def _on_add_folder(self):
        level = self.view.spin_folder_level.value()
        attr = self.view.cmb_folder_attr.currentText()
        self.view.add_folder_item(level, attr)

    def _on_add_file(self):
        idx = self.view.spin_file_idx.value()
        attr = self.view.cmb_file_attr.currentText()
        self.view.add_file_item(idx, attr)

    def _on_save_attempt(self):
        """Valida e salva o template em formato JSON."""
        name = self.view.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self.view, "Erro", "Defina um nome para o template.")
            return

        # 1. Processa Mapeamento de Pastas
        f_map = self._parse_folder_list()
        
        # 2. Processa Mapeamento de Arquivos
        fn_map = self._parse_file_list()

        try:
            template = DirTemplate(
                name=name,
                separator=self.view.txt_separator.text(),
                use_separator=self.view.chk_use_sep.isChecked(),
                folder_map=f_map,
                filename_map=fn_map,
                identity_from_parent=self.view.chk_parent.isChecked()
            )
            
            # ATUALIZAÇÃO: Extensão e método de persistência JSON
            path = self.save_directory / f"{name}.json"
            template.to_json(path)
            
            QMessageBox.information(self.view, "Sucesso", f"Template salvo em:\n{path}")
            self.view.accept() # Fecha o diálogo com sucesso
            
        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao salvar: {e}")

    # --- Parsers ---

    def _parse_folder_list(self):
        """Lê a lista visual de pastas e converte para dicionário {nivel: [attrs]}."""
        f_map = {}
        list_widget = self.view.list_folders
        
        for i in range(list_widget.count()):
            text = list_widget.item(i).text()
            try:
                # Parser: "[Nível X] -> attr"
                part1 = text.split("] ->")[0]
                level = int(part1.split("Nível ")[1])
                attr = text.split("-> ")[1].strip()
                
                if level not in f_map:
                    f_map[level] = []
                f_map[level].append(attr)
            except:
                continue
        return f_map

    def _parse_file_list(self):
        """Lê a lista visual de arquivos e converte para dicionário {idx: attr}."""
        fn_map = {}
        list_widget = self.view.list_files
        
        for i in range(list_widget.count()):
            text = list_widget.item(i).text()
            try:
                # Parser: "[Parte X] -> attr"
                part1 = text.split("] ->")[0]
                idx = int(part1.split("Parte ")[1])
                attr = text.split("-> ")[1].strip()
                fn_map[idx] = attr
            except:
                continue
        return fn_map