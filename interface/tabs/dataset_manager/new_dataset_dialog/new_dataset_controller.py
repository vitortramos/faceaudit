from pathlib import Path
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QFileDialog, QMessageBox

# Models
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.dir_template import DirTemplate

# View Local
from .new_dataset_view import NewDatasetView

# --- SUB-CONTROLLER ---
from .template_creator_dialog.template_creator_controller import TemplateCreatorController

class NewDatasetController(QtCore.QObject):
    """
    Controlador para criação de novos datasets.
    Gerencia a View principal e o sub-controller de criação de templates.
    """
    def __init__(self, parent_widget=None, templates_path: Path = None):
        super().__init__()
        
        # Inicializa a View
        self.view = NewDatasetView(parent_widget)
        
        # Gerencia o caminho dos templates
        self.templates_path = templates_path if templates_path else Path("data/templates")
        if not self.templates_path.exists():
            try:
                self.templates_path.mkdir(parents=True, exist_ok=True)
            except: pass

        self.templates_map = {} 
        
        self._setup_connections()
        self._load_templates()
        
        # Estado inicial (Força atualização da UI para o estado padrão)
        self._on_recursive_toggled(self.view.chk_recursive.isChecked())

    def exec(self) -> bool:
        """Exibe o diálogo modalmente e retorna True se Aceito."""
        return self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def _setup_connections(self):
        # Conexões da View
        self.view.btn_browse.clicked.connect(self._browse_folder)
        self.view.chk_recursive.toggled.connect(self._on_recursive_toggled)
        self.view.btn_browse_align_target.clicked.connect(
            self._browse_align_target,
        )

        # Conexão dos botões de diálogo (OK/Cancel)
        self.view.buttons.accepted.connect(self._on_accept)
        self.view.buttons.rejected.connect(self.view.reject)

        # --- AÇÃO: ABRIR O CRIADOR DE TEMPLATE ---
        self.view.btn_new_template.clicked.connect(self._abrir_criador_template)

    def _browse_align_target(self):
        path = QFileDialog.getExistingDirectory(
            self.view, "Pasta destino dos crops alinhados",
        )
        if path:
            self.view.txt_align_target.setText(path)

    def _abrir_criador_template(self):
        """
        Instancia o Sub-Controller de Templates e aguarda o resultado.
        """
        creator_ctrl = TemplateCreatorController(
            parent_view=self.view, 
            save_directory=self.templates_path
        )
        
        if creator_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Se o usuário salvou um novo template, recarregamos a lista
            self._load_templates()
            
            # Seleciona o último item adicionado (o novo template)
            count = self.view.cmb_template.count()
            if count > 0:
                self.view.cmb_template.setCurrentIndex(count - 1)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self.view, "Selecionar Pasta")
        if path:
            self.view.txt_path.setText(path)
            # Sugere nome do dataset baseado na pasta se estiver vazio
            if not self.view.txt_name.text():
                self.view.txt_name.setText(Path(path).name)

    def _on_recursive_toggled(self, checked):
        # Alterna entre a página Flat (0) e Recursiva (1) no StackedWidget
        self.view.stack_strategy.setCurrentIndex(1 if checked else 0)

    def _load_templates(self):
        """Lê os arquivos JSON da pasta de templates e preenche o ComboBox."""
        self.view.cmb_template.clear()
        
        # Item padrão (sem template)
        self.view.cmb_template.addItem("Nenhum (Usar Identidade Base)", None)
        
        if not self.templates_path.exists():
            return

        self.templates_map = {}
        
        # --- CORREÇÃO: Agora busca por .json ---
        files = sorted(list(self.templates_path.glob("*.json")))
        
        for file_path in files:
            try:
                # --- CORREÇÃO: Usando from_json() em vez de from_pickle() ---
                template = DirTemplate.from_json(file_path)
                self.templates_map[template.name] = template
                
                # Exibe com ícone visual, mas o UserData é o nome chave
                self.view.cmb_template.addItem(f"📄 {template.name}", template.name)
            except Exception as e:
                import traceback
                print(f"Erro ao carregar template {file_path.name}: {e}")
                traceback.print_exc()

    def _on_accept(self):
        # Validação Básica antes de fechar
        if not self.view.txt_path.text().strip():
            QMessageBox.warning(self.view, "Erro", "Selecione uma pasta com imagens.")
            return

        if not self.view.chk_recursive.isChecked():
            if not self.view.txt_identity_flat.text().strip():
                QMessageBox.warning(self.view, "Erro", "No modo não-recursivo, a 'Identidade' é obrigatória.")
                return

        # "Alinhar agora" requer FF habilitado (sem FF não tem como
        # alinhar) + pasta destino válida.
        if self.view.chk_align_during.isChecked():
            if not self.view.chk_ff.isChecked():
                QMessageBox.warning(
                    self.view, "Erro",
                    "Alinhar faces durante a criação requer "
                    "Forensic Face habilitado (necessário pro "
                    "detect+align).",
                )
                return
            if not self.view.txt_align_target.text().strip():
                QMessageBox.warning(
                    self.view, "Erro",
                    "Selecione a pasta destino dos crops alinhados.",
                )
                return

        self.view.accept()

    def get_config(self) -> DatasetIngestionConfig:
        """
        Retorna o objeto de configuração preenchido com os dados da UI.
        Chamado pelo Pai (DatasetManager) após o exec() retornar True.
        """
        is_recursive = self.view.chk_recursive.isChecked()
        
        # Lógica de Identidade e Template
        if is_recursive:
            # Modo Recursivo
            identity = "Unknown" # Será sobrescrito pelo template durante a ingestão
            
            # Recupera o objeto Template baseado na seleção
            idx = self.view.cmb_template.currentIndex()
            template_name = self.view.cmb_template.itemData(idx) # Pegamos o dado associado
            
            template_obj = self.templates_map.get(template_name)
            
            def_gender = None
            def_ethnicity = None
        else:
            # Modo Flat
            identity = self.view.txt_identity_flat.text().strip()
            template_obj = None
            
            # Filtra texto vazio do combo de gênero
            g_text = self.view.cmb_gender.currentText()
            def_gender = g_text if g_text else None
            
            e_text = self.view.txt_ethnicity.text().strip()
            def_ethnicity = e_text if e_text else None

        # Monta o DTO
        config = DatasetIngestionConfig(
            input_path=self.view.txt_path.text().strip(),
            dataset_name=self.view.txt_name.text().strip(),
            description=self.view.txt_desc.toPlainText().strip(),

            recursive=is_recursive,
            template=template_obj,

            default_identity=identity,
            origin=self.view.txt_origin.text().strip(),
            default_gender=def_gender,
            default_ethnicity=def_ethnicity,

            # Flags por serviço
            compute_ff=self.view.chk_ff.isChecked(),
            compute_ofiq=self.view.chk_ofiq.isChecked(),
            compute_serfiq=self.view.chk_serfiq.isChecked(),

            # Params operacionais lidos dos painéis avançados (defaults
            # se o usuário não expandiu).
            ff_models=self.view.get_ff_models(),
            ff_det_size=self.view.get_ff_det_size(),
            ff_extended=self.view.get_ff_extended(),
            serfiq_t=self.view.get_serfiq_t(),
            serfiq_alpha=self.view.get_serfiq_alpha(),

            # Flag: as imagens fonte já são faces alinhadas 112×112.
            # Propagado pra FaceDataset.is_aligned no ingestion service.
            is_aligned=self.view.chk_is_aligned.isChecked(),

            # Modo "alinhar agora": detect+align+save em 1 passo.
            align_during_ingestion=self.view.chk_align_during.isChecked(),
            aligned_target_dir=(
                Path(self.view.txt_align_target.text().strip())
                if self.view.chk_align_during.isChecked()
                and self.view.txt_align_target.text().strip()
                else None
            ),
        )
        return config