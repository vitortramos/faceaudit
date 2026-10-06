from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from faceaudit.models.dir_template import BUILTIN_COLUMNS, IGNORE_TOKEN

# Tipos oferecidos para coluna de template, com rótulo legível. Só os que
# fazem sentido num trecho de caminho de arquivo: vetor e lista não saem de
# um pedaço de nome de pasta.
TIPOS_DE_COLUNA = {
    "str": "Texto",
    "int": "Número inteiro",
    "float": "Número decimal",
    "bool": "Sim/Não",
}


class TemplateCreatorView(QDialog):
    """View do Criador de Templates de Estrutura.

    Três abas:
      1. 🏷️ Colunas — universo de colunas disponíveis (built-ins fixas
         + customs adicionáveis).
      2. 📁 Estrutura de Pastas — separador próprio + mapeamento por nível.
      3. 📄 Nome do Arquivo — separador próprio + mapeamento por parte.

    Os combos das abas 2 e 3 são populados dinamicamente a partir do
    conteúdo da aba 1 (built-ins + customs + "Ignorar").
    """

    # Role usado em items do QListWidget pra guardar a chave canônica
    # (inglês). O label visível pode ser PT-BR (built-in) ou igual à
    # chave (custom).
    COLUMN_KEY_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
    COLUMN_TYPE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Criador de Template de Estrutura")
        self.resize(620, 640)

        self._init_ui()

    # ─────────────────────────── UI build ────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Cabeçalho ──
        top_form = QFormLayout()
        self.txt_name = QLineEdit(self)
        self.txt_name.setPlaceholderText("Ex: BFW_Oficial")
        top_form.addRow("Nome do Template:", self.txt_name)
        layout.addLayout(top_form)

        # ── Tabs ──
        self.tabs = QTabWidget(self)

        # Aba 1: Colunas (declara o universo)
        self.tab_columns = QWidget(self)
        self._init_columns_tab()
        self.tabs.addTab(self.tab_columns, "🏷️ Colunas")

        # Aba 2: Pastas
        self.tab_folders = QWidget(self)
        self._init_folders_tab()
        self.tabs.addTab(self.tab_folders, "📁 Estrutura de Pastas")

        # Aba 3: Arquivos
        self.tab_files = QWidget(self)
        self._init_files_tab()
        self.tabs.addTab(self.tab_files, "📄 Nome do Arquivo")

        layout.addWidget(self.tabs)

        # Atalho identidade
        self.chk_parent = QCheckBox(
            "Usar nome da pasta pai imediata como Identidade",
            parent=self,
        )
        layout.addWidget(self.chk_parent)

        # ── Pré-visualização ──
        self._init_preview_section(layout)

        # Botões
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _init_columns_tab(self) -> None:
        layout = QVBoxLayout(self.tab_columns)
        layout.addWidget(QLabel(
            "Colunas disponíveis para mapear nas abas seguintes. "
            "Built-ins (🔒) vão direto para FaceInfo. "
            "Customs são gravadas em <code>extra_data</code>.",
            parent=self.tab_columns,
        ))

        self.list_columns = QListWidget(self.tab_columns)
        layout.addWidget(self.list_columns)

        controls = QHBoxLayout()
        self.btn_add_column = QPushButton("➕ Adicionar Coluna Custom", self.tab_columns)
        self.btn_remove_column = QPushButton("🗑️ Remover", self.tab_columns)
        self.btn_remove_column.setEnabled(False)
        controls.addWidget(self.btn_add_column)
        controls.addWidget(self.btn_remove_column)
        controls.addStretch()
        layout.addLayout(controls)

        # Habilita "Remover" apenas quando a seleção é uma coluna custom
        self.list_columns.itemSelectionChanged.connect(self._update_remove_column_state)

        # Popula built-ins de cara
        self._populate_builtins()

    def _populate_builtins(self) -> None:
        for key, label_pt in BUILTIN_COLUMNS.items():
            item = QListWidgetItem(f"🔒 {label_pt}  ({key})")
            item.setData(self.COLUMN_KEY_ROLE, key)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.list_columns.addItem(item)

    def _init_preview_section(self, layout: QVBoxLayout) -> None:
        """Seção de pré-visualização: o usuário digita um caminho de
        exemplo (relativo à raiz do dataset) e vê, em tempo real, o
        valor que cada coluna receberia — usando a MESMA lógica de
        extração da ingestão real."""
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        layout.addWidget(QLabel(
            "<b>🔍 Pré-visualização</b> — caminho de exemplo "
            "(relativo à pasta do dataset):",
            parent=self,
        ))

        self.txt_preview_path = QLineEdit(self)
        self.txt_preview_path.setPlaceholderText(
            "Ex: African/Aaron_Lewis_0001.jpg"
        )
        layout.addWidget(self.txt_preview_path)

        self.tbl_preview = QTableWidget(0, 2, self)
        self.tbl_preview.setHorizontalHeaderLabels(["Coluna", "Valor extraído"])
        self.tbl_preview.verticalHeader().setVisible(False)
        self.tbl_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_preview.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.tbl_preview.setMaximumHeight(150)
        header = self.tbl_preview.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tbl_preview)

    def set_preview_rows(self, rows: list) -> None:
        """Preenche a tabela de preview. `rows` é uma lista de
        (label:str, valor:str)."""
        self.tbl_preview.setRowCount(len(rows))
        for r, (label, value) in enumerate(rows):
            self.tbl_preview.setItem(r, 0, QTableWidgetItem(str(label)))
            self.tbl_preview.setItem(r, 1, QTableWidgetItem(str(value)))

    def _init_folders_tab(self) -> None:
        layout = QVBoxLayout(self.tab_folders)

        # Separador local
        sep_row = QHBoxLayout()
        self.chk_use_folder_sep = QCheckBox(
            "Usar separador no nome da pasta", parent=self.tab_folders,
        )
        self.txt_folder_separator = QLineEdit("_", parent=self.tab_folders)
        self.txt_folder_separator.setFixedWidth(40)
        self.txt_folder_separator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.txt_folder_separator.setEnabled(False)
        self.chk_use_folder_sep.toggled.connect(self.txt_folder_separator.setEnabled)
        sep_row.addWidget(self.chk_use_folder_sep)
        sep_row.addWidget(self.txt_folder_separator)
        sep_row.addStretch()
        layout.addLayout(sep_row)

        # Divisão só no último separador (rsplit 1×). Útil quando o
        # separador também aparece dentro do valor.
        self.chk_folder_last_sep = QCheckBox(
            "Usar apenas o ÚLTIMO separador como divisor", parent=self.tab_folders,
        )
        self.chk_folder_last_sep.setToolTip(
            "Divide o nome da pasta só na última ocorrência do separador "
            "(2 partes). Use quando o separador também faz parte do valor."
        )
        self.chk_folder_last_sep.setEnabled(False)
        self.chk_use_folder_sep.toggled.connect(self.chk_folder_last_sep.setEnabled)
        layout.addWidget(self.chk_folder_last_sep)

        layout.addWidget(QLabel("Mapeamento de Pastas (Defina o Nível):",
                                parent=self.tab_folders))

        self.list_folders = QListWidget(self.tab_folders)
        layout.addWidget(self.list_folders)

        controls = QHBoxLayout()
        self.spin_folder_level = QSpinBox(self.tab_folders)
        self.spin_folder_level.setRange(0, 10)
        self.spin_folder_level.setToolTip("Nível da pasta (0=Raiz do Dataset)")
        self.spin_folder_level.setPrefix("Nível ")

        self.cmb_folder_attr = QComboBox(self.tab_folders)

        self.btn_add_f = QPushButton("Adicionar", self.tab_folders)
        self.btn_rem_f = QPushButton("Remover", self.tab_folders)

        controls.addWidget(self.spin_folder_level)
        controls.addWidget(self.cmb_folder_attr, 1)
        controls.addWidget(self.btn_add_f)
        controls.addWidget(self.btn_rem_f)
        layout.addLayout(controls)

    def _init_files_tab(self) -> None:
        layout = QVBoxLayout(self.tab_files)

        # Separador local
        sep_row = QHBoxLayout()
        self.chk_use_filename_sep = QCheckBox(
            "Usar separador no nome do arquivo", parent=self.tab_files,
        )
        self.txt_filename_separator = QLineEdit("_", parent=self.tab_files)
        self.txt_filename_separator.setFixedWidth(40)
        self.txt_filename_separator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.txt_filename_separator.setEnabled(False)
        self.chk_use_filename_sep.toggled.connect(self.txt_filename_separator.setEnabled)
        sep_row.addWidget(self.chk_use_filename_sep)
        sep_row.addWidget(self.txt_filename_separator)
        sep_row.addStretch()
        layout.addLayout(sep_row)

        # Divisão só no último separador (rsplit 1×). Caso RFW:
        # "Aaron_Lewis_0001" -> ["Aaron_Lewis", "0001"].
        self.chk_filename_last_sep = QCheckBox(
            "Usar apenas o ÚLTIMO separador como divisor", parent=self.tab_files,
        )
        self.chk_filename_last_sep.setToolTip(
            "Divide o nome do arquivo só na última ocorrência do separador "
            "(2 partes). Ex.: 'Aaron_Lewis_0001' vira 'Aaron_Lewis' (Parte 0) "
            "+ '0001' (Parte 1). Use quando o separador também faz parte da "
            "identidade."
        )
        self.chk_filename_last_sep.setEnabled(False)
        self.chk_use_filename_sep.toggled.connect(self.chk_filename_last_sep.setEnabled)
        layout.addWidget(self.chk_filename_last_sep)

        layout.addWidget(QLabel("Mapeamento de Arquivo (Split):",
                                parent=self.tab_files))

        self.list_files = QListWidget(self.tab_files)
        layout.addWidget(self.list_files)

        controls = QHBoxLayout()
        self.spin_file_idx = QSpinBox(self.tab_files)
        self.spin_file_idx.setRange(0, 10)
        self.spin_file_idx.setPrefix("Parte ")

        self.cmb_file_attr = QComboBox(self.tab_files)

        self.btn_add_file = QPushButton("Adicionar", self.tab_files)
        self.btn_rem_file = QPushButton("Remover", self.tab_files)

        controls.addWidget(self.spin_file_idx)
        controls.addWidget(self.cmb_file_attr, 1)
        controls.addWidget(self.btn_add_file)
        controls.addWidget(self.btn_rem_file)
        layout.addLayout(controls)

    # ─────────────────────── API consumida pelo controller ──────────

    def selected_column_key(self) -> str:
        """Chave canônica da coluna selecionada na aba Colunas (ou '' se vazio)."""
        items = self.list_columns.selectedItems()
        if not items:
            return ""
        return items[0].data(self.COLUMN_KEY_ROLE) or ""

    def add_custom_column_item(self, key: str, dtype: str = "str") -> None:
        """Adiciona uma custom column visualmente (sem 🔒).

        O tipo aparece ao lado do nome porque é escolha que não dá para
        inferir olhando a lista depois, e errá-la só se descobre quando a
        coluna chega ao dataset como texto.
        """
        item = QListWidgetItem(f"📝 {key}  ·  {TIPOS_DE_COLUNA.get(dtype, dtype)}")
        item.setData(self.COLUMN_KEY_ROLE, key)
        item.setData(self.COLUMN_TYPE_ROLE, dtype)
        self.list_columns.addItem(item)

    def get_custom_column_types(self) -> dict:
        """`{nome: tipo}` das colunas custom, na ordem em que aparecem."""
        tipos = {}
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            key = item.data(self.COLUMN_KEY_ROLE)
            if key and key not in BUILTIN_COLUMNS:
                tipos[key] = item.data(self.COLUMN_TYPE_ROLE) or "str"
        return tipos

    def remove_selected_custom_column(self) -> None:
        items = self.list_columns.selectedItems()
        if not items:
            return
        for item in items:
            row = self.list_columns.row(item)
            self.list_columns.takeItem(row)

    def get_custom_column_keys(self) -> list:
        """Retorna na ordem em que aparecem (built-ins excluídos)."""
        keys = []
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            key = item.data(self.COLUMN_KEY_ROLE)
            if key and key not in BUILTIN_COLUMNS:
                keys.append(key)
        return keys

    def populate_attribute_combos(self, column_keys: list) -> None:
        """Repopula os combos de atributo das abas 2 e 3 a partir do
        universo conhecido. Sempre adiciona IGNORE_TOKEN ao final."""
        for combo in (self.cmb_folder_attr, self.cmb_file_attr):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for key in column_keys:
                label = BUILTIN_COLUMNS.get(key, key)
                combo.addItem(label, userData=key)
            combo.addItem("Ignorar", userData=IGNORE_TOKEN)
            # Restaura seleção se a chave ainda existe
            if current:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def add_folder_item(self, level: int, attr_key: str) -> None:
        """Adiciona linha visual com texto canônico (compat parser)."""
        self.list_folders.addItem(f"[Nível {level}] -> {attr_key}")

    def remove_folder_item(self) -> None:
        curr = self.list_folders.currentRow()
        if curr >= 0:
            self.list_folders.takeItem(curr)

    def add_file_item(self, idx: int, attr_key: str) -> None:
        self.list_files.addItem(f"[Parte {idx}] -> {attr_key}")

    def remove_file_item(self) -> None:
        curr = self.list_files.currentRow()
        if curr >= 0:
            self.list_files.takeItem(curr)

    # ────────────────────── internal slots ──────────────────────────

    def _update_remove_column_state(self) -> None:
        key = self.selected_column_key()
        is_custom = bool(key) and key not in BUILTIN_COLUMNS
        self.btn_remove_column.setEnabled(is_custom)
