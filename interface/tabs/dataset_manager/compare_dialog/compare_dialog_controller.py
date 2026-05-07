from PyQt6 import QtCore, QtWidgets
from typing import Optional

# --- SERVICES & MODELS ---
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.dataset_comparison_service import DatasetComparisonService
from faceaudit.services.dataset_stats_service import DatasetStatsService
from faceaudit.models.comparison_result import ComparisonResult

# --- VIEW ---
from .compare_dialog_view import CompareDialogView

class CompareDialogController(QtCore.QObject):
    """
    Controlador do Diálogo de Comparação.
    
    Responsabilidade:
    1. Receber os Datasets A e B (já em memória).
    2. Usar DatasetStatsService para extrair os metadados (compute_from_dataset).
    3. Usar DatasetComparisonService para gerar o diff.
    4. Atualizar a View.
    """

    # Sem @inject! As dependências são exigidas explicitamente no contrato.
    def __init__(self, 
                 dataset_a: FaceDataset, 
                 dataset_b: FaceDataset, 
                 stats_service: DatasetStatsService,
                 comparison_service: DatasetComparisonService,
                 parent_view: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent_view) # Boa prática do PyQt
        
        self.ds_a = dataset_a
        self.ds_b = dataset_b
        
        # --- RECEBIMENTO DOS SERVIÇOS ---
        self.stats_service = stats_service
        self.comparison_service = comparison_service

        # Inicializa a View
        self.view = CompareDialogView(parent_view)
        
        name_a = self.ds_a.name if self.ds_a.name else "Dataset A"
        name_b = self.ds_b.name if self.ds_b.name else "Dataset B"
        self.view.setWindowTitle(f"Comparação: {name_a} vs {name_b}")
        
        # Carrega os dados imediatamente
        self._load_and_show()

    def exec(self):
        """Exibe o diálogo modalmente."""
        return self.view.exec()

    def _load_and_show(self):
        """
        Orquestra o fluxo: Dataset -> Stats -> ComparisonResult -> View
        """
        # 1. Extrai estatísticas (Sem I/O de disco, usa Pandas na RAM)
        stats_a = self.stats_service.compute_from_dataset(self.ds_a)
        stats_b = self.stats_service.compute_from_dataset(self.ds_b)

        name_a = self.ds_a.name if self.ds_a.name else "Dataset A"
        name_b = self.ds_b.name if self.ds_b.name else "Dataset B"

        # 2. Gera o relatório de comparação
        result: ComparisonResult = self.comparison_service.compare_stats(
            stats_a, stats_b, name_a, name_b
        )

        # 3. Atualiza a interface
        self._populate_view(result)

    def _populate_view(self, result: ComparisonResult):
        # A. Resumo
        self.view.set_summary_data(
            total_a=result.total_imgs_a,
            total_b=result.total_imgs_b,
            ids_a=result.total_ids_a,
            ids_b=result.total_ids_b,
            unique_ids_a=result.unique_ids_a_count,
            unique_ids_b=result.unique_ids_b_count,
            common_ids=result.common_ids_count,
            common_cols=result.common_columns,
            missing_cols=result.missing_in_b
        )

        # B. Detalhes (Abas extras)
        # Remove abas antigas se houver (mantém só a primeira de Resumo)
        while self.view.tabs.count() > 1:
            self.view.tabs.removeTab(1)

        # PR 22 (Tema 4.5) — render genérico a partir de `category_diffs`.
        # Itera as chaves do dict (ordenadas pra estabilidade visual:
        # canônicas demográficas conhecidas primeiro, depois custom columns
        # em ordem alfabética) e cria uma aba por categoria. Fallback aos
        # campos legados (`gender_diffs`/`ethnicity_diffs`/`origin_diffs`)
        # quando `category_diffs` não está populado — preserva
        # comportamento pra ComparisonResult criados manualmente em testes
        # ou código antigo.
        category_diffs = getattr(result, "category_diffs", None) or {}
        if category_diffs:
            for col_name in self._sort_categorical_columns(category_diffs.keys()):
                label = self._label_for(col_name, default=col_name)
                self._add_detail_tab(label, category_diffs[col_name])
        else:
            # Fallback retrocompat
            gender_label = self._label_for("gender", default="Gênero")
            ethnicity_label = self._label_for("ethnicity", default="Etnia")
            origin_label = self._label_for("origin", default="Origem")
            self._add_detail_tab(gender_label, result.gender_diffs)
            self._add_detail_tab(ethnicity_label, result.ethnicity_diffs)
            self._add_detail_tab(origin_label, result.origin_diffs)

    # Ordem canônica para as abas demográficas — mantém estabilidade visual
    # entre execuções e prioriza as colunas mais comuns no topo.
    _CANONICAL_CATEGORY_ORDER = ("gender", "ethnicity", "origin", "source")

    def _sort_categorical_columns(self, columns):
        """Canônicas primeiro (na ordem fixa), depois custom em ordem alfabética."""
        cols = list(columns)
        canonical = [c for c in self._CANONICAL_CATEGORY_ORDER if c in cols]
        rest = sorted(c for c in cols if c not in self._CANONICAL_CATEGORY_ORDER)
        return canonical + rest

    def _label_for(self, column_name: str, default: str) -> str:
        """Resolve o label de uma coluna via column_specs do dataset A.

        Procura no dataset A primeiro (referência); se não encontrar, tenta
        no B; se nada, retorna o default. PR 22: também aceita labels de
        colunas CUSTOM (qualquer categórica de baixa cardinalidade vira aba).
        """
        for ds in (self.ds_a, self.ds_b):
            specs = getattr(ds, "column_specs", None) or {}
            spec = specs.get(column_name)
            if spec is not None:
                label = getattr(spec, "label", None)
                if label:
                    return label
        return default

    def _add_detail_tab(self, title, diff_list):
        if not diff_list: return
        self.view.add_comparison_tab(title, diff_list)