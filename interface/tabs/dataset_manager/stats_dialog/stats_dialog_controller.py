import pandas as pd
from PyQt6 import QtCore
from .stats_dialog_view import StatsDialogView

class StatsDialogController(QtCore.QObject):
    """
    Controlador das Estatísticas.
    Extrai dados do Dataset, calcula porcentagens e formata para a View.
    """
    def __init__(self, dataset, parent_view=None):
        super().__init__()
        self.dataset = dataset
        self.view = StatsDialogView(title_suffix=dataset.name, parent=parent_view)
        
        self._calculate_and_populate()

    def exec(self):
        """Exibe o diálogo modalmente."""
        return self.view.exec()

    def _calculate_and_populate(self):
        # Obtém estatísticas brutas do modelo (pandas series)
        # Lembre-se: O FaceDataset agora retorna {'identity':..., 'origin':...}
        stats = self.dataset.get_all_statistics()
        
        # Definição das abas a serem criadas
        # (Titulo da Aba, Chave no Dict de Stats, Icone)
        categories = [
            ("Identidades", 'identity', "👤"),
            ("Gênero", 'gender', "⚧️"),
            ("Etnia", 'ethnicity', "🌍"),
            ("Fonte", 'origin', "💾") 
        ]

        for title, key, icon in categories:
            # Agora ele vai encontrar os dados em 'origin'
            series = stats.get(key)
            self._process_category(title, icon, series)

    def _process_category(self, title, icon, series):
        """Processa uma Serie do Pandas e envia para a View criar a aba."""
        data_rows = []
        total_unique = 0

        if series is not None and not series.empty:
            total_unique = len(series)
            total_imgs = series.sum()

            for label, count in series.items():
                # Tratamento de dados
                label_str = str(label) if str(label) != 'nan' else "(Não Informado)"
                
                # Cálculo de Regra de Negócio
                pct = (count / total_imgs) * 100
                
                # Adiciona à lista de dados limpos
                data_rows.append((label_str, count, pct))
        
        # Envia dados prontos para a View desenhar
        self.view.add_statistics_tab(title, icon, total_unique, data_rows)