# interface/tabs/strategy_manager_v2/comparison/__init__.py

"""
Pacote da janela de comparação entre estratégias treinadas.

Componentes:
- `comparison_window`: janela top-level (QDialog não-modal).
- `comparison_left_panel`: painel esquerdo com (kind selector + items tree).
- `comparison_kind`: enum dos 3 tipos de comparação suportados.
- `availability_resolver`: descobre quais (estratégia, recurso) têm o dado.
- `views/`: widgets de conteúdo, um por kind.
"""
