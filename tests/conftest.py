"""Ajustes de ambiente compartilhados pela suíte de testes.

Força a plataforma Qt **offscreen** quando nenhuma foi definida — permite
rodar os testes de UI (pytest-qt) em máquinas headless / CI sem abrir
janelas. `setdefault` respeita uma escolha explícita do desenvolvedor
(ex: quem quiser depurar visualmente exporta `QT_QPA_PLATFORM` antes).

Import-time only; não cria QApplication (pytest-qt cuida disso via
`qtbot`/`qapp`). Inócuo para os testes de backend que não tocam em Qt.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
