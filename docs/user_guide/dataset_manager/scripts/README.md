# Scripts de captura de telas — Gerenciador de Datasets

Este diretório contém o script usado para gerar os PNGs em `../assets/`
de forma reprodutível, em ambiente headless (sem display físico).

## Como funciona

`capture_dataset_manager.py` sobe o FaceAudit em modo headless via Xvfb,
carrega um parquet de teste, abre cada diálogo programaticamente, faz o
`QWidget.grab()` e salva como PNG. Faz monkey-patch em `QDialog.exec()`
para que diálogos modais não bloqueiem o loop de eventos.

A fonte aplicada é **Inter** (default 10pt) e a resolução do Xvfb é
**2560×1440**.

## Pré-requisitos

```bash
# Sistema
apt-get install -y xvfb fonts-inter

# Python: dependências do FaceAudit (PyQt6, pandas, etc.)
pip install -e ../../../..

# Dados de teste (NÃO commitados — contêm imagens reais):
#   - Um parquet .parquet (gerado em outra máquina com ForensicFace)
#   - A pasta de imagens correspondente
```

## Como rodar

```bash
export FACEAUDIT_TEST_PARQUET=/caminho/para/Dataset_teste.parquet
export FACEAUDIT_TEST_IMAGES=/caminho/para/teste_alinhado

xvfb-run -a -s "-screen 0 2560x1440x24" python capture_dataset_manager.py
```

PNGs serão escritos em `../assets/`.

## Diálogos de vídeo + Novo/Enriquecer (offscreen, Windows)

`capture_video_dialogs.py` gera **quatro** telas sem depender de Xvfb,
parquet de teste nem GPU — roda com o QPA `offscreen` (funciona no
Windows):

```bash
QT_QPA_PLATFORM=offscreen python capture_video_dialogs.py
```

Produz `44_video_setup.png`, `45_video_review.png` (a partir de uma pasta
de run **sintética** construída na hora) e regenera `30_novo_dataset.png`
e `32_enriquecer.png` com o layout redesenhado (o Enriquecer usa um
dataset sintético de cobertura mista para exibir os selos
presente/parcial/ausente).

> ⚠️ O QPA `offscreen` sobe com **0 fontes** — o script carrega TTFs do
> Windows (`segoeui.ttf` etc.) na base de fontes do Qt, senão o texto
> vira tofu (□□□). Em Linux/macOS, ajuste os caminhos de fonte em
> `_load_fonts()` ou rode via `xvfb-run` como o script principal.

## Stub do ForensicFace

A pasta `fake_forensicface/forensicface/` contém um stub mínimo da
biblioteca `forensicface` (que requer modelos ONNX pesados). Permite
que a UI inicialize, mas **não** executa inferência. Use-o **apenas**
para captura de telas; nunca para criar datasets reais.

## Observações

- **Comparar** (`fluxo_comparar`) não é capturado porque exige dois
  datasets carregados simultaneamente. Para incluí-lo na documentação,
  carregue dois parquets diferentes e adapte o script.
- **Mesclar** (`fluxo_merge`) abre um `QFileDialog` direto (sem diálogo
  próprio); o monkey-patch retorna string vazia e o fluxo aborta.
- A aba **Recomendações** do diálogo de Auditoria aparece vazia porque
  o worker assíncrono não roda durante a captura. Para captar com
  resultados, é necessário aguardar o worker — não implementado neste
  script.
