"""Qualidade da última gravação JPEG, lida da tabela de quantização.

**Por que não bastam os bits por pixel.** Eles dependem do CONTEÚDO: uma
imagem lisa gravada em qualidade 85 pode render 0,26 bits/pixel (91:1), que
"parece" compressão brutal, enquanto uma foto detalhada na mesma qualidade
rende 2 bits/pixel (12:1). Comparar duas imagens diferentes por bits/pixel
não diz qual foi mais comprimida — diz qual tinha mais detalhe.

**A tabela de quantização não depende do conteúdo.** Ela é escolhida pelo
codificador a partir do fator de qualidade pedido, e fica gravada no
arquivo. Invertendo a fórmula do IJG (a biblioteca de referência que quase
todo codificador segue) recupera-se esse fator. É a medida da ÚLTIMA
gravação: reabrir e salvar de novo substitui a tabela.

**Limites, que a coluna declara.** O valor é uma ESTIMATIVA. Codificador
que usa tabelas próprias (alguns celulares, Photoshop na escala 0–12,
encoders de vídeo) não casa exatamente com a escala do IJG, e aí o número
indica a ordem de grandeza, não o ajuste literal. Coeficiente saturado em
255 (qualidade muito baixa) perde resolução na inversão. Nada disso é
motivo para omitir a medida — é motivo para ela não ser lida como se fosse
o parâmetro exato que alguém digitou.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

# Tabelas de quantização padrão do Anexo K do JPEG (as que o IJG escala).
# Ordem em zigue-zague, como aparecem no arquivo.
_PADRAO_LUMINANCIA: Tuple[int, ...] = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)
_PADRAO_CROMINANCIA: Tuple[int, ...] = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)


@dataclass(frozen=True)
class JpegCompression:
    """O que se consegue dizer sobre a última gravação de um JPEG."""

    quality_estimate: Optional[float] = None
    # Quantas tabelas o arquivo traz: 1 = luminância só (tons de cinza, ou
    # croma compartilhado), 2+ = luminância e crominância separadas.
    n_quant_tables: int = 0
    chroma_subsampling: Optional[str] = None
    progressive: Optional[bool] = None
    # Tabela saturada em 255 indica qualidade no fundo da escala, onde a
    # inversão perde resolução — a estimativa fica menos precisa e quem lê
    # precisa saber.
    saturated: bool = False
    error: Optional[str] = None

    @property
    def tem_estimativa(self) -> bool:
        return self.quality_estimate is not None


def probe_jpeg_compression(image_path) -> JpegCompression:
    """Lê a tabela de quantização e estima a qualidade. Nunca levanta.

    Só faz sentido em JPEG: formato sem perdas não tem tabela de
    quantização, e devolver um número ali seria inventar.
    """
    caminho = Path(str(image_path))
    try:
        from PIL import Image
    except Exception as e:                           # noqa: BLE001
        return JpegCompression(error=f"Pillow indisponível: {e}")

    try:
        with Image.open(caminho) as img:
            if (img.format or "").upper() not in {"JPEG", "MPO"}:
                return JpegCompression()
            tabelas = getattr(img, "quantization", None) or {}
            subamostragem = _subamostragem(img)
            progressivo = "progression" in getattr(img, "info", {})
    except Exception as e:                           # noqa: BLE001
        return JpegCompression(error=str(e))

    if not tabelas:
        return JpegCompression(
            chroma_subsampling=subamostragem, progressive=progressivo,
            error="arquivo sem tabela de quantização",
        )

    qualidade, saturada = _estimar_qualidade(tabelas)
    return JpegCompression(
        quality_estimate=qualidade,
        n_quant_tables=len(tabelas),
        chroma_subsampling=subamostragem,
        progressive=progressivo,
        saturated=saturada,
    )


def build_compression_columns(image_path) -> Dict[str, Any]:
    """Colunas de compressão JPEG. Vazio quando não é JPEG."""
    info = probe_jpeg_compression(image_path)
    colunas: Dict[str, Any] = {}
    if info.quality_estimate is not None:
        colunas["jpeg_quality_estimate"] = round(info.quality_estimate, 1)
        colunas["jpeg_quality_is_saturated"] = info.saturated
    if info.chroma_subsampling:
        colunas["jpeg_chroma_subsampling"] = info.chroma_subsampling
    if info.progressive is not None and info.n_quant_tables:
        colunas["jpeg_progressive"] = bool(info.progressive)
    return colunas


# ── Inversão da fórmula do IJG ───────────────────────────────────────────

def _estimar_qualidade(tabelas: Dict[int, Sequence[int]]) -> Tuple[Optional[float], bool]:
    """Fator de qualidade médio implicado pelas tabelas presentes.

    O IJG monta a tabela como
        `fator = 5000/q` (q < 50) ou `200 − 2q` (q ≥ 50)
        `tabela[i] = limita((padrao[i] × fator + 50) / 100, 1, 255)`
    Invertendo coeficiente por coeficiente e tirando a média dos fatores
    recupera-se `q`. Média, e não um coeficiente único, porque arredondamento
    e tabelas não-padrão desviam cada coeficiente de um jeito diferente.
    """
    fatores: list = []
    fatores_saturados: list = []
    saturada = False

    for indice, tabela in sorted(tabelas.items()):
        padrao = _PADRAO_LUMINANCIA if indice == 0 else _PADRAO_CROMINANCIA
        valores = list(tabela)[:64]
        if len(valores) < 64:
            continue
        for valor, referencia in zip(valores, padrao):
            try:
                v = float(valor)
            except (TypeError, ValueError):
                continue
            if v <= 0 or referencia <= 0:
                continue
            fator = (v * 100.0 - 50.0) / float(referencia)
            if v >= 255:
                # Saturado: a inversão não distingue onde o fator parou, só
                # que passou do teto. Fica de reserva.
                saturada = True
                fatores_saturados.append(fator)
            else:
                fatores.append(fator)

    if not fatores:
        # TODOS saturados — acontece no fundo da escala (qualidade ~1), onde
        # a tabela inteira bate no teto. Usar os saturados dá um piso em vez
        # de nada: saber que a qualidade está no fundo é informação, e
        # devolver vazio aqui apagaria justamente o caso mais extremo.
        fatores = fatores_saturados

    if not fatores:
        return None, saturada

    fator = sum(fatores) / len(fatores)
    if fator <= 0:
        return 100.0, saturada
    qualidade = (5000.0 / fator) if fator > 100.0 else ((200.0 - fator) / 2.0)
    # A escala do JPEG é 1..100; sair dela significa tabela não-padrão, e aí
    # o extremo é a informação honesta.
    return max(1.0, min(100.0, qualidade)), saturada


def _subamostragem(img) -> Optional[str]:
    """"4:4:4", "4:2:2", "4:2:0" — ou None quando não dá para afirmar.

    Quanto de crominância foi descartado é informação de procedência: foto
    de câmera costuma vir 4:2:0, e um reencode em 4:4:4 denuncia passagem
    por outro programa.
    """
    try:
        from PIL import JpegImagePlugin

        layout = JpegImagePlugin.get_sampling(img)
    except Exception:                                # noqa: BLE001
        return None
    return {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(layout)
