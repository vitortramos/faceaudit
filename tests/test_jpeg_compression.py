"""Qualidade JPEG estimada da tabela de quantização.

O motivo de existir está num número: gravando o MESMO ajuste de qualidade
85, uma imagem lisa rende 0,26 bits/pixel e uma ruidosa rende 6,07 — 23
vezes mais. Comparar duas imagens diferentes por bits/pixel diz qual tinha
mais detalhe, não qual foi mais comprimida.

A tabela de quantização não depende do conteúdo: ela é escolhida pelo
codificador a partir do fator de qualidade pedido e fica gravada no arquivo.
Invertendo a fórmula do IJG recupera-se esse fator.

Os testes gravam com qualidade CONHECIDA e exigem que a estimativa a
recupere — em conteúdo liso e em conteúdo ruidoso, para provar a
independência de conteúdo que é a razão da medida existir.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from faceaudit.services.jpeg_compression_service import (  # noqa: E402
    build_compression_columns,
    probe_jpeg_compression,
)

ALTURA, LARGURA = 240, 320
_TOLERANCIA = 2.0        # a inversão é exata até o arredondamento da tabela


def _liso() -> np.ndarray:
    faixa = np.tile(np.linspace(0, 255, LARGURA, dtype=np.uint8), (ALTURA, 1))
    return np.dstack([faixa] * 3)


def _ruidoso() -> np.ndarray:
    return np.random.default_rng(11).integers(
        0, 256, (ALTURA, LARGURA, 3), dtype=np.uint8,
    )


def _gravar(tmp_path, arr, q, **kwargs):
    destino = tmp_path / f"q{q}_{kwargs.get('subsampling', 'd')}.jpg"
    Image.fromarray(arr).save(destino, quality=q, **kwargs)
    return destino


# ── Recupera a qualidade conhecida ───────────────────────────────────────

@pytest.mark.parametrize("q", [30, 40, 50, 60, 75, 85, 90, 95])
def test_recupera_a_qualidade_em_conteudo_liso(tmp_path, q):
    info = probe_jpeg_compression(_gravar(tmp_path, _liso(), q))

    assert info.tem_estimativa
    assert abs(info.quality_estimate - q) <= _TOLERANCIA, (
        f"gravado em {q}, estimado {info.quality_estimate}"
    )


@pytest.mark.parametrize("q", [30, 40, 50, 60, 75, 85, 90, 95])
def test_recupera_a_qualidade_em_conteudo_ruidoso(tmp_path, q):
    info = probe_jpeg_compression(_gravar(tmp_path, _ruidoso(), q))

    assert abs(info.quality_estimate - q) <= _TOLERANCIA


def test_a_estimativa_nao_depende_do_conteudo(tmp_path):
    """O ponto da medida. Mesma qualidade, conteúdos opostos: bits/pixel
    difere por ordens de grandeza, a estimativa não."""
    liso = _gravar(tmp_path / "a", _liso(), 85) if (tmp_path / "a").mkdir(
        exist_ok=True,
    ) is None else None
    ruidoso = _gravar(tmp_path, _ruidoso(), 85)

    q_liso = probe_jpeg_compression(liso).quality_estimate
    q_ruidoso = probe_jpeg_compression(ruidoso).quality_estimate

    bpp_liso = liso.stat().st_size * 8 / (ALTURA * LARGURA)
    bpp_ruidoso = ruidoso.stat().st_size * 8 / (ALTURA * LARGURA)

    assert abs(q_liso - q_ruidoso) <= 0.5, (
        "a estimativa tem de ser praticamente a mesma"
    )
    assert bpp_ruidoso > bpp_liso * 5, (
        "e bits/pixel tem de divergir muito — é essa divergência que torna "
        "bits/pixel enganoso como indicador de qualidade"
    )


def test_qualidade_maior_da_estimativa_maior(tmp_path):
    baixa = probe_jpeg_compression(_gravar(tmp_path, _ruidoso(), 40))
    alta = probe_jpeg_compression(_gravar(tmp_path, _ruidoso(), 90))

    assert alta.quality_estimate > baixa.quality_estimate


def test_regravar_substitui_a_tabela(tmp_path):
    """É a medida da ÚLTIMA gravação: reabrir e salvar em outra qualidade
    troca a tabela, e a estimativa acompanha."""
    original = _gravar(tmp_path, _ruidoso(), 95)
    regravado = tmp_path / "regravado.jpg"
    with Image.open(original) as img:
        img.save(regravado, quality=45)

    assert abs(probe_jpeg_compression(original).quality_estimate - 95) <= _TOLERANCIA
    assert abs(probe_jpeg_compression(regravado).quality_estimate - 45) <= _TOLERANCIA


# ── Limites declarados ───────────────────────────────────────────────────

def test_qualidade_no_fundo_da_escala_e_sinalizada(tmp_path):
    """Coeficiente saturado em 255 perde resolução na inversão — quem lê
    precisa saber, senão o número passa por exato."""
    info = probe_jpeg_compression(_gravar(tmp_path, _ruidoso(), 1))

    assert info.saturated is True
    assert info.quality_estimate is not None, "mesmo saturada, estima"


def test_qualidade_alta_nao_e_sinalizada(tmp_path):
    assert probe_jpeg_compression(_gravar(tmp_path, _ruidoso(), 90)).saturated is False


def test_estimativa_fica_na_escala_do_jpeg(tmp_path):
    for q in (1, 50, 100):
        valor = probe_jpeg_compression(
            _gravar(tmp_path, _ruidoso(), q),
        ).quality_estimate
        assert 1.0 <= valor <= 100.0


# ── Formato sem perdas ───────────────────────────────────────────────────

def test_png_nao_recebe_estimativa(tmp_path):
    """Não há tabela de quantização; devolver um número seria inventar."""
    destino = tmp_path / "x.png"
    Image.fromarray(_liso()).save(destino)

    info = probe_jpeg_compression(destino)

    assert info.tem_estimativa is False
    assert build_compression_columns(destino) == {}


def test_arquivo_inexistente_nao_estoura(tmp_path):
    info = probe_jpeg_compression(tmp_path / "nao_existe.jpg")

    assert info.error is not None
    assert info.tem_estimativa is False


def test_arquivo_corrompido_nao_estoura(tmp_path):
    destino = tmp_path / "corrompido.jpg"
    destino.write_bytes(b"isto nao e um jpeg")

    assert probe_jpeg_compression(destino).tem_estimativa is False


# ── Subamostragem de croma ───────────────────────────────────────────────

@pytest.mark.parametrize("sub,esperado", [(0, "4:4:4"), (1, "4:2:2"), (2, "4:2:0")])
def test_subamostragem_detectada(tmp_path, sub, esperado):
    destino = tmp_path / f"sub{sub}.jpg"
    Image.fromarray(_ruidoso()).save(destino, quality=85, subsampling=sub)

    assert probe_jpeg_compression(destino).chroma_subsampling == esperado


def test_progressivo_detectado(tmp_path):
    seq = tmp_path / "seq.jpg"
    prog = tmp_path / "prog.jpg"
    Image.fromarray(_ruidoso()).save(seq, quality=85)
    Image.fromarray(_ruidoso()).save(prog, quality=85, progressive=True)

    assert probe_jpeg_compression(seq).progressive is False
    assert probe_jpeg_compression(prog).progressive is True


# ── Colunas ──────────────────────────────────────────────────────────────

def test_colunas_do_jpeg(tmp_path):
    colunas = build_compression_columns(_gravar(tmp_path, _ruidoso(), 78))

    assert abs(colunas["jpeg_quality_estimate"] - 78) <= _TOLERANCIA
    assert colunas["jpeg_chroma_subsampling"] == "4:2:0"
    assert colunas["jpeg_quality_is_saturated"] is False
    assert colunas["jpeg_progressive"] is False


def test_colunas_estao_no_catalogo():
    """Sem spec, a coluna nasce como CUSTOM e a interface não consegue nem
    togglar a visibilidade dela."""
    from faceaudit.models.column_catalog import DatasetType, get_default_specs

    specs = get_default_specs(DatasetType.FACE)
    for nome in (
        "jpeg_quality_estimate", "jpeg_quality_is_saturated",
        "jpeg_chroma_subsampling", "jpeg_progressive",
    ):
        assert nome in specs, nome
        assert specs[nome].visible_default is False


def test_estimativa_nao_e_declarada_pelo_arquivo():
    """`is_declared` marca o que o arquivo AFIRMA. A estimativa é conta
    nossa sobre a tabela, não uma etiqueta — e confundir as duas num laudo
    seria atribuir ao arquivo uma afirmação que ele não fez."""
    from faceaudit.models.column_catalog import DatasetType, get_default_specs

    spec = get_default_specs(DatasetType.FACE)["jpeg_quality_estimate"]

    assert spec.is_declared is False
    assert spec.source.name == "DERIVED"
    assert "ESTIMATIVA" in spec.description
