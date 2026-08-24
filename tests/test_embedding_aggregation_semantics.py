"""Semântica da agregação de embeddings (`apply_smart_aggregation`).

A agregação tinha ZERO cobertura de teste, e é justamente ela que produz
templates sintéticos consumidos por toda a infra de pareamento. Estes testes
travam as seis correções e, principalmente, os invariantes que as motivaram:

- `face_hash` é a CHAVE DE PAR do motor (`_align_pairs`,
  `_extract_canonical_rarity`, ScoreVectorPairCompatibilityRule, exclusão de
  auto-comparação no PAIR_MINING) e vai pro disco como coluna obrigatória nos
  parquets de score. Precisa ser determinístico entre execuções e distinto de
  qualquer hash de foto real.
- Um template é a média de N faces: não é imagem, não tem pose, e "qualidade
  de imagem" não é definida pra ele. Metadado de foto não sobrevive à fusão.
- Identidade é o que define o agrupamento — sem ela, agregar não significa
  nada e a operação aborta.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.aggregation_config import AggregationConfig
from faceaudit.services.embedding_math_service import EmbeddingMathService

_MUDO = lambda _m: None  # noqa: E731 — silencia o log da agregação nos testes


def _face(i: int, identity: str = "A", **kw) -> EmbeddingInfo:
    base = dict(
        embedding=np.random.RandomState(i).rand(8),
        face_hash=f"real{i:03d}",
        image_path=f"/fotos/{i}.jpg",
        identity=identity,
        file_name=f"{i}.jpg",
        source="src",
        origin="orig",
        yaw=float(i),
        pitch=1.0,
        quality_scores={"fiqa": 50.0 + i},
    )
    base.update(kw)
    return EmbeddingInfo(**base)


def _dataset(n=4, identity="A", **kw) -> EmbeddingDataset:
    return EmbeddingDataset(name="base", items=[_face(i, identity) for i in range(n)], **kw)


def _agregar(ds, **cfg) -> EmbeddingDataset:
    return EmbeddingMathService().apply_smart_aggregation(
        ds, AggregationConfig(**cfg), logger_callback=_MUDO,
    )


def _combinatorial(ds, k=2) -> EmbeddingDataset:
    return _agregar(ds, method=AggregationMethod.COMBINATORIAL, combination_group_size=k)


# ───────────────────── face_hash: identidade da linha ─────────────────────

def test_hash_e_deterministico_para_a_mesma_entrada():
    """`hash()` embutido é salgado por processo — produzia identificador novo
    a cada execução e quebrava o pareamento entre runs."""
    h1 = [i.face_hash for i in _combinatorial(_dataset()).items]
    h2 = [i.face_hash for i in _combinatorial(_dataset()).items]
    assert h1 == h2
    assert all(h.startswith("SYNTH_") for h in h1)


def test_hash_e_deterministico_ENTRE_PROCESSOS():
    """O teste que realmente pega o bug original.

    `hash()` de string é salgado por processo (PYTHONHASHSEED) desde a 3.3 —
    dentro de UMA execução ele é estável, então um teste in-process passa
    mesmo com o bug presente. A quebra só aparece entre execuções, que é
    exatamente o cenário real: reexecutar a estratégia gerava identificadores
    novos para os mesmos templates e o pareamento com artefatos já em disco
    falhava como "N pares não encontrados".

    Roda a MESMA agregação em dois subprocessos com sementes de hash
    diferentes e exige o mesmo resultado.
    """
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import numpy as np
        from faceaudit.enums.aggregation_method import AggregationMethod
        from faceaudit.models import EmbeddingDataset, EmbeddingInfo
        from faceaudit.models.aggregation_config import AggregationConfig
        from faceaudit.services.embedding_math_service import EmbeddingMathService

        itens = [
            EmbeddingInfo(embedding=np.random.RandomState(i).rand(8),
                          face_hash=f"real{i:03d}", image_path="", identity="A",
                          file_name=f"{i}.jpg")
            for i in range(4)
        ]
        out = EmbeddingMathService().apply_smart_aggregation(
            EmbeddingDataset(name="base", items=itens),
            AggregationConfig(method=AggregationMethod.COMBINATORIAL,
                              combination_group_size=2),
            logger_callback=lambda m: None,
        )
        print(",".join(i.face_hash for i in out.items))
    """)

    def _rodar(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
        saida = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env,
        )
        assert saida.returncode == 0, saida.stderr[-2000:]
        return saida.stdout.strip().splitlines()[-1]

    assert _rodar("0") == _rodar("12345")


def test_hash_nao_colide_com_nenhuma_foto_real():
    """O defeito da agregação legada: template herdava o hash de uma das fotos
    e se passava por imagem real na detecção de vazamento."""
    ds = _dataset()
    reais = {i.face_hash for i in ds.items}
    sinteticos = {i.face_hash for i in _combinatorial(ds).items}
    assert reais.isdisjoint(sinteticos)


def test_hash_invariante_a_ordem_das_faces_no_dataset():
    """Mesmo conjunto de faces fundidas → mesmo template, venham na ordem que
    vierem. Sem isso, reordenar o dataset invalidaria o pareamento."""
    ds = _dataset()
    invertido = EmbeddingDataset(name="base", items=list(reversed(ds.items)))
    assert sorted(i.face_hash for i in _agregar(ds).items) == \
           sorted(i.face_hash for i in _agregar(invertido).items)


def test_hash_muda_quando_o_conjunto_de_faces_muda():
    """Contraprova do teste anterior: o hash tem que DEPENDER dos contribuintes,
    senão dois templates diferentes colidiriam."""
    a = _agregar(_dataset(n=3)).items[0].face_hash
    b = _agregar(_dataset(n=4)).items[0].face_hash
    assert a != b


def test_combinatorial_nao_gera_hash_nem_nome_duplicado():
    """Bug de colisão: `new_fname` derivava só de items[indices[0]].file_name,
    então todo combo com o mesmo primeiro índice virava o MESMO nome e o mesmo
    hash. Com k=2 sobre 4 faces, (0,1) (0,2) (0,3) colidiam. Duplicata aqui faz
    `_extract_canonical_rarity` abortar a estratégia lá na frente."""
    out = _combinatorial(_dataset(n=4), k=2)
    assert len(out.items) == 6                      # C(4,2)
    assert len({i.face_hash for i in out.items}) == 6
    assert len({i.file_name for i in out.items}) == 6


# ───────────── metadado de foto não sobrevive à agregação ─────────────

@pytest.mark.parametrize("method", [AggregationMethod.CENTROID, AggregationMethod.COMBINATORIAL])
def test_template_nao_carrega_metadado_de_foto(method):
    out = _agregar(_dataset(), method=method, combination_group_size=2)
    for t in out.items:
        assert t.image_path == "", "template não é imagem nenhuma"
        assert t.pose == {}, "pose de uma média de faces não é definida"
        assert t.quality_scores == {}, "qualidade de IMAGEM não se aplica a template"
        assert t.face_image is None
        assert t.yaw is None and t.pitch is None   # properties leem o dict vazio


def test_origem_sobrevive_se_unanime_e_some_se_divergir():
    """`source`/`origin` descrevem a coleção, não a foto — sobrevivem enquanto
    forem os mesmos. Divergiu, vira ausente em vez de virar 'a primeira'."""
    unanime = _agregar(_dataset()).items[0]
    assert unanime.source == "src" and unanime.origin == "orig"

    misto = EmbeddingDataset(name="b", items=[
        _face(0, "A", origin="lote1"), _face(1, "A", origin="lote2"),
    ])
    assert _agregar(misto).items[0].origin is None


# ───────────────────────── identidade obrigatória ─────────────────────────

@pytest.mark.parametrize("ruim", [None, "", "   "])
def test_identidade_ausente_aborta(ruim):
    """Antes, faces sem identidade eram fundidas num balde 'unknown' — pessoas
    diferentes viravam um template só, e 'unknown' passava a contar como
    identidade real em n_ids, MIN_IDENTITIES, FisherRule e no PLDA."""
    ds = EmbeddingDataset(name="b", items=[_face(0, "A"), _face(1, ruim)])
    with pytest.raises(ValueError, match="identidade"):
        _agregar(ds)


def test_mensagem_de_erro_diz_quantas_e_quais():
    ds = EmbeddingDataset(name="b", items=[_face(0, "A"), _face(1, None), _face(2, None)])
    with pytest.raises(ValueError) as exc:
        _agregar(ds)
    assert "2 de 3" in str(exc.value)


# ──────────────────── proveniência no loaded_metadata ────────────────────

def test_metadata_do_dataset_e_propagado_e_anotado():
    """Split e merge propagam `loaded_metadata` em toda saída; a agregação
    descartava. E a anotação é o que permite distinguir 'campo de foto vazio
    por design' de 'vazio por falha de ingestão'."""
    ds = _dataset(loaded_metadata={"origem_do_dataset": "teste"})
    meta = _agregar(ds).loaded_metadata

    assert meta["origem_do_dataset"] == "teste", "metadado de entrada preservado"
    assert meta["aggregation_applied"] == "true"
    assert meta["aggregation_method"] == "centroid"
    assert meta["aggregation_source_items"] == "4"
    assert meta["aggregation_output_templates"] == "1"
    assert meta["aggregation_identities"] == "1"
    assert "image_path" in meta["aggregation_cleared_photo_fields"]
    assert meta["aggregation_hash_scheme"]


def test_metadata_registra_parametros_da_combinatoria():
    meta = _combinatorial(_dataset(), k=2).loaded_metadata
    assert meta["aggregation_group_size"] == "2"
    assert "aggregation_max_per_identity" in meta
    # CENTROID não tem k — a chave não deve aparecer inventada.
    assert "aggregation_group_size" not in _agregar(_dataset()).loaded_metadata


# ───────────────────────── matemática preservada ─────────────────────────

def test_centroid_funde_identidade_em_um_template_normalizado():
    ds = EmbeddingDataset(name="b", items=[_face(i, "A") for i in range(3)]
                                          + [_face(i, "B") for i in range(3, 6)])
    out = _agregar(ds)
    assert len(out.items) == 2
    assert {i.identity for i in out.items} == {"A", "B"}
    for t in out.items:
        np.testing.assert_allclose(np.linalg.norm(t.embedding), 1.0, rtol=1e-5)


def test_ponderacao_por_qualidade_continua_lendo_a_qualidade_da_ENTRADA():
    """A qualidade some da SAÍDA, mas a ponderação usa a da ENTRADA — limpar o
    output não pode ter quebrado o peso. Com pesos extremos, o template tem que
    pender para a face de maior qualidade."""
    ds = EmbeddingDataset(name="b", items=[
        _face(0, "A", embedding=np.array([1.0, 0.0]), quality_scores={"fiqa": 1.0}),
        _face(1, "A", embedding=np.array([0.0, 1.0]), quality_scores={"fiqa": 999.0}),
    ])
    ponderado = _agregar(ds, use_quality_weighting=True,
                         quality_weight_type=QualityScoreType.FIQA).items[0].embedding
    simples = _agregar(ds).items[0].embedding

    assert ponderado[1] > ponderado[0], "deve pender para a face de maior FIQA"
    np.testing.assert_allclose(simples, [2 ** -0.5, 2 ** -0.5], rtol=1e-5)


def test_combinatorial_respeita_o_teto_por_identidade():
    out = _combinatorial(_dataset(n=6), k=2)          # C(6,2) = 15
    assert len(out.items) == 15
    limitado = EmbeddingMathService().apply_smart_aggregation(
        _dataset(n=6),
        AggregationConfig(method=AggregationMethod.COMBINATORIAL,
                          combination_group_size=2, max_synthetic_templates_per_id=4),
        logger_callback=_MUDO,
    )
    assert len(limitado.items) == 4


def test_combinatorial_com_poucas_faces_cai_em_centroide():
    out = _combinatorial(_dataset(n=2), k=5)
    assert len(out.items) == 1


# ───────── regressões encontradas na revisão adversarial ─────────

def test_hash_nao_colapsa_quando_face_hash_e_vazio():
    """`face_hash=""` é estado NORMAL: `to_embedding_dataset` faz
    `face_hash=face.hash or ""` e `from_parquet` faz `hash_value or ''` para
    qualquer parquet sem a coluna. Derivar o hash do template só dos hashes
    contribuintes fazia TODO k-combo da identidade gerar payload byte-idêntico
    → 10 templates com 10 embeddings distintos colapsavam em 1 chave de par, e
    o PAIR_MINING descartaria todos os genuínos como auto-comparação."""
    ds = EmbeddingDataset(name="b", items=[
        EmbeddingInfo(embedding=np.random.RandomState(i).rand(8), face_hash="",
                      image_path=f"/p/{i}.jpg", identity="A", file_name=f"{i}.jpg")
        for i in range(5)
    ])
    out = _combinatorial(ds, k=3)
    assert len(out.items) == 10                              # C(5,3)
    assert len({t.face_hash for t in out.items}) == 10
    assert len({t.file_name for t in out.items}) == 10


def test_hash_vazio_continua_invariante_a_ordem():
    """O fallback de conteúdo não pode depender da POSIÇÃO no dataset."""
    itens = [
        EmbeddingInfo(embedding=np.random.RandomState(i).rand(8), face_hash="",
                      image_path=f"/p/{i}.jpg", identity="A", file_name=f"{i}.jpg")
        for i in range(4)
    ]
    direto = _agregar(EmbeddingDataset(name="b", items=list(itens)))
    invertido = _agregar(EmbeddingDataset(name="b", items=list(reversed(itens))))
    assert {t.face_hash for t in direto.items} == {t.face_hash for t in invertido.items}


def test_faces_byte_identicas_abortam():
    """Duas linhas com o mesmo hash e embeddings diferentes é corrupção
    silenciosa a jusante. Duplicata real de arquivo é problema de DADO — falha
    alto, no mesmo espírito do gate de identidade."""
    def _dup(nome):
        return EmbeddingInfo(embedding=np.array([0.0, 1.0, 0.0, 0.0]), face_hash="B",
                             image_path="/b.jpg", identity="ID", file_name=nome)
    ds = EmbeddingDataset(name="b", items=[
        EmbeddingInfo(embedding=np.array([1.0, 0.0, 0.0, 0.0]), face_hash="A",
                      image_path="/a.jpg", identity="ID", file_name="a.jpg"),
        _dup("b.jpg"), _dup("b.jpg"),
    ])
    with pytest.raises(ValueError, match="repetido"):
        _combinatorial(ds, k=2)


def test_subgrupo_com_pesos_zerados_nao_estoura():
    """A guarda de soma-zero era do GRUPO, mas o fatiamento é por SUBGRUPO:
    um combo cujas k faces têm qualidade 0.0 somava zero mesmo com o grupo
    somando positivo, e `np.average` levantava ZeroDivisionError."""
    ds = EmbeddingDataset(name="b", items=[
        _face(i, "A", quality_scores={"ofiq": v})
        for i, v in enumerate([0.0, 0.0, 0.0, 70.0, 80.0])
    ])
    out = EmbeddingMathService().apply_smart_aggregation(
        ds,
        AggregationConfig(method=AggregationMethod.COMBINATORIAL,
                          combination_group_size=3, use_quality_weighting=True,
                          quality_weight_type=QualityScoreType.OFIQ),
        logger_callback=_MUDO,
    )
    assert len(out.items) == 10
    assert all(np.all(np.isfinite(t.embedding)) for t in out.items)


def test_qualidade_nan_nao_produz_template_nan():
    """NaN atravessava as três guardas: não é None, `sum == 0` é False para
    NaN, e `norm > 1e-9` é False — o template saía todo NaN, sem erro."""
    ds = EmbeddingDataset(name="b", items=[
        _face(0, "A", quality_scores={"fiqa": float("nan")}),
        _face(1, "A", quality_scores={"fiqa": 0.5}),
    ])
    t = _agregar(ds, use_quality_weighting=True,
                 quality_weight_type=QualityScoreType.FIQA).items[0]
    assert np.all(np.isfinite(t.embedding))


def test_combinatoria_respeita_o_teto_sem_materializar_o_espaco():
    """C(60,5) = 5.461.512 tuplas (~480 MB) eram construídas só para descartar
    tudo menos 100. Com 100 faces e k=5 são 75 milhões — o processo morria."""
    import time

    ds = EmbeddingDataset(name="b", items=[
        EmbeddingInfo(embedding=np.random.RandomState(i).rand(8),
                      face_hash=f"h{i:03d}", image_path="", identity="A",
                      file_name=f"{i}.jpg")
        for i in range(60)
    ])
    t0 = time.perf_counter()
    out = EmbeddingMathService().apply_smart_aggregation(
        ds,
        AggregationConfig(method=AggregationMethod.COMBINATORIAL,
                          combination_group_size=5,
                          max_synthetic_templates_per_id=100),
        logger_callback=_MUDO,
    )
    elapsed = time.perf_counter() - t0

    assert len(out.items) == 100
    assert len({t.face_hash for t in out.items}) == 100
    assert elapsed < 5.0, f"materializou o espaço combinatório ({elapsed:.1f}s)"


def test_selecao_da_combinatoria_e_reprodutivel():
    ds = _dataset(n=12)
    a = sorted(t.face_hash for t in _combinatorial(ds, k=3).items)
    b = sorted(t.face_hash for t in _combinatorial(ds, k=3).items)
    assert a == b


def test_proveniencia_nao_declara_ponderacao_que_nao_ocorreu():
    """Métrica ausente em todas as faces → pesos unitários → média simples.
    Gravar "ponderada" na proveniência de um dataset que vai a laudo é falso."""
    ds = EmbeddingDataset(name="b", items=[
        _face(0, "A", quality_scores={}), _face(1, "A", quality_scores={}),
    ])
    meta = _agregar(ds, use_quality_weighting=True,
                    quality_weight_type=QualityScoreType.FIQA).loaded_metadata

    assert meta["aggregation_quality_weighting"] == ""
    assert meta["aggregation_quality_weighting_requested"] == "fiqa"
    assert meta["aggregation_quality_weighting_effective"] == "false"

    # E o caminho feliz continua declarando corretamente.
    meta_ok = _agregar(_dataset(), use_quality_weighting=True,
                       quality_weight_type=QualityScoreType.FIQA).loaded_metadata
    assert meta_ok["aggregation_quality_weighting_effective"] == "true"


def test_metrica_nula_com_ponderacao_ligada_e_corrigida_na_origem():
    """Estado alcançável por JSON editado à mão. O campo None não renderiza no
    editor inline (despacha por tipo do valor) e o executor recusa executar —
    beco sem saída. O DTO corrige ao construir."""
    cfg = AggregationConfig(use_quality_weighting=True, quality_weight_type=None)
    assert cfg.quality_weight_type is QualityScoreType.FIQA
    assert AggregationConfig.from_dict(
        {"use_quality_weighting": True, "quality_weight_type": None}
    ).quality_weight_type is QualityScoreType.FIQA


# ───────────────────────── método legado removido ─────────────────────────

def test_agregacao_legada_nao_existe_mais():
    """`apply_identity_aggregation` tinha zero chamadores, estourava TypeError
    se chamada (construía EmbeddingInfo sem `image_path`) e herdava o hash de
    uma foto real — a convenção errada ao lado da certa."""
    assert not hasattr(EmbeddingMathService, "apply_identity_aggregation")
