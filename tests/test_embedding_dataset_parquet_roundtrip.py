"""Round-trip do `EmbeddingDataset` em parquet: metadado e pose.

Duas assimetrias de longa data entre escrita e leitura:

1. `from_parquet` SEMPRE leu toda a metadata do schema para `loaded_metadata`,
   mas `to_parquet` nunca a gravou. Metadado de dataset — proveniência de
   agregação, aviso de viés de seleção, origem — vivia só em memória e
   evaporava no primeiro save.
2. `from_parquet` aceitava colunas `yaw`/`pitch` e, quando ausentes,
   inventava `0.0`. Como `to_parquet` não gravava pose, TODO round-trip
   trocava a pose real por zeros — e 0° é pose fisicamente válida, então o
   dado inventado era indistinguível de medição. A `HasPoseMetadataRule` lia
   cobertura 1.0 e liberava o HUBNESS_ABLATION BY_POSE sobre zeros falsos.
"""

from __future__ import annotations

import numpy as np

from faceaudit.models import EmbeddingDataset, EmbeddingInfo


def _face(i: int, **kw) -> EmbeddingInfo:
    base = dict(
        embedding=np.random.RandomState(i).rand(8), face_hash=f"h{i:03d}",
        image_path=f"/fotos/{i}.jpg", identity="A", file_name=f"{i}.jpg",
    )
    base.update(kw)
    return EmbeddingInfo(**base)


def _ida_e_volta(ds: EmbeddingDataset, tmp_path, nome="ds") -> EmbeddingDataset:
    caminho = tmp_path / f"{nome}.parquet"
    ds.to_parquet(caminho)
    return EmbeddingDataset.from_parquet(caminho)


# ───────────────────────── loaded_metadata ─────────────────────────

def test_loaded_metadata_sobrevive_ao_parquet(tmp_path):
    ds = EmbeddingDataset(
        name="b", items=[_face(0)],
        loaded_metadata={"origem": "teste", "aviso_de_vies": "seleção condicionada"},
    )
    meta = _ida_e_volta(ds, tmp_path).loaded_metadata
    assert meta["origem"] == "teste"
    assert meta["aviso_de_vies"] == "seleção condicionada"


def test_loaded_metadata_sobrevive_a_dois_ciclos(tmp_path):
    """O segundo save relê a metadata (que agora inclui a chave reservada
    `pandas`) e a reescreve — não pode corromper o schema."""
    ds = EmbeddingDataset(name="b", items=[_face(0)], loaded_metadata={"origem": "teste"})
    volta = _ida_e_volta(_ida_e_volta(ds, tmp_path, "a"), tmp_path, "b")
    assert volta.loaded_metadata["origem"] == "teste"
    assert len(volta.items) == 1


def test_extra_metadata_vence_o_loaded_metadata(tmp_path):
    """`extra_metadata` é override explícito no momento da chamada."""
    ds = EmbeddingDataset(name="b", items=[_face(0)], loaded_metadata={"k": "do_dataset"})
    caminho = tmp_path / "ds.parquet"
    ds.to_parquet(caminho, extra_metadata={"k": "do_caller"})
    assert EmbeddingDataset.from_parquet(caminho).loaded_metadata["k"] == "do_caller"


# ─────────────────────────────── pose ───────────────────────────────

def test_pose_real_sobrevive_ao_parquet(tmp_path):
    """Antes se perdia no save e voltava como 0.0."""
    ds = EmbeddingDataset(name="b", items=[
        _face(0, yaw=15.5, pitch=-3.25), _face(1, yaw=-40.0, pitch=8.0),
    ])
    volta = _ida_e_volta(ds, tmp_path)
    assert [(i.yaw, i.pitch) for i in volta.items] == [(15.5, -3.25), (-40.0, 8.0)]


def test_pose_ausente_nao_vira_zero(tmp_path):
    """AUSENTE ≠ ZERO: 0° é pose válida (rosto frontal), então inventar zero
    torna o dado fabricado indistinguível de medição real."""
    volta = _ida_e_volta(EmbeddingDataset(name="b", items=[_face(0)]), tmp_path)
    assert volta.items[0].pose == {}
    assert volta.items[0].yaw is None and volta.items[0].pitch is None


def test_pose_zero_legitima_e_preservada_como_zero(tmp_path):
    """Contraprova: quem TEM pose 0.0 medida continua com 0.0, não vira None."""
    volta = _ida_e_volta(
        EmbeddingDataset(name="b", items=[_face(0, yaw=0.0, pitch=0.0)]), tmp_path,
    )
    assert volta.items[0].yaw == 0.0 and volta.items[0].pitch == 0.0


def test_pose_parcial_preserva_so_o_eixo_presente(tmp_path):
    volta = _ida_e_volta(
        EmbeddingDataset(name="b", items=[_face(0, yaw=12.0)]), tmp_path,
    )
    assert volta.items[0].yaw == 12.0
    assert volta.items[0].pitch is None


# ────────────────── interação com dataset agregado ──────────────────

def test_dataset_agregado_mantem_campos_de_foto_vazios_apos_o_save(tmp_path):
    """A decisão de que um template não carrega metadado de foto precisa
    sobreviver ao disco — senão o `from_parquet` a desfazia inventando 0.0."""
    from faceaudit.models.aggregation_config import AggregationConfig
    from faceaudit.services.embedding_math_service import EmbeddingMathService

    entrada = EmbeddingDataset(name="b", items=[
        _face(i, yaw=10.0 + i, pitch=2.0, quality_scores={"fiqa": 60.0}) for i in range(3)
    ])
    agregado = EmbeddingMathService().apply_smart_aggregation(
        entrada, AggregationConfig(), logger_callback=lambda _m: None,
    )
    volta = _ida_e_volta(agregado, tmp_path, "agg")

    t = volta.items[0]
    assert t.image_path == "" and t.pose == {} and t.quality_scores == {}
    assert volta.loaded_metadata["aggregation_applied"] == "true"
    assert volta.loaded_metadata["aggregation_method"] == "centroid"
