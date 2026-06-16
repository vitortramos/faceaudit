"""Demonstração executável do adaface_embedding_simulator 0.3.0.

Este script roda fim-a-fim usando APENAS a API pública do módulo
``adaface_embedding_simulator`` e depende somente de ``numpy`` + ``pandas``.
O FaceAudit NÃO é necessário: nada de ``faceaudit_seed`` (Modo B) nem do
adapter concreto é importado. A ancoragem em "centros reais" é demonstrada
pelo caminho de injeção do CORE — ``sample_identity_population(centers=...)``
— com uma pequena matriz sintética de centros, que é totalmente livre de
FaceAudit.

Quatro blocos independentes, cada um numa função própria:

  1. ``demo_normal_split``        — split cooperativo padrão (perfil
     COOPERATIVE_PLDA_BASELINE) via ``make_cooperative_plda_config`` +
     ``sample_dataset`` + ``save_bundle`` (NPZ + metadata CSV + manifesto +
     diagnósticos de norma).
  2. ``demo_idperturb_cap``       — perfil IDPERTURB_CAP via
     ``make_idperturb_cap_config``; confere que as colunas novas
     ``cap_lower_bound_effective`` e ``pre_nuisance_center_cosine`` existem e
     respeitam o invariante ``pre_nuisance_center_cosine >= cap_lower_bound_effective``.
  3. ``demo_config_round_trip``   — round-trip de configuração
     ``save_manifest`` -> ``config_from_manifest``, provando que a config
     desserializada reproduz o mesmo objeto validado.
  4. ``demo_inject_real_centers`` — ancoragem em centros injetados via
     ``sample_identity_population(centers=..., identity_labels=...,
     source_sample_counts=...)`` + ``sample_observations``, mostrando as
     colunas de proveniência ``source_identity`` e ``source_sample_count``.

ATENÇÃO FORENSE: este simulador NÃO implementa a rede AdaFace. Ele cria um
mundo sintético controlado. Para qualquer afirmação científica, calibre os
parâmetros contra embeddings reais representativos e reporte qual perfil foi
usado. A ancoragem em centros reais (bloco 4 / Modo A) gera variação
sintética em torno de âncoras biométricas REAIS — o conjunto-âncora deve
permanecer disjunto de qualquer conjunto de avaliação.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    config_from_manifest,
    make_cooperative_plda_config,
    make_idperturb_cap_config,
    sample_pair_table,
)


# ---------------------------------------------------------------------------
# Bloco 1 — split normal cooperativo + persistência de bundle
# ---------------------------------------------------------------------------
def demo_normal_split(output_dir: Path) -> None:
    """Gera train/dev/test com identidades disjuntas e salva os bundles."""
    print("\n=== Bloco 1: split cooperativo padrao ===")

    # O perfil COOPERATIVE_PLDA_BASELINE usa o modo de dispersao intra-classe
    # VMF (default), que reproduz a saida 0.2 BIT-A-BIT (travado por golden test).
    simulator = AdaFaceEmbeddingSimulator(
        make_cooperative_plda_config(dimension=128),
        random_seed=2026,
    )

    # Cada split recebe um identity_offset distinto para garantir conjuntos de
    # identidades DISJUNTOS, e seeds explicitas (populacao + observacao) para
    # reprodutibilidade independente da ordem de chamada.
    train = simulator.sample_dataset(
        identity_count=300,
        samples_per_identity=(5, 12),
        split_name="train",
        identity_offset=0,
        random_seed=101,
        population_random_seed=1001,
    )
    dev = simulator.sample_dataset(
        identity_count=120,
        samples_per_identity=(4, 10),
        split_name="dev",
        identity_offset=100_000,
        random_seed=102,
        population_random_seed=1002,
    )
    test = simulator.sample_dataset(
        identity_count=200,
        samples_per_identity=(4, 10),
        split_name="test",
        identity_offset=200_000,
        random_seed=103,
        population_random_seed=1003,
    )

    # save_bundle escreve NPZ + metadata CSV + manifesto JSON + diagnosticos de
    # norma, todos derivados do mesmo dataset.
    for dataset, split_name in [(train, "train"), (dev, "dev"), (test, "test")]:
        paths = dataset.save_bundle(output_dir, split_name)
        print(f"  {split_name}: {dataset.raw_embeddings.shape[0]} obs -> {paths['npz'].name}")

    # Tabelas de pares balanceados por identidade, com scores de cosseno.
    sample_pair_table(
        dev,
        genuine_pair_count=2_000,
        random_impostor_pair_count=2_000,
        random_seed=201,
    ).to_csv(output_dir / "dev_pairs.csv", index=False)
    sample_pair_table(
        test,
        genuine_pair_count=5_000,
        random_impostor_pair_count=5_000,
        random_seed=202,
    ).to_csv(output_dir / "test_pairs.csv", index=False)

    print("  Diagnostico de norma (test, U-invertido esperado):")
    print(test.norm_bin_diagnostics(bin_count=5).to_string(index=False))


# ---------------------------------------------------------------------------
# Bloco 2 — perfil IDPERTURB_CAP (cap de cosseno uniforme)
# ---------------------------------------------------------------------------
def demo_idperturb_cap(output_dir: Path) -> None:
    """Demonstra o cap de cosseno (IDperturb) e as colunas de metadata novas."""
    print("\n=== Bloco 2: perfil IDPERTURB_CAP ===")

    # make_idperturb_cap_config troca o modo intra-classe para IDPERTURB_CAP:
    # cada amostra desenha s ~ U[lb, 1] (UNIFORME EM COSSENO, nao uniforme na
    # area do cap em alta dimensao); E[cos] = (1 + lb) / 2; lb = 1 => sem
    # perturbacao. O overlap guard (bissetor) impoe um piso por identidade e
    # TEM PRIORIDADE, entao o lb EFETIVO pode exceder o nominal configurado.
    simulator = AdaFaceEmbeddingSimulator(
        make_idperturb_cap_config(dimension=128, lower_bound=0.6),
        random_seed=4242,
    )
    dataset = simulator.sample_dataset(
        identity_count=150,
        samples_per_identity=(6, 12),
        split_name="cap_demo",
        identity_offset=0,
        random_seed=303,
        population_random_seed=3003,
    )
    dataset.save_bundle(output_dir, "cap_demo")

    metadata = dataset.metadata

    # As colunas novas da 0.3 precisam existir no modo cap.
    for column in ("intra_class_sampling_mode", "cap_lower_bound_effective",
                   "pre_nuisance_center_cosine", "hybrid_resample_count",
                   "hybrid_used_cap_fallback"):
        assert column in metadata.columns, f"coluna ausente: {column}"

    # No modo cap, cap_lower_bound_effective NAO e NaN (no modo VMF seria).
    lb_effective = metadata["cap_lower_bound_effective"].to_numpy()
    pre_cosine = metadata["pre_nuisance_center_cosine"].to_numpy()
    assert np.all(np.isfinite(lb_effective)), "cap_lower_bound_effective deveria ser finito no modo cap"

    # Invariante chave da 0.3: o cosseno PRE-ruido respeita o piso efetivo.
    assert np.all(pre_cosine >= lb_effective - 1e-9), (
        "violado: pre_nuisance_center_cosine >= cap_lower_bound_effective"
    )

    # O lb efetivo pode passar do cap_maximum_lower_bound porque o overlap guard
    # tem prioridade. Sempre reporte percentis do lb EFETIVO, nao do nominal.
    print(f"  modo intra-classe: {metadata['intra_class_sampling_mode'].iloc[0]}")
    print(f"  lb efetivo  p5/p50/p95: "
          f"{np.percentile(lb_effective, [5, 50, 95]).round(4).tolist()}")
    print(f"  cos pre-ruido p5/p50/p95: "
          f"{np.percentile(pre_cosine, [5, 50, 95]).round(4).tolist()}")
    print(f"  E[cos] empirico={pre_cosine.mean():.4f}  "
          f"(aprox (1 + E[lb])/2 = {(1 + lb_effective.mean()) / 2:.4f})")
    print("  invariante pre_nuisance_center_cosine >= cap_lower_bound_effective: OK")


# ---------------------------------------------------------------------------
# Bloco 3 — round-trip de configuracao via manifesto
# ---------------------------------------------------------------------------
def demo_config_round_trip(output_dir: Path) -> None:
    """save_manifest -> config_from_manifest reproduz a config validada."""
    print("\n=== Bloco 3: round-trip de configuracao ===")

    original_config = make_idperturb_cap_config(dimension=64, lower_bound=0.55)
    simulator = AdaFaceEmbeddingSimulator(original_config, random_seed=777)
    dataset = simulator.sample_dataset(
        identity_count=40,
        samples_per_identity=4,
        split_name="round_trip",
        random_seed=404,
        population_random_seed=4004,
    )

    # O manifesto carrega o bloco "config" serializado (enums como strings,
    # arrays como listas) junto de seeds e hash da fonte.
    manifest_path = dataset.save_manifest(output_dir / "round_trip_manifest.json")
    recovered_config = config_from_manifest(manifest_path)

    # A config recuperada e validada e bate nos campos chave.
    assert recovered_config.dimension == original_config.dimension
    assert recovered_config.profile_name == original_config.profile_name
    assert (
        recovered_config.directional.intra_class_sampling_mode
        == original_config.directional.intra_class_sampling_mode.value
    )
    assert np.isclose(
        recovered_config.directional.cap_base_lower_bound,
        original_config.directional.cap_base_lower_bound,
    )
    print(f"  manifesto: {manifest_path.name}")
    print(f"  perfil recuperado: {recovered_config.profile_name} "
          f"(dim={recovered_config.dimension})")
    print("  round-trip config_from_manifest: OK")


# ---------------------------------------------------------------------------
# Bloco 4 — ancoragem em centros injetados (Modo A, livre de FaceAudit)
# ---------------------------------------------------------------------------
def demo_inject_real_centers(output_dir: Path) -> None:
    """Ancora a populacao em centros 'reais' injetados via centers=.

    Em producao, esses centros viriam de identity_centers_from_dataset
    (faceaudit_seed, Modo A). Aqui, para manter o exemplo livre de FaceAudit,
    sintetizamos uma pequena matriz de centros aleatorios e a injetamos pelo
    MESMO caminho do CORE: sample_identity_population(centers=...).

    LEMBRETE FORENSE: o Modo A NAO e anonimizacao — ele gera variacao sintetica
    em torno de ancoras biometricas REAIS. O conjunto-ancora deve permanecer
    disjunto de qualquer conjunto de avaliacao (em producao, use
    faceaudit_seed.assert_disjoint_source_identities).
    """
    print("\n=== Bloco 4: ancoragem em centros injetados (Modo A) ===")

    dimension = 128
    n_anchor_identities = 25

    # Matriz de centros 'reais' sintetica (sera L2-normalizada internamente).
    rng = np.random.default_rng(909)
    fake_real_centers = rng.normal(size=(n_anchor_identities, dimension))
    identity_labels = [f"real_subject_{i:03d}" for i in range(n_anchor_identities)]
    # Quantas imagens reais teriam formado cada media esferica (proveniencia).
    source_sample_counts = rng.integers(3, 20, size=n_anchor_identities)

    simulator = AdaFaceEmbeddingSimulator(
        make_cooperative_plda_config(dimension=dimension),
        random_seed=5050,
    )

    # Caminho de injecao: passa-se centers= em vez de identity_count=. O sampler
    # de centros vMF e pulado (RNG nao consumido), preservando bit-a-bit o
    # restante do pipeline.
    population = simulator.sample_identity_population(
        centers=fake_real_centers,
        identity_labels=identity_labels,
        source_sample_counts=source_sample_counts,
        identity_offset=900_000,
        random_seed=5005,
        stream_name="anchored_population",
    )

    # Observa-se a populacao ancorada normalmente.
    dataset = simulator.sample_observations(
        population,
        samples_per_identity=(4, 8),
        split_name="anchored",
        random_seed=505,
    )
    dataset.save_bundle(output_dir, "anchored")

    metadata = dataset.metadata
    # Colunas de proveniencia presentes APENAS para populacoes ancoradas.
    assert "source_identity" in metadata.columns
    assert "source_sample_count" in metadata.columns
    assert set(metadata["source_identity"].unique()) <= set(identity_labels)

    print(f"  identidades ancoradas: {metadata['identity_id'].nunique()}")
    print(f"  observacoes geradas:   {len(metadata)}")
    print("  amostra de proveniencia (source_identity / source_sample_count):")
    preview = (
        metadata[["identity_id", "source_identity", "source_sample_count"]]
        .drop_duplicates()
        .head(5)
    )
    print(preview.to_string(index=False))


def main() -> None:
    output_dir = Path("synthetic_adaface_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_normal_split(output_dir)
    demo_idperturb_cap(output_dir)
    demo_config_round_trip(output_dir)
    demo_inject_real_centers(output_dir)

    print(f"\nTodos os artefatos foram escritos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
