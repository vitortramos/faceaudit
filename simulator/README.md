# AdaFace Embedding Simulator

Gerador sintético configurável de embeddings **AdaFace-like** para experimentos controlados com similaridade de cosseno, LDA, *whitening*, Gaussianização radial, PLDA, PSDA, calibração e tipicidade.

O pacote não implementa a rede neural AdaFace. Ele cria um mundo sintético auditável no qual é possível variar separadamente:

- geometria populacional esférica por mistura de vMF;
- dispersão intraidentidade;
- norma do embedding e caudas radiais baixa e alta;
- qualidade latente;
- estrutura anisotrópica global, por domínio e por identidade;
- viés médio de domínio, viés domínio × identidade e viés de sessão;
- acoplamentos entre qualidade, norma e direção;
- condições de aquisição alinhadas por identidade.

## Instalação

```bash
pip install numpy pandas
```

Para executar os testes:

```bash
pip install pytest
python -m pytest -q
```

## Exemplo rápido

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_cooperative_plda_config(dimension=512),
    random_seed=2026,
)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 10),
    split_name="test",
    identity_offset=200_000,
    random_seed=103,
    population_random_seed=1003,
)

test.save_bundle("synthetic_output", "test")
```

## Stress test cross-condition alinhado

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cross_condition_stress_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_cross_condition_stress_config(dimension=512),
    random_seed=3030,
)

aligned = simulator.sample_cross_condition_datasets(
    identity_count=800,
    identity_offset=300_000,
    split_name="cross_condition_test",
    samples_per_identity_by_domain={
        "controlled": (3, 6),
        "surveillance": (3, 6),
    },
    sessions_per_identity=(1, 3),
    population_random_seed=4001,
    observation_random_seeds={
        "controlled": 4002,
        "surveillance": 4003,
    },
)
```

## Documentação

Consulte [`DOCUMENTATION.md`](DOCUMENTATION.md) para a referência completa da API, fórmulas, parâmetros configuráveis, perfis prontos, exemplos de ablação, integração com o FaceAudit e roteiro de calibração contra embeddings reais.
