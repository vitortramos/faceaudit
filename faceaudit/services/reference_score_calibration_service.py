# faceaudit/services/reference_score_calibration_service.py

"""Calibração da distribuição de referência de scores (CDF + threshold OOD).

PR 4 (V2.15): `CROSS_FIT_IDENTITY` implementado para MoVMF e k-NN.
`IN_SAMPLE_LEGACY` permanece como caminho do k-NN nativo (delega o
LOO de imagem do fit). Revisão externa round 1 e round 2
(2026-06-09) corrigiu: instalação atômica da referência OOF no
modelo via `install_reference_score_calibration` (P0-1); ponderação
identitária do OOF (P0-4); base canônica por backend via
`reference_score_basis_type` (P0-R2-2); agregação por centroide em
IDENTITY_CENTROIDS (P0-R2-3); inspeção de seed por
`inspect.signature` (P1-R2-10).

A separação de responsabilidades segue a decisão do plano: modelo
matemático faz só o fit paramétrico (EM no MoVMF; armazenamento de
referência no k-NN). Serviço orquestra folds, scores OOF, CDF empírica,
threshold OOD, refit final em toda a população.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def _factory_accepts_seed(factory: Callable) -> bool:
    """True quando `factory` aceita o keyword argument `seed`.

    P1-R2-10 (revisão externa 2026-06-09 round 2): substitui o
    `try: factory(seed=...); except TypeError: factory()` que
    mascarava TypeErrors legítimos disparados pelo construtor
    interno do modelo.
    """
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        # built-in / C-extension sem assinatura introspectível
        return False
    params = sig.parameters
    if "seed" in params:
        return True
    # Aceita **kwargs?
    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _build_model(
    factory: Callable, *, seed: int, expected_backend=None,
):
    """Cria um modelo da factory propagando `seed` de forma uniforme.

    P0-R3-3 (revisão externa 2026-06-09 round 3): substitui as três
    formas distintas que existiam no serviço — `trainer_factory()`,
    `trainer_factory(seed=...)`, `try/except TypeError` — por um
    único ponto de entrada. Usado em peek, em cada fold, e no refit
    final. Garante que factories com `seed` obrigatório não quebrem
    o peek/refit (problema apontado pelo revisor: `peek_model =
    trainer_factory()` sem seed levanta `TypeError` quando a factory
    é `def trainer_factory(*, seed): ...`).

    P0-R3-2: também valida `model.model_type == expected_backend`
    quando `expected_backend` é fornecido, evitando que erro de
    wiring no registry produza um modelo silenciosamente do tipo
    errado.
    """
    if _factory_accepts_seed(factory):
        model = factory(seed=seed)
    else:
        model = factory()
        # Best-effort: configura random_state se o backend expõe.
        if hasattr(model, "random_state"):
            model.random_state = seed

    if expected_backend is not None:
        actual = getattr(model, "model_type", None)
        if actual != expected_backend:
            raise BackendMismatchError(
                expected=expected_backend, actual=actual,
            )
    return model


class BackendMismatchError(RuntimeError):
    """Trainer factory devolveu modelo de um backend distinto do
    declarado no parâmetro `backend` do serviço.

    P0-R3-2 (revisão externa 2026-06-09 round 3): fail-closed em
    casos de wiring incorreto do registry (caller declara MOVMF mas
    factory devolve k-NN, ou vice-versa). Antes, o serviço executava
    silenciosamente até o final usando o modelo errado.
    """

    def __init__(self, *, expected, actual) -> None:
        super().__init__(
            f"BackendMismatch: serviço chamado com backend="
            f"{getattr(expected, 'to_str', lambda: str(expected))()!r} "
            f"mas trainer_factory devolveu model.model_type="
            f"{getattr(actual, 'to_str', lambda: str(actual))()!r}. "
            "Verifique o registry de backends ou o config da instrução."
        )
        self.expected = expected
        self.actual = actual


class CrossFitContractMismatchError(RuntimeError):
    """Modelo construído pela factory em fold/refit difere semanticamente
    do peek_model em campos que afetam a interpretação dos scores OOF.

    P0-R5-5 (revisão externa 2026-06-09 round 5): fail-closed quando a
    factory devolve modelos com `reference_score_basis_type`,
    `population_weighting_mode`, `embedding_dim` ou `enabled_score_types`
    incompatíveis entre folds — antes, o serviço concatenava silenciosamente
    scores em escalas diferentes (ex.: fold ímpar em LOG_DENSITY, par em
    MEAN_KNN_SIMILARITY).
    """

    def __init__(self, *, field: str, peek_value, fold_value, fold_index=None) -> None:
        where = "refit" if fold_index is None else f"fold {fold_index}"
        super().__init__(
            f"CrossFitContractMismatch: campo {field!r} divergiu entre "
            f"peek_model e {where}. peek={peek_value!r}, "
            f"{where}={fold_value!r}. A factory deve produzir modelos "
            "semanticamente equivalentes em peek/folds/refit."
        )
        self.field = field
        self.peek_value = peek_value
        self.fold_value = fold_value
        self.fold_index = fold_index


def _contract_snapshot(model) -> dict:
    """Captura os campos que precisam ser estáveis entre peek/folds/refit
    para que os scores OOF sejam concatenáveis sem ambiguidade.

    Lê o valor atual em cada chamada — não fica em cache. `_validate_*`
    abaixo compara strings/ints (não enums brutos) para tolerar pequenas
    variações de identidade entre instâncias do mesmo enum (por exemplo
    quando o backend reconstrói).
    """
    def _normalize(v):
        if v is None:
            return None
        if hasattr(v, "to_str"):
            return v.to_str()
        if isinstance(v, frozenset):
            return tuple(sorted(_normalize(x) for x in v))
        if isinstance(v, (tuple, list, set)):
            return tuple(sorted(_normalize(x) for x in v))
        return str(v)

    def _safe_attr(name):
        # Algumas properties levantam quando model ainda não foi fitted
        # (`embedding_dim`, `enabled_score_types`). Tratamos como None
        # nesse caso — `_validate_contract_matches` tolera ausência.
        try:
            return getattr(model, name, None)
        except (RuntimeError, AttributeError):
            return None

    return {
        "model_type": _normalize(_safe_attr("model_type")),
        "reference_score_basis_type": _normalize(
            _safe_attr("reference_score_basis_type")
        ),
        "population_weighting_mode": _normalize(
            _safe_attr("population_weighting_mode")
        ),
        "embedding_dim": _safe_attr("embedding_dim"),
        "enabled_score_types": _normalize(_safe_attr("enabled_score_types")),
    }


def _validate_contract_matches(peek: dict, other: dict, *, fold_index=None) -> None:
    """Compara dois snapshots de contrato. Difere → CrossFitContractMismatchError.

    `embedding_dim` é tolerante a None (alguns backends só calculam após
    o fit); só compara se ambos têm valor.
    """
    for field in (
        "model_type",
        "reference_score_basis_type",
        "population_weighting_mode",
        "enabled_score_types",
    ):
        if peek.get(field) != other.get(field):
            raise CrossFitContractMismatchError(
                field=field,
                peek_value=peek.get(field),
                fold_value=other.get(field),
                fold_index=fold_index,
            )
    if peek.get("embedding_dim") is not None and other.get("embedding_dim") is not None:
        if peek["embedding_dim"] != other["embedding_dim"]:
            raise CrossFitContractMismatchError(
                field="embedding_dim",
                peek_value=peek["embedding_dim"],
                fold_value=other["embedding_dim"],
                fold_index=fold_index,
            )

from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.protocols.typicality_scoring_model_protocol import (
    FittableTypicalityModelProtocol,
)


def oof_quantile_in_range(q: float) -> bool:
    """0 < q < 0.5 — protege contra divisão por zero ou quantis acima da
    metade que não fazem sentido como threshold OOD inferior."""
    return q is not None and 0.0 < q < 0.5


def _safe_float(x) -> Optional[float]:
    """Converte para float; retorna None se input é None ou não-finito.

    R4: usado para coletar `ood_threshold_by_fold` de modelos transitórios
    cujos campos podem ser None (ood_quantile=0.0) ou ±inf.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


@dataclass(frozen=True)
class ReferenceScoreCalibrationResult:
    """Resultado da calibração de score de referência.

    P1-R3-9 (revisão externa 2026-06-09 round 3): nomeação do campo
    `oof_cdf` corrigida. O nome implicava "função de distribuição
    cumulativa" mas armazenava na verdade os **scores OOF ordenados**.
    Agora exposto via três campos separados:
    - `oof_reference_scores_sorted`: scores ordenados (era `oof_cdf`).
    - `oof_reference_weights_sorted`: pesos por score (mesma ordem).
    - `oof_reference_cdf`: cumsum(weights)/sum — a CDF de fato.

    `oof_cdf` continua como property deprecated por back-compat.

    Em IN_SAMPLE_LEGACY (k-NN nativo):
        - `refit_model`: k-NN com LOO de imagem.
        - oof_reference_*: todos None (não há OOF).

    Em CROSS_FIT_IDENTITY (MoVMF + k-NN):
        - `refit_model`: modelo treinado em toda a população — com
          a referência OOF JÁ INSTALADA atomicamente (P0-1).
        - `oof_reference_scores_sorted`: em IDENTITY_CENTROIDS é o
          score de cada centroide identitário; em
          INVERSE_IDENTITY_FREQUENCY é por face.
        - `oof_reference_weights_sorted`: pesos derivados do
          `population_weighting_mode`.
        - `oof_reference_cdf`: CDF empírica ponderada.
        - `ood_threshold`: quantil ponderado.
        - `split_fingerprint`: hash do mapeamento identidade → fold.
        - `fold_seeds`: lista de seeds por fold de fato propagadas
          ao trainer_factory.

    R4 (revisão externa 2026-06-09 round 4): adicionados os campos
    preditivos OOF, com semântica documentada em
    `docs/typicality_models.md` §5.2/§5.3:

        - `oof_predictive_avg_score`: média ponderada de TODOS os
          scores OOF (mesmos pesos da CDF). No MoVMF
          (basis=LOG_DENSITY_ESTIMATE) é a estimativa OOF de
          E[log p_θ(x)] — a log-likelihood preditiva OOF que o
          revisor recomenda como critério primário para selecionar K.
          No k-NN (basis=MEAN_KNN_SIMILARITY) é T_k médio populacional
          — métrica relativa entre k's, NÃO log-densidade.
        - `oof_predictive_avg_score_by_fold`: médias por fold.
          Dispersão entre folds é diagnóstico-chave de estabilidade.
        - `oof_predictive_score_type`: "LOG_DENSITY_ESTIMATE" ou
          "MEAN_KNN_SIMILARITY" — chave de interpretação.
        - `n_identities_by_fold`, `n_scores_by_fold`,
          `ood_threshold_by_fold`: diagnósticos por fold (cobertura
          e estabilidade do quantil).
    """

    refit_model: FittableTypicalityModelProtocol
    mode: ReferenceScoreCalibrationMode
    oof_reference_scores_sorted: Optional[np.ndarray] = None
    oof_reference_weights_sorted: Optional[np.ndarray] = None
    oof_reference_cdf: Optional[np.ndarray] = None
    # Métricas preditivas OOF (R4 — solicitação do revisor externo
    # 2026-06-09). Implementação do que estava implícito no procedimento
    # de cross-fit mas não exposto: log-likelihood preditiva OOF
    # agregada por fold e global, mais contagens por fold e thresholds
    # OOD intermediários para diagnóstico de estabilidade.
    #
    # Em MoVMF (basis=LOG_DENSITY_ESTIMATE), `oof_predictive_avg_score`
    # é diretamente o estimador OOF de E[log p_θ(x)] sobre a população
    # — quão bem o modelo atribui densidade a observações fora do
    # treino. É o **critério preditivo principal para seleção de K**
    # (P1-R5 — revisor 2026-06-09 round 5: PLL OOF NÃO substitui
    # conceitualmente AIC/BIC, que continuam sendo diagnósticos
    # auxiliares penalizados in-sample; também não é análoga a Cllr_min,
    # que mede outra coisa — perda mínima alcançável após transformação
    # monotônica ideal de scores em problema de LR). Em k-NN
    # (basis=MEAN_KNN_SIMILARITY), é T_k médio populacional (similaridade
    # média aos k vizinhos mais próximos, NÃO ao k-ésimo) — útil como
    # métrica relativa entre k's, não interpretável como log-densidade.
    oof_predictive_avg_score: Optional[float] = None
    oof_predictive_avg_score_by_fold: Optional[tuple] = None
    oof_predictive_score_type: Optional[str] = None
    n_identities_by_fold: Optional[tuple] = None
    n_scores_by_fold: Optional[tuple] = None
    ood_threshold_by_fold: Optional[tuple] = None
    ood_threshold: Optional[float] = None
    split_fingerprint: Optional[str] = None
    fold_seeds: Optional[Sequence[int]] = None
    n_folds_used: int = 0

    @property
    def oof_cdf(self) -> Optional[np.ndarray]:
        """DEPRECATED (P1-R3-9): nome confuso — armazenava SCORES,
        não CDF. Use `oof_reference_scores_sorted`.

        Emite DeprecationWarning. Será removido em revisão final V4.
        """
        import warnings
        warnings.warn(
            "ReferenceScoreCalibrationResult.oof_cdf foi renomeado para "
            "`oof_reference_scores_sorted` (o campo armazena scores OOF "
            "ordenados, não a CDF). A CDF de fato está em "
            "`oof_reference_cdf`. Alias deprecated, remoção planejada.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.oof_reference_scores_sorted


class ReferenceScoreCalibrationService:
    """Serviço de calibração de score de referência.

    Implementação por backend dispatchada via `TypicalityModelType`.
    Em PR 2 só `IN_SAMPLE_LEGACY` está implementado.
    """

    def calibrate(
        self,
        *,
        backend: TypicalityModelType,
        mode: ReferenceScoreCalibrationMode,
        trainer_factory,                                   # callable: () -> FittableTypicalityModelProtocol
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
        n_folds: int,
        random_state: int,
    ) -> ReferenceScoreCalibrationResult:
        """Dispatch principal.

        Args:
            backend: tipo do modelo (KNN / MOVMF).
            mode: CROSS_FIT_IDENTITY ou IN_SAMPLE_LEGACY.
            trainer_factory: callable sem args que constrói uma instância
                Fittable nova do backend, com hyperparams já fixados.
            embeddings: matriz (N, D) L2-normalizada (verificação fica no
                modelo conforme a `unit_norm_policy` do config).
            identity_ids: identidades por face. Obrigatório para
                `CROSS_FIT_IDENTITY` e para os modos populacionais
                baseados em identidade.
            n_folds: número de folds quando `CROSS_FIT_IDENTITY`.
            random_state: semente do split por identidade.
        """
        if mode == ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY:
            return self._calibrate_in_sample_legacy(
                backend=backend,
                trainer_factory=trainer_factory,
                embeddings=embeddings,
                identity_ids=identity_ids,
                random_state=random_state,
            )
        if mode == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY:
            return self._calibrate_cross_fit_identity(
                backend=backend,
                trainer_factory=trainer_factory,
                embeddings=embeddings,
                identity_ids=identity_ids,
                n_folds=n_folds,
                random_state=random_state,
            )
        raise ValueError(
            f"ReferenceScoreCalibrationService: modo não suportado: {mode}."
        )

    @staticmethod
    def _calibrate_cross_fit_identity(
        *,
        backend: TypicalityModelType,
        trainer_factory,
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
        n_folds: int,
        random_state: int,
    ) -> ReferenceScoreCalibrationResult:
        """K-fold por identidade → scores OOF → CDF + threshold; refit final.

        Procedimento (documentado em `docs/typicality_models.md` §5.1):

        1. Particiona identidades únicas em `n_folds` folds disjuntos.
        2. Para cada fold: treina backend nos folds train; pontua faces
           cujas identidades estão no fold test (scores OOF).
        3. Concatena OOF, ordena → CDF empírica; threshold OOD =
           quantile.
        4. Refit final em toda a população: o modelo persistido é este.

        Em PR 4 a implementação cobre k-NN (LOIO real, substituindo LOO
        de imagem) e MoVMF (k-fold por identidade). O `trainer_factory`
        é chamado uma vez por fold + uma vez no refit final.
        """
        import hashlib

        from faceaudit.enums.typicality_score_type import TypicalityScoreType

        # P0-R3-4 (revisão externa round 3): validação explícita de
        # `n_folds`. Antes, `n_folds = 0` caía em `ValueError` genérico
        # de `np.array_split`; `n_folds = 1` ia até o final e levantava
        # "nenhum score OOF produzido" — diagnóstico ruim, longe da
        # causa.
        if not isinstance(n_folds, (int, np.integer)) or int(n_folds) < 2:
            raise ValueError(
                f"CROSS_FIT_IDENTITY exige n_folds >= 2 (recebido "
                f"{n_folds!r}). Cross-fit com um único fold é semanticamente "
                "indistinguível de IN_SAMPLE."
            )
        n_folds = int(n_folds)

        if identity_ids is None or len(identity_ids) != embeddings.shape[0]:
            raise ValueError(
                "CROSS_FIT_IDENTITY exige `identity_ids` paralelo a "
                "`embeddings` (não pode ser None nem ter tamanho diferente)."
            )

        # P0-R3-4: normaliza `identity_ids` para strings cedo. Antes,
        # IDs heterogêneos como `(1, "1", 2, "2")` quebravam
        # `np.unique` com erro de ordenação entre str e int.
        ids_arr = np.array([str(i) for i in identity_ids], dtype=object)
        unique_ids = np.unique(ids_arr)
        if len(unique_ids) < n_folds:
            raise ValueError(
                f"CROSS_FIT_IDENTITY: {len(unique_ids)} identidades únicas "
                f"insuficientes para {n_folds} folds — reduza cv_n_folds "
                f"ou aumente a referência."
            )

        rng = np.random.default_rng(random_state)
        shuffled_ids = unique_ids.copy()
        rng.shuffle(shuffled_ids)
        # Splits balanceados por identidade
        fold_assignments = np.array_split(shuffled_ids, n_folds)

        # Fingerprint do split (identidade → fold index)
        identity_to_fold = {}
        for fold_idx, fold_ids in enumerate(fold_assignments):
            for ident in fold_ids:
                identity_to_fold[str(ident)] = fold_idx
        split_fp_payload = "\n".join(
            f"{ident}:{idx}"
            for ident, idx in sorted(identity_to_fold.items())
        )
        split_fingerprint = hashlib.sha256(
            split_fp_payload.encode("utf-8")
        ).hexdigest()

        # Antecipa o `population_weighting_mode` para decidir como
        # agregar OOF dentro de cada fold. Precisamos de um trainer
        # apenas pra ler o atributo — não fit().
        # P0-R3-2 + P0-R3-3: peek via helper único com seed e validação
        # de backend.
        peek_model = _build_model(
            trainer_factory,
            seed=int(random_state),
            expected_backend=backend,
        )
        pop_mode = getattr(peek_model, "population_weighting_mode", None)
        # P0-R5-5: captura snapshot semântico do peek para validar
        # consistência em folds/refit. Fail-closed em divergência.
        peek_contract = _contract_snapshot(peek_model)

        # P0-4 (revisão externa 2026-06-09): coletamos scores OOF E os
        # identity_ids correspondentes. Antes só guardávamos scores e o
        # peso da CDF OOF era uniforme face-a-face — uma identidade com
        # 20 imagens contribuía 20× mais à CDF mesmo em
        # INVERSE_IDENTITY_FREQUENCY (frequência de coleta vazando
        # silenciosamente na referência populacional).
        # P0-R2-3 (round 2): em IDENTITY_CENTROIDS agregamos os
        # EMBEDDINGS de teste por identidade DENTRO de cada fold,
        # renormalizamos, e pontuamos o centroide pelo fold_model.
        # Antes, agregávamos os SCORES por média (semanticamente
        # diferente de pontuar o centroide; diff ~0.27 no MoVMF /
        # 0.07 no kNN).
        oof_score_chunks: list[np.ndarray] = []
        oof_ids_chunks: list[np.ndarray] = []
        fold_seeds: list[int] = []
        # R4 (revisão externa 2026-06-09 round 4): diagnósticos por fold.
        # Capturados durante o loop e expostos no result para auditoria
        # de estabilidade entre folds (recomendação central do revisor:
        # sem dispersão entre folds, K=1/K=3 não pode ser comparado
        # honestamente).
        fold_avg_scores: list[float] = []
        fold_n_identities: list[int] = []
        fold_n_scores: list[int] = []
        fold_ood_thresholds: list[Optional[float]] = []
        chosen_type_overall: Optional[object] = None
        # Detecta modo agregação OOF por identidade
        from faceaudit.enums.population_weighting_mode import (
            PopulationWeightingMode,
        )
        use_centroid_oof = (
            pop_mode == PopulationWeightingMode.IDENTITY_CENTROIDS
        )
        for fold_idx, fold_ids in enumerate(fold_assignments):
            fold_seed = int(random_state) + fold_idx
            fold_seeds.append(fold_seed)
            train_mask = np.array(
                [str(i) not in set(map(str, fold_ids)) for i in ids_arr],
                dtype=bool,
            )
            test_mask = ~train_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            # P1-12 + P1-R2-10 + P0-R3-2/3: helper único de
            # instanciação com seed por fold e validação de backend.
            fold_model = _build_model(
                trainer_factory,
                seed=fold_seed,
                expected_backend=backend,
            )
            # P0-R5-5: contrato semântico do fold_model deve bater com
            # o peek (basis, pop_mode, enabled_score_types). Validação
            # ANTES do fit para falhar cedo.
            _validate_contract_matches(
                peek_contract,
                _contract_snapshot(fold_model),
                fold_index=fold_idx,
            )
            train_ids = tuple(str(i) for i in ids_arr[train_mask])
            fold_model.fit(
                embeddings[train_mask], identity_ids=train_ids,
            )

            # P0-R2-2 (revisão externa 2026-06-09 round 2): consulta
            # a base canônica do backend em vez de aplicar uma lista
            # de preferência genérica. Antes, KNN exposto a
            # LOG_DENSITY_ESTIMATE + MEAN_KNN_SIMILARITY era servido
            # com LOG_DENSITY (primeiro da lista), mas
            # `_compute_info_single` consulta T_k contra a CDF
            # interna — escala incompatível → todos os percentis em
            # 100%. A propriedade `reference_score_basis_type` declara
            # explicitamente qual grandeza alimenta a referência
            # interna de cada backend (KNN → MEAN_KNN_SIMILARITY;
            # MoVMF → LOG_DENSITY_ESTIMATE).
            chosen_type = getattr(
                fold_model, "reference_score_basis_type", None
            )
            if chosen_type is None or chosen_type not in fold_model.enabled_score_types:
                raise RuntimeError(
                    "CROSS_FIT_IDENTITY: backend não expõe "
                    "`reference_score_basis_type` válido para construir "
                    "CDF OOF — verifique se o modelo implementa o "
                    "protocolo atualizado (P0-R2-2)."
                )

            if use_centroid_oof:
                # P0-R2-3: agrega EMBEDDINGS por identidade dentro
                # do fold; renormaliza L2 (centroide na esfera);
                # pontua o centroide com o fold_model. Coerente com
                # o `fit` em IDENTITY_CENTROIDS, que vê 1 centroide
                # por identidade na referência.
                test_embeddings = embeddings[test_mask]
                test_ids = ids_arr[test_mask]
                uniq_ids, inv_idx = np.unique(test_ids, return_inverse=True)
                centroids = np.zeros(
                    (len(uniq_ids), test_embeddings.shape[1]),
                    dtype=np.float64,
                )
                counts = np.zeros(len(uniq_ids), dtype=np.int64)
                for emb, j in zip(test_embeddings, inv_idx):
                    centroids[j] += emb
                    counts[j] += 1
                # Renormaliza L2 — preserva a geometria esférica.
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                centroids /= norms
                batch = fold_model.compute_scores(centroids, chosen_type)
                fold_scores = np.asarray(batch.scores, dtype=np.float64)
                fold_ids = uniq_ids.copy()
            else:
                batch = fold_model.compute_scores(
                    embeddings[test_mask], chosen_type,
                )
                fold_scores = np.asarray(batch.scores, dtype=np.float64)
                fold_ids = ids_arr[test_mask].copy()

            oof_score_chunks.append(fold_scores)
            oof_ids_chunks.append(fold_ids)

            # R4: diagnóstico por fold. Avg do fold ponderado pelo modo:
            # FACE_LEVEL_RAW / IDENTITY_CENTROIDS → média simples;
            # INVERSE_IDENTITY_FREQUENCY → pesos 1/n_id dentro do fold
            # (cada identidade contribui igualmente ao avg).
            if pop_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
                _, fold_inv, fold_counts = np.unique(
                    fold_ids, return_inverse=True, return_counts=True,
                )
                w_fold = 1.0 / fold_counts[fold_inv]
                fold_avg = float(np.sum(w_fold * fold_scores) / w_fold.sum())
            else:
                fold_avg = float(np.mean(fold_scores))
            fold_avg_scores.append(fold_avg)
            fold_n_identities.append(int(len(np.unique(fold_ids))))
            fold_n_scores.append(int(len(fold_scores)))
            fold_ood_thresholds.append(
                _safe_float(getattr(fold_model, "_ood_log_likelihood_threshold", None))
                if backend == TypicalityModelType.MOVMF
                else _safe_float(getattr(getattr(fold_model, "_state", None), "ood_threshold", None))
            )
            # `chosen_type` é determinístico por backend, mas capturamos
            # o primeiro fold para anexar ao result.
            if chosen_type_overall is None:
                chosen_type_overall = chosen_type

        if not oof_score_chunks:
            raise RuntimeError(
                "CROSS_FIT_IDENTITY: nenhum score OOF produzido — verifique "
                "splits ou trainer."
            )
        oof_scores_unsorted = np.concatenate(oof_score_chunks)
        oof_ids_unsorted = np.concatenate(oof_ids_chunks)

        # Refit final em toda a população
        # P0-R3-3: refit também via helper, com seed = random_state
        # principal (não offset de fold). Garante reprodutibilidade.
        refit_model = _build_model(
            trainer_factory,
            seed=int(random_state),
            expected_backend=backend,
        )
        # P0-R5-5: refit também precisa preservar o contrato do peek.
        _validate_contract_matches(
            peek_contract, _contract_snapshot(refit_model), fold_index=None,
        )
        all_ids = tuple(str(i) for i in ids_arr)
        refit_model.fit(embeddings, identity_ids=all_ids)

        # Pesos OOF derivados do `population_weighting_mode` do modelo
        # treinado. Lógica:
        #   INVERSE_IDENTITY_FREQUENCY → w_ij = 1/n_i_oof por face.
        #   IDENTITY_CENTROIDS → 1 score por identidade (já agregado
        #     ao pontuar o centroide dentro do fold em P0-R2-3),
        #     w = 1.
        #   FACE_LEVEL_RAW → uniforme face-a-face (legado).
        if pop_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
            _, inv_idx, id_counts = np.unique(
                oof_ids_unsorted, return_inverse=True, return_counts=True,
            )
            oof_weights = (1.0 / id_counts[inv_idx]).astype(np.float64)
            oof_scores_final = oof_scores_unsorted
        elif pop_mode == PopulationWeightingMode.IDENTITY_CENTROIDS:
            # P0-R2-3: scores já são "1 por identidade" (centroide
            # pontuado dentro do fold). Apenas pesos uniformes.
            oof_weights = np.ones_like(oof_scores_unsorted, dtype=np.float64)
            oof_scores_final = oof_scores_unsorted
        else:  # FACE_LEVEL_RAW ou ausente
            oof_weights = np.ones_like(oof_scores_unsorted, dtype=np.float64)
            oof_scores_final = oof_scores_unsorted

        # Ordena scores + pesos paralelamente para a CDF/threshold.
        order = np.argsort(oof_scores_final)
        oof_arr_sorted = oof_scores_final[order]
        oof_weights_sorted = oof_weights[order]

        # Threshold OOD = quantil ponderado em q = ood_quantile.
        # Computa CDF SEMPRE — usada tanto para threshold quanto
        # exposta no result (P1-R3-9).
        ood_quantile = float(getattr(refit_model, "ood_quantile", 0.01))
        total_w = oof_weights_sorted.sum()
        cdf_arr = np.cumsum(oof_weights_sorted) / total_w
        if oof_quantile_in_range(ood_quantile):
            idx = int(np.searchsorted(cdf_arr, ood_quantile, side="left"))
            idx = min(max(idx, 0), len(oof_arr_sorted) - 1)
            threshold = float(oof_arr_sorted[idx])
        else:
            threshold = None

        # R4 (revisão externa 2026-06-09 round 4): log-likelihood
        # preditiva OOF agregada. É a média ponderada de TODOS os
        # scores OOF com os mesmos pesos usados na CDF — coerente com
        # a interpretação "estimador da log-densidade esperada na
        # população". No MoVMF, score_type = LOG_DENSITY_ESTIMATE
        # implica que este número é diretamente PLL_OOF (alias exposto
        # no summary do MoVMF como `oof_predictive_avg_log_likelihood`).
        # No k-NN (score_type = MEAN_KNN_SIMILARITY), é a similaridade
        # média ao k-ésimo vizinho — útil como métrica relativa entre
        # ks, não como log-densidade.
        oof_predictive_avg = float(
            np.sum(oof_weights * oof_scores_final) / oof_weights.sum()
        )

        # P0-1: instala a referência OOF no modelo final. Após este
        # método retornar, `refit_model.get_percentiles()` e
        # `is_in_distribution()` consultam a distribuição OOF — não a
        # in-sample do refit. O `ReferenceScoreCalibrationResult`
        # continua retornando os mesmos arrays para auditoria.
        #
        # R4 + P0-R5-1 (revisão externa 2026-06-09 round 5): o serviço
        # passa TODOS os diagnósticos por fold ao install. NÃO há
        # fallback try/except TypeError — backend deve implementar o
        # contrato. (Decisão: retrocompat dispensável; o `install` é
        # parte da fronteira pública e backends desatualizados devem
        # falhar fechado em vez de descartar métricas silenciosamente.)
        refit_model.install_reference_score_calibration(
            reference_scores=oof_arr_sorted,
            reference_weights=oof_weights_sorted,
            ood_threshold=threshold,
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            split_fingerprint=split_fingerprint,
            oof_predictive_avg_score=oof_predictive_avg,
            oof_predictive_avg_score_by_fold=tuple(fold_avg_scores),
            oof_predictive_score_type=(
                chosen_type_overall.to_str()
                if chosen_type_overall is not None
                and hasattr(chosen_type_overall, "to_str")
                else (str(chosen_type_overall) if chosen_type_overall else None)
            ),
            n_identities_by_fold=tuple(fold_n_identities),
            n_scores_by_fold=tuple(fold_n_scores),
            ood_threshold_by_fold=tuple(fold_ood_thresholds),
            fold_seeds=tuple(fold_seeds),
            n_folds_used=int(n_folds),
        )

        # P1-R3-9: arrays expostos no result são read-only.
        # Auditoria não deve poder mutar o snapshot devolvido.
        scores_ro = np.ascontiguousarray(oof_arr_sorted, dtype=np.float64)
        scores_ro.setflags(write=False)
        weights_ro = np.ascontiguousarray(oof_weights_sorted, dtype=np.float64)
        weights_ro.setflags(write=False)
        cdf_ro = np.ascontiguousarray(cdf_arr, dtype=np.float64)
        cdf_ro.setflags(write=False)

        return ReferenceScoreCalibrationResult(
            refit_model=refit_model,
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_reference_scores_sorted=scores_ro,
            oof_reference_weights_sorted=weights_ro,
            oof_reference_cdf=cdf_ro,
            oof_predictive_avg_score=oof_predictive_avg,
            oof_predictive_avg_score_by_fold=tuple(fold_avg_scores),
            oof_predictive_score_type=(
                chosen_type_overall.to_str()
                if chosen_type_overall is not None
                and hasattr(chosen_type_overall, "to_str")
                else (str(chosen_type_overall) if chosen_type_overall else None)
            ),
            n_identities_by_fold=tuple(fold_n_identities),
            n_scores_by_fold=tuple(fold_n_scores),
            ood_threshold_by_fold=tuple(fold_ood_thresholds),
            ood_threshold=threshold,
            split_fingerprint=split_fingerprint,
            fold_seeds=tuple(fold_seeds),
            n_folds_used=int(n_folds),
        )

    @staticmethod
    def _calibrate_in_sample_legacy(
        *,
        backend: TypicalityModelType,
        trainer_factory,
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
        random_state: int,
    ) -> ReferenceScoreCalibrationResult:
        """k-NN nativo já faz LOO de imagem internamente.

        P0-R5-4 (revisão externa 2026-06-09 round 5): o caminho legado
        agora propaga `random_state` ao trainer (antes usava seed=0
        hardcoded, divergindo da configuração do caller) e valida o
        backend declarado contra a instância retornada via `_build_model`
        com `expected_backend` (antes não validava, aceitando silenciosamente
        factories que devolviam backend errado em modo legado).
        """
        model = _build_model(
            trainer_factory,
            seed=int(random_state),
            expected_backend=backend,
        )
        if identity_ids is not None:
            model.fit(embeddings, identity_ids=tuple(str(i) for i in identity_ids))
        else:
            model.fit(embeddings)
        # Threshold OOD persistido no próprio modelo via `ood_quantile`.
        return ReferenceScoreCalibrationResult(
            refit_model=model,
            mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
            oof_reference_scores_sorted=None,
            oof_reference_weights_sorted=None,
            oof_reference_cdf=None,
            ood_threshold=None,
            split_fingerprint=None,
            fold_seeds=None,
            n_folds_used=0,
        )
