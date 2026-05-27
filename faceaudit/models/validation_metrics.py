from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationMetrics:
    """
    Indicadores absolutos de desempenho de um modelo biométrico forense.

    Organizado em tiers de relevância (ordem dos fields espelha a UI):

    **Tier 1 — Principais** (sempre exibidas no summary header):
        eer, eer_threshold, cllr, cllr_min, auc, wrong_sign_gen_rate,
        wrong_sign_imp_rate

    **Tier 2 — Decomposição per-hypothesis**:
        cllr_gen, cllr_imp, cllr_min_gen, cllr_min_imp

    **Tier 3 — Análise affine** (Brümmer-Doddington):
        cllr_affine, calibration_slope, calibration_intercept

    **Tier 4 — ECE numérico**:
        ece_calibration_loss_area, ece_max_gap, ece_argmax_prior_log_odds

    **Tier 5 — Tail/outliers** (diagnóstico interno):
        loss_gen_q95, loss_imp_q95, top_1pct_loss_share

    Campos não-Tier-1 são `Optional[float] = None` pra preservar retrocompat
    com payloads antigos de StrategyValidationResult salvos antes desta
    versão. Loaders devem tratar None como "não calculado".

    Properties derivadas fornecem decomposições úteis:
        cal_loss            = cllr − cllr_min          (perda calibração total)
        cal_loss_gen        = cllr_gen − cllr_min_gen  (perda nos genuínos)
        cal_loss_imp        = cllr_imp − cllr_min_imp  (perda nos impostores)
        affine_cal_loss     = cllr − cllr_affine       (corrigível por affine)
        nonlinear_cal_loss  = cllr_affine − cllr_min   (residual não-linear)
    """

    # ── Tier 1: principais ───────────────────────────────────────────────
    eer: float                 # Equal Error Rate (Poder de Discriminação bruto)
    eer_threshold: float       # O valor exato de LR/Score onde o EER ocorre
    cllr: float                # Cost of Log-Likelihood Ratio (Desempenho Forense Real)
    cllr_min: float            # Cllr Mínimo possível (Desempenho ótimo se perfeitamente calibrado)
    auc: Optional[float] = None  # Area Under the Curve (ROC)

    # ── Tier 1 (cont.): taxas de "sinal errado" ──────────────────────────
    wrong_sign_gen_rate: Optional[float] = None  # P(log-LR < 0 | Hp) — falso negativo direto
    wrong_sign_imp_rate: Optional[float] = None  # P(log-LR > 0 | Hd) — falso positivo direto

    # ── Tier 2: decomposição per-hypothesis ──────────────────────────────
    cllr_gen: Optional[float] = None        # mean over gen de log2(1+e^(-llr))
    cllr_imp: Optional[float] = None        # mean over imp de log2(1+e^(+llr))
    cllr_min_gen: Optional[float] = None    # mesmo, sobre PAVA-calibrados
    cllr_min_imp: Optional[float] = None

    # ── Tier 3: análise affine (Brümmer-Doddington) ──────────────────────
    cllr_affine: Optional[float] = None         # Cllr após affine ótimo (slope·llr + intercept)
    calibration_slope: Optional[float] = None   # slope do affine fit
    calibration_intercept: Optional[float] = None  # intercept do affine fit

    # ── Tier 4: ECE numérico ─────────────────────────────────────────────
    ece_calibration_loss_area: Optional[float] = None  # ∫(ECE − ECE_min) d(prior_log10_odds)
    ece_max_gap: Optional[float] = None                # max(ECE − ECE_min)
    ece_argmax_prior_log_odds: Optional[float] = None  # prior onde ece_max_gap acontece

    # ── Tier 5: tail/outliers ────────────────────────────────────────────
    loss_gen_q95: Optional[float] = None         # quantil 95% de log2(1+e^(-llr_gen))
    loss_imp_q95: Optional[float] = None         # quantil 95% de log2(1+e^(+llr_imp))
    top_1pct_loss_share: Optional[float] = None  # fração do Cllr vinda dos 1% piores samples

    # ── Tier 1 (cont.): diagnóstico do calibrador (CV in-domain) ─────────
    # Cllr in-sample e CV out-of-fold no DOMÍNIO de calibração. Anexados
    # via SCORE_CALIBRATION → score_dataset.metadata → STRATEGY_EVALUATION.
    # Permitem decompor a perda total em: overfit in-domain (cv_gap) e
    # transfer loss / domain shift (calibrator_to_validation_gap).
    calibrator_cllr_train: Optional[float] = None      # Cllr no próprio treino do calibrador (otimista)
    calibrator_cllr_train_gen: Optional[float] = None  # decomposição genuínos
    calibrator_cllr_train_imp: Optional[float] = None  # decomposição impostores
    calibrator_cllr_cv: Optional[float] = None         # Cllr CV out-of-fold no domínio de calibração
    calibrator_cllr_cv_gen: Optional[float] = None     # decomposição genuínos
    calibrator_cllr_cv_imp: Optional[float] = None     # decomposição impostores

    # =========================================================================
    # PROPRIEDADES DERIVADAS
    # =========================================================================

    @property
    def cal_loss(self) -> float:
        """Perda de Calibração total: cllr − cllr_min. Garantida ≥ 0."""
        return max(0.0, self.cllr - self.cllr_min)

    @property
    def is_well_calibrated(self) -> bool:
        """Heurística forense: cal_loss < 0.1 bits → calibração aceitável."""
        return self.cal_loss < 0.1

    @property
    def cal_loss_gen(self) -> Optional[float]:
        """Perda de calibração nos genuínos. None se Tier 2 não foi calculado."""
        if self.cllr_gen is None or self.cllr_min_gen is None:
            return None
        return max(0.0, self.cllr_gen - self.cllr_min_gen)

    @property
    def cal_loss_imp(self) -> Optional[float]:
        """Perda de calibração nos impostores. None se Tier 2 não foi calculado."""
        if self.cllr_imp is None or self.cllr_min_imp is None:
            return None
        return max(0.0, self.cllr_imp - self.cllr_min_imp)

    @property
    def affine_cal_loss(self) -> Optional[float]:
        """Perda recuperável só com recalibração affine (slope+intercept)."""
        if self.cllr_affine is None:
            return None
        return max(0.0, self.cllr - self.cllr_affine)

    @property
    def nonlinear_cal_loss(self) -> Optional[float]:
        """Perda residual não-corrigível por affine (precisa PAVA/isotônica)."""
        if self.cllr_affine is None:
            return None
        return max(0.0, self.cllr_affine - self.cllr_min)

    # ── Gaps diagnósticos do calibrador ──────────────────────────────────
    # Decompõem a perda total em duas componentes ortogonais:
    #   cv_gap                       = in-domain overfit do calibrador
    #   calibrator_to_validation_gap = transfer loss (domain shift CFP→BFW etc)
    # Um calibrador pode ter cv_gap≈0 (sem overfit interno) e ainda assim
    # ter grande calibrator_to_validation_gap (caso típico CFP → BFW).

    @property
    def calibrator_cv_gap(self) -> Optional[float]:
        """In-domain overfit do calibrador. > 0 = calibrador overfitou
        sobre o próprio dataset de calibração. Comparação válida entre
        train e CV no MESMO domínio. NÃO mede transfer pra validação
        externa (use calibrator_to_validation_gap)."""
        if self.calibrator_cllr_cv is None or self.calibrator_cllr_train is None:
            return None
        return self.calibrator_cllr_cv - self.calibrator_cllr_train

    @property
    def calibrator_to_validation_gap(self) -> Optional[float]:
        """Perda de transferência: Cllr validação externa − Cllr CV no
        domínio de calibração.

        > 0 = validação externa pior que o desempenho out-of-fold no
        treino do calibrador. Pode REFLETIR domain shift, mas também:
        validação mais difícil (qualidade/pose/demografia diferente),
        protocolo de pares diferente, número efetivo de identidades
        menor, ou simplesmente ruído estatístico.

        NÃO é overfit do calibrador (esse é `calibrator_cv_gap`) — um
        calibrador pode ter cv_gap≈0 e ainda assim grande gap aqui.
        Mas este número isolado NÃO prova causa específica; serve como
        sinal de investigação ('algo é diferente entre os dois conjuntos')."""
        if self.calibrator_cllr_cv is None:
            return None
        return self.cllr - self.calibrator_cllr_cv
