# Conditional Monotonicity Rule — Design Document

**Versão**: 1.0 (consolidada após duas rodadas de revisão crítica do colaborador forense)
**Data**: 2026-06-07
**Status**: V1 aprovada para implementação; V1.5 e V2 enfileiradas; projeto paralelo `TYPICALITY_EFFECT_AUDIT` definido.

---

## 1. Visão geral

Regra plugável `ValidationRuleType.CONDITIONAL_MONOTONICITY` que verifica, **após o refit final** do calibrador, que a derivada `∂ log_LR / ∂ s` é estritamente positiva em todas as comparações efetivamente observadas em `D_cal`. Auditoria estendida cobre cantos da caixa central de raridades observadas (`central_box`, P5-P95) e — em V2 — o domínio explicitamente declarado pela política de reportabilidade (`reportable_box`).

**Anexa-se** ao `CALIBRATION_MODEL_TRAINING` como regra default auto-populada, paralela às existentes (`MIN_CALIBRATION_PAIRS`, `SCORE_CALIBRATION_STATE`).

**Cobertura V1**: todas as 15 arquiteturas declarativas de `LR_ARCHITECTURE_VECTORIZATION`.

---

## 2. Escopo de domínios de auditoria

Três domínios distintos, respondendo a perguntas diferentes:

| Domínio | Definição | Função | Status policy V1 |
|---|---|---|---|
| `observed_rows` | Para cada par `(probe, reference)` em `D_cal`, avaliar a derivada em sua exata combinação `(r̃_F, r̃_P)` | Gate empírico mínimo obrigatório | `CRITICAL` se violado |
| `central_box` | Cantos do retângulo `[P5(r̃_F), P95(r̃_F)] × [P5(r̃_P), P95(r̃_P)]` | Diagnóstico conservador de estabilidade na região central | `WARNING` se violado |
| `reportable_box` | Domínio explicitamente declarado pela política de reportabilidade | Gate operacional formal (V2) | `CRITICAL` quando V2 ativa |

**Justificativa da separação:**
- `P5-P95` marginal não cobre 95% em 2D (sob independência ~81%).
- Derivada pode ser positiva no centro e negativa nas caudas observadas → gate só na caixa central tem furo.
- Canto retangular pode corresponder a combinação `(r̃_F, r̃_P)` que praticamente não existe na população observada → uma violação em canto vazio merece WARNING + investigação, não rejeição.

---

## 3. Arquitetura: protocolo + providers especializados

```python
# faceaudit/protocols/monotonicity_derivative_provider_protocol.py
class MonotonicityDerivativeProviderProtocol(Protocol):
    """Fornece derivadas ∂ log_LR / ∂ s por região para auditoria."""
    def evaluate_regions(
        self,
        *,
        calibrator_metadata: dict,
        raw_input_payload: object,
    ) -> list[MonotonicityRegionResult]:
        ...
```

**Implementações V1:**
- `LRArchitectureDerivativeProvider` — usa `LRFeatureTermSpec` do registry; gera derivadas das 15 arquiteturas declarativas.

**Implementação V1.5 (não inclusa em V1):**
- `PiecewiseLinearDerivativeProvider` — calcula inclinações acumuladas do piecewise legado com `K` knots arbitrários (`m_j = β_1 + Σ_{k=1}^{j} γ_k`).

**Não estender `LRFeatureTermSpec` para abarcar piecewise legado.** A abstração existente foi desenhada para catálogo estático; piecewise gera termos dinâmicos `max(s − k_j, 0)`. Forçar acoplamento errado.

---

## 4. Estrutura de dados

```python
@dataclass(frozen=True)
class MonotonicityRegionResult:
    region_id: str                          # "s_gt_s0", "s_lt_s0", "segment_2"
    region_description: str                 # human-readable

    # Para arquiteturas com derivada constante na região (sem raridade):
    slope_value: float | None

    # Para arquiteturas com dependência de raridade (LR_ARCH_VEC multivariada):
    slope_at_observed_rows: np.ndarray | None       # (n_pairs,) derivada por par
    slope_at_central_box_corners: list[float] | None  # 1, 2 ou 4 cantos
    worst_observed_row: tuple[int, dict] | None     # (índice do par, dict de features)
    worst_central_corner: tuple[str, dict] | None   # (label do canto, dict)

    # Suporte:
    n_observed_rows_in_region: int
    feature_variance_in_region: dict[str, float]
    zero_variance_terms: list[str]
    near_zero_variance_terms: list[str]
```

---

## 5. Parâmetros configuráveis

Via `ValidationRuleConfig.params`:

| Parâmetro | Tipo | Default | Função |
|---|---|---|---|
| `calibrator_resource_id` | str | (obrigatório) | Calibrador a auditar — dataset resolvido pela proveniência |
| `numerical_epsilon` | float | `1e-12` | Tolerância para artefatos de ponto flutuante |
| `operational_min_slope` | float \| None | `None` | Margem substantiva opcional (V1: só warning relativo) |
| `box_quantile_low` | float | `0.05` | P_low para `central_box` |
| `box_quantile_high` | float | `0.95` | P_high para `central_box` |
| `policy` | Enum | `WARNING_FOR_CENTRAL_BOX` | Política inicial (V2: `REPORTABLE_BOX_GATE`) |

**`dataset_resource_id_override`** existe apenas como override explícito, com warning obrigatório e registro de auditoria.

---

## 6. Algoritmo

```
1. Resolver dataset via calibrator metadata:
   - calibrator.training_raw_input_resource_id → raw_input_payload
   - calibrator.training_raw_input_fingerprint → expected fingerprint

2. Recalcular fingerprints do calibrator.architecture_state:
   - LRArchitectureState.from_dict(...)
   - Verifica vectorization_contract_fingerprint, provenance_record_fingerprint
   - Falha imediata com StateIntegrityError em mismatch

3. Comparar contrato:
   - calibrator.vectorization_contract_fingerprint == raw_input_payload.vectorization_contract_fingerprint
   - Falha CRITICAL em divergência

4. Selecionar provider:
   - LRArchitectureDerivativeProvider para arquiteturas de LR_ARCH_VEC

5. Para cada região retornada pelo provider:
   - Avaliar slope_at_observed_rows (todos os pares da região em D_cal)
   - Avaliar slope_at_central_box_corners (cantos do retângulo P5-P95)
   - Identificar feature_variance_in_region, zero_variance_terms

6. Computar métricas globais:
   - min_slope_observed_rows = min(slopes em observed)
   - min_slope_central_box = min(slopes nos cantos)
   - max_slope_observed_rows, max_slope_central_box
   - relative_slope_ratio = min_slope / |β_raw[s]|
   - min_slope_x_score_range = min_slope * (P95(s) - P5(s))   # variação mínima em nats

7. Aplicar política de status:
   - min_slope_observed_rows <= numerical_epsilon → CRITICAL
   - slope positiva em observed mas negativa em canto central_box → WARNING
   - feature com zero variance em região → WARNING
   - relative_slope_ratio < limiar diagnóstico → WARNING ("curva quase plana")
   - Caso contrário: OK

8. Persistir metadata completo no calibrador via CalibrationModelMetadataService.
```

---

## 7. Política formal de status (V1)

| Situação | Status |
|---|---|
| `min_slope_observed_rows <= numerical_epsilon` | **CRITICAL** |
| Slope positiva em observed_rows mas negativa em canto da central_box | WARNING |
| Slope positiva mas muito próxima de zero conforme operational_min_slope | WARNING |
| Feature com variância zero em alguma região | WARNING |
| Região piecewise sem suporte de uma das classes (V1.5) | WARNING ou CRITICAL conforme política |
| `vectorization_contract_fingerprint` divergente entre calibrador e raw_input | **CRITICAL** |
| Raw_input não localizável | **CRITICAL** |
| Calibrador sem `architecture_state` (legado) | **CRITICAL** (V1) ou `NOT_AUDITED` com override (V1.5+) |

Quando V2 ativa:
| `min_slope_reportable_box <= numerical_epsilon` | **CRITICAL** |

---

## 8. Metadados persistidos no calibrador

```python
calibrator.metadata["monotonicity_audit"] = {
    # Versionamento e proveniência
    "rule_version": "1.0",
    "rule_run_timestamp": "2026-06-07T17:34:00Z",
    "audit_status": "OK" | "WARNING" | "CRITICAL" | "NOT_AUDITED",
    "warnings": ["..."],
    "policy": "WARNING_FOR_CENTRAL_BOX",
    
    # Identificação do que foi auditado
    "architecture": "HIERARCHICAL_CENTERED_ONE_FACE",
    "architecture_spec_fingerprint": "...",
    "vectorization_contract_fingerprint": "...",
    "calibrator_fingerprint": "...",
    "training_raw_input_resource_id": "...",
    "training_raw_input_fingerprint": "...",
    
    # Configuração da auditoria
    "audit_domain_modes": ["observed_rows", "central_box"],
    "box_quantile_low": 0.05,
    "box_quantile_high": 0.95,
    "numerical_epsilon": 1e-12,
    "operational_min_slope": None,
    
    # Resultados — observed_rows
    "min_slope_observed_rows": +0.123,
    "worst_observed_row": {"index": 12345, "s": 0.10, "r_F": 0.95, ...},
    
    # Resultados — central_box
    "min_slope_central_box": +0.089,
    "worst_central_corner": {"label": "corner_pp_min_rf_min_rp", "values": {...}},
    
    # Resultados — relativos
    "relative_slope_ratio": 0.045,
    "min_slope_x_score_range": +0.062,   # nats sobre faixa P5-P95 de s
    
    # Por região
    "slopes_by_region": {
        "s_gt_s0": {...},
        "s_lt_s0": {...}
    },
    
    # Suporte
    "segment_support": {"s_gt_s0": 134000, "s_lt_s0": 26000},
    "zero_variance_feature_terms": [],
    "near_zero_variance_feature_terms": [],
}
```

---

## 9. Casos de teste

`tests/test_conditional_monotonicity_rule.py`:

1. COSSENO — slope = β_s constante; OK trivial
2. HINGE_PURA com `β_h+ > -β_s` → OK em ambas as regiões
3. HINGE_PURA com `β_h+ < -β_s` → CRITICAL em `s > s_0`
4. HIER_ONE_FACE_F — caixa 1D; verificar nos 2 cantos
5. HIER full — caixa 2D; verificar nos 4 cantos
6. BIL cheia com coefs de §7.4 TIER4 (todos positivos) → OK
7. BIL cheia com 1 coef forçado negativo no canto → CRITICAL com `worst_corner` identificado
8. `operational_min_slope` > 0 → WARNING em curva quase plana
9. Policy `WARNING` vs `ERROR` — mesma matemática, status distintos
10. Caixa custom via quantis personalizados — respeitar
11. Persistência completa de metadados pós-validate

**Testes adicionais (sua sugestão):**

12. Violação apenas abaixo de P5 ou acima de P95 → observed_rows CRITICAL mas central_box OK (prova separação)
13. Raridades fortemente correlacionadas → canto vazio violante; observed_rows OK, central_box WARNING
14. Feature com variância zero → diagnóstico explícito
15. Scaler ausente ou incompatível → `MissingScalerError`
16. Fingerprint divergente → `StateIntegrityError` antes de auditoria
17. `s = s_0` exatamente → testa slopes laterais distintos (`left_slope_at_s_0`, `right_slope_at_s_0`)
18. Arquitetura nova adicionada sem derivada → teste paramétrico FALHA
19. Refit aprovado, fold interno reprovado → V1.5; em V1 marca `fold_audit_status = NOT_EVALUATED`
20. Calibrador legado sem `architecture_state` → CRITICAL em casework

**Teste paramétrico de cobertura** (`tests/test_monotonicity_provider_coverage.py`):

```python
@pytest.mark.parametrize("arch_spec", LR_ARCHITECTURE_SPECS.values(),
                         ids=lambda s: s.architecture.name)
def test_provider_covers_architecture(arch_spec):
    provider = LRArchitectureDerivativeProvider(arch_spec)
    # mock calibrator + raw_input
    regions = provider.evaluate_regions(calibrator_metadata=..., raw_input_payload=...)
    assert len(regions) >= 1
    assert all(hasattr(r, 'slope_at_observed_rows') for r in regions)
```

Adicionar nova arquitetura ao `LR_ARCHITECTURE_SPECS` sem cobrir o provider faz a suíte falhar imediatamente.

---

## 10. Integração com infraestrutura existente

1. **Enum**: `faceaudit/enums/validation_rule_type.py` — adicionar `CONDITIONAL_MONOTONICITY`
2. **Protocolo**: `faceaudit/protocols/monotonicity_derivative_provider_protocol.py`
3. **Service**: `faceaudit/services/lr_architecture_derivative_provider.py`
4. **Regra**: `faceaudit/validation_rules/conditional_monotonicity_rule.py` extendendo `ValidationRuleBase`
5. **Factory**: `faceaudit/factories/validation_rule_factory.py` — mapping `CONDITIONAL_MONOTONICITY → ConditionalMonotonicityRule`
6. **Auto-populate**: `faceaudit/models/instructions/calibration_model_training_instruction.py._ensure_default_validation_rules` — auto-adiciona `CONDITIONAL_MONOTONICITY` para qualquer arquitetura LR_ARCH_VEC
7. **UI**: automático via `InstructionConfigPanel` (a regra aparece na seção âmbar com botões 🗑/📤/➕)
8. **Persistência**: `CalibrationModelMetadataService.bind_audit_result(model, audit_metadata)`

---

## 11. Política para modelos legados

Modelos calibrados antes da implementação da regra não terão metadata de auditoria.

| Ambiente | Tratamento |
|---|---|
| Desenvolvimento | `audit_status = NOT_AUDITED` com warning |
| Validação formal | Re-auditoria obrigatória |
| Casework | Bloquear, OU aceitar com `legacy_override_reason` + `override_user` + `override_timestamp` persistidos explicitamente |

---

## 12. Roadmap V1 → V1.5 → V2 + projeto paralelo

### V1 — gate mínimo confiável

**Escopo:** Todas as arquiteturas declarativas de `LR_ARCHITECTURE_VECTORIZATION`.

**Inclui:**
- `LRArchitectureDerivativeProvider` via `LRFeatureTermSpec`
- `observed_rows` + `central_box`
- Resolução do dataset pela proveniência do calibrador
- Validação de fingerprints antes da auditoria
- `numerical_epsilon` + `operational_min_slope` opcional
- `relative_slope_ratio` + `min_slope_x_score_range`
- Diagnóstico de feature degenerada
- Metadata completo
- Policy: WARNING para envelope central, CRITICAL para observed_rows
- Testes paramétricos para todas as arquiteturas registradas

### V1.5 — estabilidade e compatibilidade ampla

**Component A — `PiecewiseLinearDerivativeProvider`** ✅ IMPLEMENTADO

- Cobre `LogisticRegressionCalibrationModel` com `piecewise_linear_config.mode = PIECEWISE_LINEAR` (calibradores piecewise legado de score 1D, com K knots arbitrários).
- Não exige `architecture_state` nem fingerprint matching — todos os diagnósticos necessários já são persistidos pelo próprio calibrador no `_assert_monotonic_after_fit`: `piecewise_knots_absolute`, `segment_slopes_raw`, `hinge_anchor_s0`.
- Constrói `MonotonicityRegionResult` por segmento (K+1 segmentos para K knots) com `slope_value` constante por região.
- Aceita `reportable_box["s"]` ou `["base_score"]` (V2) — chave nova adicionada ao parser da factory; cantos da intersecção entre cada segmento e a caixa; segmentos sem intersecção são pulados (não auditados naquela caixa).
- Observed_scores opcionais via `dataset_resource_id_override` — piecewise legado tipicamente não persiste `training_raw_input_resource_id`, então o fluxo continua funcional sem ele.
- Dispatch automático na regra via campos do metadata: `architecture_state` presente → `LRArchitectureDerivativeProvider`; `piecewise_enabled=True` (sem `architecture_state`) → `PiecewiseLinearDerivativeProvider`.
- Testes: 11 testes V1.5 (`monotonic_returns_correct_segments`, `detects_violation_via_min_slope`, `reportable_box_with_intersection`, `accepts_base_score_alias`, `raises_on_knots_slopes_mismatch`, `raises_when_metadata_missing_fields`, `dispatches_to_piecewise_when_enabled`, `piecewise_non_monotonic_is_critical`, `piecewise_reportable_box_violation_is_critical`, `piecewise_reportable_box_outside_violation_is_ok`, `without_architecture_state_and_without_piecewise_errors`).

**Components B–D** (não bloqueantes para uso do provider piecewise):
- Auditoria por fold com cache validado por fingerprint
- Diagnóstico completo de suporte por segmento (já parcialmente coberto pelo `segment_counts_*` que o calibrador persiste)
- Política expandida para modelos legados

### V2 — endurecimento operacional

**Component 1 — `reportable_box`** ✅ IMPLEMENTADO

- Parâmetro opcional `reportable_box: Optional[dict]` na regra (default None preserva V1.5).
- Aceita chaves `"r_probe"`, `"r_reference"`, `"r_single"` mapeadas para `(lo, hi)` em escala bruta.
- Factory parser tolerante valida estrutura/par/limites (lo ≤ hi) — config inválida levanta `ValueError`.
- Provider (`LRArchitectureDerivativeProvider.evaluate_regions`) ganha parâmetro `reportable_box`; quando fornecido, avalia slopes nos cantos cartesianos das dimensões declaradas via mesmo helper `_evaluate_corners` reutilizado pelo `central_box`.
- `MonotonicityRegionResult` ganha 4 campos novos: `slope_at_reportable_box_corners`, `reportable_corner_labels`, `worst_reportable_corner_label`, `worst_reportable_corner_values` + método `min_slope_reportable_box()`.
- Policy: `min_slope_reportable_box ≤ numerical_epsilon` → `CRITICAL` (gate operacional). Domínio é declarado pelo usuário como política — violação ali é mais grave que canto da `central_box`.
- `audit_metadata` ganha entrada `reportable_box` e inclui `"reportable_box"` em `audit_domain_modes` quando ativo.
- `RULE_VERSION` → `"2.0"`.
- Testes: 6 testes V2 (`reportable_box_absent_preserves_v15_behavior`, `_healthy_status_ok`, `_violation_critical`, `_invalid_dict_raises`, `_invalid_bounds_raises`, `_non_pair_raises`).

**Components 2-4** (pendente de design adicional antes de codar):
- Opt-out restrito por ambiente DEV/VALIDATION/PRODUCTION — exige decisão sobre como o ambiente é declarado (config do projeto? env var? setting persistido?).
- Bloqueio de casework para modelos não certificados — exige integração no `score_calibration_instruction_executor` + schema de certificação.
- Histórico de overrides — exige schema de persistência (quem, quando, por quê, hash do modelo no momento do override).
- Dashboard consolidado de auditoria.

### Projeto paralelo — `TYPICALITY_EFFECT_AUDIT`

Auditoria conceitualmente distinta, **complementar** à monotonicidade:

- Curvas contrafactuais de log-LR vs cosseno para raridade P10, P50, P90
- `Δ_log_LR(P90 - P10)` a cosseno fixo
- Intervalos bootstrap por identidade
- Tippett plots estratificados por tipicidade
- Mapas de suporte amostral
- Avisos de extrapolação
- Comparação entre efeito frontal e profile
- Comportamento separado sob H_p e H_d

Esta auditoria responde "como a raridade altera a LR mantendo a similaridade constante?" — pergunta distinta do CONDITIONAL_MONOTONICITY ("aumentar a similaridade reduz a LR?").

---

## 13. Decisões arquiteturais explícitas (registro)

1. **Protocolo + providers** em vez de estender `LRFeatureTermSpec` para piecewise. Razão: acoplamento errado da abstração existente, que foi desenhada para catálogo estático de termos.

2. **V1 cobre LR_ARCH_VEC integralmente; V1.5 cobre piecewise legado integralmente.** Razão: contratos claros melhor que cobertura parcial difícil de explicar.

3. **Dataset resolvido pela proveniência do calibrador.** Razão: descasamento entre dataset auditado e dataset de refit gera auditoria errada sem aviso.

4. **`numerical_epsilon` (1e-12)** vs **`operational_min_slope` (None default)**. Razão: separação clara entre artefato numérico e margem substantiva.

5. **Métricas: `relative_slope_ratio` + `min_slope_x_score_range`**. Razão: duas medidas distintas de "curva quase plana"; comparáveis entre backends com escalas diferentes.

6. **Policy formal por situação desde V1.** Razão: evita discussões post-hoc sobre severidade.

7. **CONDITIONAL_MONOTONICITY e TYPICALITY_EFFECT_AUDIT são regras distintas.** Razão: gate estrutural vs auditoria científica de modulação por raridade — perguntas diferentes.

---

## 14. Limitação conceitual declarada

A regra `CONDITIONAL_MONOTONICITY` verifica:

```
∂ log_LR / ∂ s > 0   sobre os domínios auditados
```

Ela **não** responde:
- O efeito da raridade está na direção correta?
- A magnitude da modulação é adequada?
- Faces raras recebem LRs adequadas?
- Há extrapolação agressiva nas caudas?

Uma arquitetura pode ser perfeitamente monotônica e ainda assim ignorar tipicidade ou modular na direção errada. Por isso `CONDITIONAL_MONOTONICITY` é tratada como **gate necessário, não suficiente**. A próxima auditoria — `TYPICALITY_EFFECT_AUDIT` — responde essas perguntas e deve caminhar **ao lado** desta, não no lugar.
