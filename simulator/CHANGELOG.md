# Changelog

## 0.3.0

- Modo de amostragem intra-classe `IntraClassSamplingMode`: `VMF` (atual, padrão),
  `IDPERTURB_CAP` (calota esférica com cosseno uniforme, Boutros et al. CVPR 2026) e
  `HYBRID_VMF_CAP` (corpo vMF com piso rígido de calota).
- `sample_spherical_cap` (`ṽ = s·μ + √(1−s²)·u`, `s ~ U[lb,1]`) e `sample_hybrid_vmf_cap`
  (vMF truncado por rejeição com fallback de calota + diagnósticos de reamostragem).
- Modulação completa do piso `lb`: base + offset por domínio (`cap_lower_bound_offset`) +
  qualidade + penalidades não-monotônicas de norma, reusando os switches de `CouplingConfig`.
- Regra do bissetor (`cap_overlap_guard`): piso anti-colisão por densidade local dos centros
  (`floor = cos(ângulo_ao_vizinho/2)`), calculado em blocos (sem matriz C×C cheia).
- Colunas de metadata: `intra_class_sampling_mode`, `cap_lower_bound_effective`,
  `pre_nuisance_center_cosine`, `hybrid_resample_count`, `hybrid_used_cap_fallback`.
- Perfil `IDPERTURB_CAP` (`make_idperturb_cap_config`).
- Round-trip de configuração `config_from_dict` / `config_from_manifest` (fecha o
  `config_to_dict` que só tinha o lado de ida).
- Adapter concreto FaceAudit (`faceaudit_concrete_adapter`) para `EmbeddingInfo`/
  `EmbeddingDataset` reais (float32, parquet, cross-condition).
- Testes para calota, modulação de `lb`, guarda do bissetor, híbrido, invariantes de
  pré/pós-nuisance e golden VMF (saída VMF inalterada vs 0.2.0).

## 0.2.0

- Separação entre população latente de identidades e amostragem de observações.
- Geração de datasets cross-condition alinhados por identidade.
- Streams aleatórios determinísticos independentes da ordem das chamadas.
- Seeds explícitas por população, split e condição.
- Modos de nuisance: `global_shared`, `per_domain` e `hierarchical`.
- Perturbações específicas por identidade com base de baixo posto.
- Viés médio de domínio, viés domínio × identidade e viés de sessão.
- Atualização tangencial por retração ou mapa exponencial.
- Diagnósticos da norma das perturbações tangenciais.
- Chaves de ablação para separar os caminhos qualidade → regime, qualidade → norma, qualidade → direção e norma → direção.
- Nome explícito `reference_scaled_log_norm_z`, preservando `robust_norm_z` como alias legado.
- Classe `RadialReference` para ajuste robusto por mediana e MAD escalado.
- Perfis `SPHERICAL_BASELINE`, `COOPERATIVE_PLDA_BASELINE` e `CROSS_CONDITION_HETEROSCEDASTIC`.
- Amostragem vetorizada de pares com estratégias `balanced_identity` e `balanced_observation`.
- Opção `unique_pairs` e aviso quando trials duplicados são removidos.
- Aviso e contadores explícitos quando `pandas.qcut()` produz menos bins do que solicitado.
- Persistência de bundle com `.npz`, CSV de metadados, CSV de diagnóstico e manifesto JSON.
- Adapter genérico de referência para criação de DTOs `EmbeddingInfo` e `EmbeddingDataset` do FaceAudit.
- Testes automatizados para geometria, vMF, regimes, reprodutibilidade, cross-condition, pares e adapter.

## 0.1.0

- Primeira versão do simulador hierárquico AdaFace-like.
