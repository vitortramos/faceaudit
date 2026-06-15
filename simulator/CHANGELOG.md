# Changelog

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
