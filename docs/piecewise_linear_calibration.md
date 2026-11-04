# Calibração piecewise-linear de razão de verossimilhança

Este documento descreve a calibração piecewise-linear implementada no
FaceAudit como extensão interna do calibrador de regressão logística.
Cobre o escopo do componente, a fundamentação forense, a formulação
matemática, as opções de configuração disponíveis na ferramenta, os
diagnósticos persistidos, considerações sobre uso operacional e
limitações conhecidas.

---

## 0. Escopo e status

### 0.1 O que está implementado

Os elementos descritos neste documento estão presentes no código:

- Expansão de basis piecewise-linear interna ao calibrador, ativada
  via `CalibrationBasisMode.PIECEWISE_LINEAR`.
- Estimação do ponto âncora $s_0$ por LogReg auxiliar univariado
  fold-aware (`HingeAnchorMode.AUTO_LOGREG`) ou valor fixo (`MANUAL`).
- Escala dos offsets em score bruto (`KnotScale.RAW`) ou em log-LR do
  calibrador auxiliar (`KnotScale.AUX_LOG_LR`).
- Posicionamento manual (`KnotPlacement.MANUAL`, default operacional)
  ou automático experimental (`KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS`,
  antigo `AUTO_FROM_MEDIANS`).
- Seleção de hiperparâmetro $C$ por `GridSearchCV` sob restrição de
  monotonicidade — candidatos cuja curva final é não-monotônica são
  descartados; selecionado o melhor $C$ entre os válidos por CV score.
- Função objetivo do CV: `neg_cllr_balanced` (padrão V2.6) — equivalente
  a −Cllr balanceado por classe (§9.3).
- Splitter do CV agrupado por identidade do probe (`StratifiedGroupKFold`,
  default V2.8 quando o input tem identidades disponíveis). Fallback
  `StratifiedKFold` por par quando identidades ausentes (§4.2).
- Splitter dinâmico opt-in `IdentityDisjointPairKFold` (V2.10) —
  particiona identidades em $K$ grupos balanceados via LPT
  scheduling; em cada fold, train e test são totalmente disjuntos
  por identidade e pares boundary (uma id dentro, outra fora)
  ficam excluídos apenas daquele fold. Refit final em todo
  $D_{\text{cal}}$ (§4.4). Mapeamento explícito identidade → fold
  persistido como `cv_strict_id_to_group_mapping` (V2.11.1) para
  reprodução exata. **Suporte multivariado (V2.13)**: o splitter é
  dimension-agnostic; a partir desta versão `STRICT_IDENTITY` é
  aplicado nativamente também quando o calibrador recebe
  $X \in \mathbb{R}^{N \times D}$ com $D > 1$ (fluxo
  `SCORE_VECTORIZATION` com múltiplas features), sem cair em
  fallback para `GROUPED_PROBE` (§4.4).
- Standardização per-fold no branch linear (V2.13). Quando
  `standardize_features=True` e $D > 1$, a `StandardScaler` é
  embutida na `sklearn.Pipeline`
  `[StandardScaler → LogisticRegression]` que serve de estimator
  do `GridSearchCV`; μ/σ por feature são aprendidos por fold do
  CV, sem leakage do test fold para o train fold. O refit final
  em $D_{\text{cal}}$ completo popula
  `self._feature_mean` / `self._feature_std` para inferência
  consistente. Em 1D ou com `standardize_features=False`, nada
  muda. Aplica-se ao branch piecewise-linear desde a V1 (a scaler
  já estava dentro da pipeline `PWL → Scaler → LogReg` em §3.4).
- Clusters de lookalikes como unidade atômica no LPT scheduling
  (V2.11.2): identidades de um cluster declarado em
  `metadata["lookalike_clusters"]` ficam no mesmo fold,
  preservando carga LPT agregada. Validação empírica em §4.4
  mostra forte sensibilidade à qualidade dos clusters declarados.
- Diagnósticos de estabilidade entre folds do CV (`cv_*` em §7.11).
- Validação pós-fit de crescimento estrito por trecho, com política
  configurável (`MonotonicityPolicy.ERROR` ou `WARNING`).
- Algoritmo geral para identificar o ponto de cruzamento efetivo de
  log-LR = 0 por segmento.
- Persistência de diagnósticos para auditoria (§7).
- Testes unitários do splitter `IdentityDisjointPairKFold` em
  `tests/test_identity_disjoint_pair_kfold.py` (V2.12, 25 testes):
  validações do construtor, balanceamento LPT vs round-robin,
  disjunção identitária por fold, contagens dinâmicas,
  fold_health, reprodutibilidade, atomicidade de clusters de
  lookalikes, descarte silencioso de identidades órfãs em
  clusters declarados.

### 0.2 Princípios metodológicos adotados

Decisões de design que orientam o componente, antes de qualquer
detalhe de implementação:

- **Avaliação contínua da força da evidência.** Score contínuo →
  log-LR contínuo, sem etapa binária de classificação match /
  non-match (Morrison et al. 2016, §1.4 deste documento).
- **Restrição de monotonicidade na escala bruta.** Sob a política
  default, a curva deve ser estritamente crescente em todos os
  segmentos parametrizados, incluindo nas caudas extrapoladas
  além do range observado; violações de monotonicidade
  (slope ≤ 0 em qualquer trecho) são rejeitadas como inviáveis
  (§3.3, §9.3).
- **Distinção entre LR bruta do modelo e LR reportável.** O
  calibrador estima; a camada operacional decide o que reporta.
  Tetos conservadores aplicados como desigualdade, nunca como
  conclusão binária (§0.4).
- **Fold-awareness sem leakage na estimação do âncora.** Cada fold
  do GridSearchCV aprende seu próprio $s_0$ e knots, evitando
  contaminação cruzada (§4.1).
- **Disjunção por identidade entre treino e validação operacionais.**
  Splits via `PAIRED_DATASET_SPLIT` em vez de `StratifiedKFold` por
  par (§4.2).

### 0.3 O que é estabelecido vs. experimental

| Componente | Status |
|---|---|
| Calibração por regressão logística (linear) | Estabelecido na literatura forense (Brümmer & du Preez 2006, Morrison 2013). |
| Expansão em funções de base lineares por trechos (*piecewise-linear basis expansion*) | Estabelecido em estatística (Hastie, Tibshirani & Friedman 2009, §5.2). |
| Validação pós-fit de crescimento estrito por trecho | Política operacional implementada no FaceAudit. |
| Heurística `AUTO_FROM_MIRRORED_MEDIANS` para posicionamento dos knots | Heurística experimental proposta no projeto. Não consta da literatura citada. Deve ser validada externamente no domínio de aplicação. |

### 0.4 LR bruta do modelo vs. LR reportável

O calibrador piecewise produz um valor de log-LR bruto a partir do
`decision_function` da regressão logística. Esse valor é a saída
matemática não censurada do componente.

Decisões sobre **reportabilidade** — limites superiores e inferiores
aceitos em laudo, política de exibição em casos onde o score
extrapola o range de treino, e relato qualitativo da força da
evidência — constituem uma camada operacional separada. O
componente piecewise não define essa política; ele se limita a
produzir o estimador. A guideline ENFSI para relatórios avaliativos
enfatiza transparência, robustez e disponibilidade para auditoria da
metodologia de atribuição do LR (ENFSI 2015).

A política de reportabilidade pode aplicar tetos conservadores ao
log-LR bruto quando a validação não sustentar reportar magnitudes
arbitrariamente grandes. **A forma correta de aplicar esses tetos é
como desigualdade**, preservando informação sobre a direção e o
limite empírico — não como substituição silenciosa por um valor
pontual censurado:

| Forma operacional | Aceitável? |
|---|---|
| `raw_model_log_lr = +21`, `reportable_display = "log-LR ≥ +12"` (ou equivalentemente "LR ≥ e¹² ≈ 162 000") | ✓ Compatível |
| `raw_model_log_lr = +21`, `reportable_log_lr = +12` (substituição silenciosa) | ✗ Engana o consumidor sobre a magnitude estimada |
| `raw_model_log_lr > 0` → "match" | ✗ Dicotomização criticada por Morrison et al. (2016) |

A primeira forma é compatível com a §1.4: a evidência continua
graduada (estima-se um **limite inferior** sustentado pela
validação), apenas com a magnitude exata censurada. A segunda
distorce a estimativa pontual ao apresentar o teto como se fosse o
log-LR efetivo. A terceira reintroduz a dicotomização criticada.

Recomenda-se persistir separadamente os campos `raw_model_log_lr`
e (quando aplicável) `reportable_log_lr_lower_bound` /
`reportable_log_lr_upper_bound`, com o consumidor da API
escolhendo a forma de apresentação.

### 0.5 Relação com `LR_ARCHITECTURE_VECTORIZATION` (V2.14)

A V2.14 introduziu uma instrução declarativa irmã —
`LR_ARCHITECTURE_VECTORIZATION` — que aplica o mesmo padrão arquitetural
descrito aqui (sklearn `Pipeline` envolvendo um transformer fold-aware
seguido de `StandardScaler` + `LogisticRegression`, com `GridSearchCV`
sobre `c_grid`) mas para um **escopo distinto**: as 15 arquiteturas
LR forenses declarativas que envolvem score base + raridade (9
two-face, do `COSINE_ONLY` ao `BILATERAL_FULL_CENTERED`, e 6 da
família `*_ONE_FACE`, que usam a raridade de uma única face). As duas integrações são
ortogonais e podem coexistir:

| Componente | Domínio | Transformer aprendido por fold | Output dim |
|---|---|---|---|
| `PiecewiseLinearScoreTransformer` (este doc) | calibração não-linear de **um** score 1D em log-LR via dobras hinge | $s_0$ (âncora da dobra) | 1D → $K+1$ features |
| `ArchitectureVectorizerTransformer` (V2.14) | montagem de features de **interação** score × raridade declarativa | $s_0$ (evidência neutra) + $\bar r_F$, $\bar r_P$ (centragem) | $(N, K)$ raw → $(N, D)$ vetorizado |

Ambos compartilham os mesmos princípios de implementação: fit fold-aware
no `Pipeline.fit` para eliminar leakage, `StandardScaler` per-fold sem
vazamento, refit final em $D_{\text{cal}}$ completo via
`GridSearchCV(refit=True)`, splitter `IdentityDisjointPairKFold` quando
identidades disponíveis. A escolha entre os dois caminhos é por
escopo do problema, não competitiva — em particular, a piecewise-linear
permanece a forma canônica de **lidar com não-linearidade local em um
único score**, enquanto a `LR_ARCHITECTURE_VECTORIZATION` cobre o
**casamento explícito de raridade e similaridade**. Documentado em
`CLAUDE.md` (seção "Arquitetura LR declarativa") e
`docs/calibracao_faceaudit.md` §4.1.9.

---

## 1. Fundamento forense

### 1.1 Proposições e LR baseada em score

A apresentação avaliativa de evidência em comparação facial é
formulada em termos de duas proposições mutuamente excludentes:

- $H_p$: as imagens comparadas provêm do mesmo indivíduo.
- $H_d$: as imagens comparadas provêm de indivíduos distintos.

A razão de verossimilhança (LR) avalia a probabilidade dos achados
condicionada a cada proposição:

$$
\text{LR}(s) \;=\; \frac{p(s \mid H_p)}{p(s \mid H_d)}.
$$

Quando $s$ é uma estatística-resumo derivada das imagens (e.g. score
de similaridade entre embeddings), o resultado é uma **LR baseada em
score** (*score-based LR*). A calibração piecewise-linear descrita
aqui é um estimador dessa LR — ela melhora o mapeamento entre score
e log-LR, mas não acrescenta informação ausente do score.

Quando o score contém apenas similaridade, a suficiência da
estatística-resumo para as proposições do caso deve ser demonstrada
empiricamente. Em determinados contextos, scores baseados apenas em
similaridade ou diferença podem ser insuficientes pois não
incorporam a tipicidade da amostra questionada em relação à
população relevante (Morrison & Enzinger, 2018). Informações
adicionais exigem modelagem própria ou fusão multivariada.

### 1.2 Convenção logarítmica

Neste documento, log-LR significa $\ln(\text{LR})$, em nats (logaritmo
natural). Conversões úteis:

| log-LR (nats) | LR aproximada | log10(LR) |
|---|---|---|
| 0 | 1 | 0 |
| $\ln(3) \approx 1{,}10$ | 3 | $\approx 0{,}48$ |
| $\ln(10) \approx 2{,}30$ | 10 | 1 |
| $\ln(100) \approx 4{,}61$ | 100 | 2 |
| $\ln(10^4) \approx 9{,}21$ | $10^4$ | 4 |

A regressão logística do scikit-learn opera em log-odds em base
natural, e o método `decision_function` retorna o logit não censurado.

### 1.3 Estimação por regressão logística

Brümmer & du Preez (2006) e Morrison (2013) propõem a regressão
logística como estimador de log-LR a partir de pares score/rótulo.
Sob certas hipóteses, o modelo

$$
\log \text{LR}(s) \;=\; \beta_z \cdot s \;+\; \beta_0
$$

é consistente quando $P(H_p \mid s)$ é logística em $s$. A regressão
logística é o calibrador padrão na literatura forense de
reconhecimento biométrico (voz, face, etc).

#### 1.3.1 Relação entre log-odds posteriores e log-LR

A regressão logística estima log-odds posteriores efetivos:

$$
\eta(s) \;=\; \log \frac{P_{\text{eff}}(H_p \mid s)}{P_{\text{eff}}(H_d \mid s)}.
$$

Para convertê-los em log-LR é necessário subtrair os log-odds do
prior efetivo no treino:

$$
\log \text{LR}(s) \;=\; \eta(s) \;-\; \log \frac{\pi_{\text{eff}}}{1 - \pi_{\text{eff}}}.
$$

No FaceAudit, o LogReg principal é treinado com
`class_weight="balanced"`, que pondera as amostras de modo que a soma
de pesos por classe seja idêntica. **Na ausência de `sample_weight`
adicionais ou de ponderações específicas por par**, isso resulta em
$\pi_{\text{eff}} = 0{,}5$ e, portanto, $\log(\pi/(1-\pi)) = 0$. Nesse
caso, `decision_function` retorna $\eta(s)$ idêntico a $\log
\text{LR}(s)$. Essa propriedade é persistida no metadata como
`effective_training_prior_hp = 0.5`, `effective_training_prior_hd =
0.5`, `prior_log_odds_correction = 0.0`.

Quando pesos adicionais são introduzidos via `sample_weight` (e.g.
ponderação por raridade de pares, importance sampling para
domain adaptation), o prior efetivo deve ser **calculado a partir da
soma final de pesos** por classe, não presumido como 0,5. A
implementação atual do FaceAudit não introduz `sample_weight` no
calibrador piecewise; alterações futuras desse componente devem
persistir `effective_weight_sum_hp` e `effective_weight_sum_hd` no
metadata e recalcular `prior_log_odds_correction` de acordo.

Configurações de `class_weight` diferentes (e.g. `class_weight=None`
ou dicionário explícito) também exigem revisão dessa relação antes
de tratar `decision_function` como log-LR direto.

### 1.4 Avaliação contínua da força da evidência

A escolha por mapear o score continuamente em log-LR — em vez de
classificá-lo binariamente em "match" ou "non-match" — é uma
decisão metodológica fundamentada e não um detalhe de implementação.
Morrison et al. (2016), em comentário ao relatório do PCAST sobre
métodos forenses de comparação de características, argumentam que
procedimentos de duas etapas (decidir match/non-match e em seguida
informar uma taxa de erro associada à decisão) são **inadequados**
quando os métodos produzem dados continuamente valorados e sujeitos
a variabilidade intrafonte.

O argumento central é o **efeito de precipício** (cliff-edge effect)
introduzido por um threshold operacional. Dado um corte arbitrário
em $s = \tau$:

- Pares com $s = \tau - \varepsilon$ recebem conclusão de não-match,
  enquanto pares com $s = \tau + \varepsilon$ recebem conclusão de
  match — apesar de seus scores serem praticamente indistinguíveis.
- Pares com $s = \tau + \varepsilon$ e $s = \tau + (s_{\max} - \tau)$
  recebem a mesma conclusão de match, apesar de o segundo carregar
  evidência substancialmente mais forte.

Em comparação facial, a variabilidade intrafonte (pose, iluminação,
expressão, idade, sensor, condições de aquisição) introduz dispersão
nos scores de pares Hp. Dicotomizar prematuramente esses scores
descarta informação genuinamente discriminativa.

A alternativa recomendada é converter cada score diretamente em uma
estimativa graduada da força da evidência:

$$
\text{LR}(s) = \frac{p(s \mid H_p)}{p(s \mid H_d)} \quad\text{em escala contínua, para todo $s$ no domínio observado.}
$$

A arquitetura do FaceAudit é coerente com essa recomendação:
embedding → score contínuo → calibrador contínuo → log-LR contínuo.
Nenhum dos calibradores disponíveis (regressão logística linear,
piecewise-linear, KDE, BiGaussianized) precisa declarar match ou
non-match. A expansão piecewise-linear descrita neste documento
preserva essa continuidade: os knots alteram **localmente** a
inclinação da curva de calibração, mas não funcionam como
thresholds classificatórios. Ver §5.1 para a distinção formal entre
o âncora $s_0$ e um threshold de decisão.

---

## 2. Por que uma calibração afim pode ser insuficiente

A linearidade $\log \text{LR}(s) = \beta_z \cdot s + \beta_0$ assume
inclinação constante em toda a faixa do score. Essa hipótese pode
falhar em distribuições reais quando há **regimes distintos** em
faixas diferentes:

- Em modelos como o PLDA, a distribuição $p(s \mid H_d)$ pode ter
  pico numericamente distante do pico de $p(s \mid H_p)$. Em regiões
  onde apenas uma classe tem massa relevante, a razão local de
  densidades muda de comportamento.
- Em regiões com pouca sobreposição entre classes, o LogReg linear
  extrapola a inclinação estimada no centro da distribuição e produz
  log-LRs nas caudas que podem divergir da razão de densidades
  empírica.
- O uso de `class_weight="balanced"` equaliza o gradiente do log-loss
  por classe, mas a flexibilidade na **forma** da curva permanece
  restrita a uma reta.

A literatura compara calibrações lineares e não lineares (Brümmer,
Swart & van Leeuwen 2014) e discute limitações de transformações
afins em pontos operacionais distintos. A calibração piecewise-linear
é uma extensão controlada da regressão logística que admite mudanças
de inclinação em pontos pré-definidos do score, sem perder a
estrutura paramétrica que a torna auditável.

---

## 3. Formulação matemática

### 3.1 Basis expansion piecewise-linear

Seja $s_0 \in \mathbb{R}$ um ponto âncora (definido em §5.1) e
$\delta_1 < \delta_2 < \cdots < \delta_K$ um conjunto de offsets
relativos ao âncora. Define-se a variável centrada $z = s - s_0$ e a
expansão piecewise-linear:

$$
\boldsymbol{\phi}(s) \;=\; \big[\, z, \; \max(z - \delta_1, 0), \; \max(z - \delta_2, 0), \; \ldots, \; \max(z - \delta_K, 0) \,\big] \in \mathbb{R}^{K+1}.
$$

A regressão logística é treinada sobre $\boldsymbol{\phi}(s)$ em vez
de $s$:

$$
\log \text{LR}(s) \;\approx\; \beta_0 \;+\; \beta_z \cdot z \;+\; \sum_{j=1}^{K} \beta_j \cdot \max(z - \delta_j, 0).
$$

Cada função $\max(z - \delta_j, 0)$ é uma **função hinge** (dobradiça)
que vale zero abaixo do knot absoluto $k_j = s_0 + \delta_j$ e cresce
linearmente com inclinação 1 acima dele.

O termo $\beta_z \cdot z$ é incluído explicitamente — todas as
features hinge são interpretadas como modificações da inclinação base
$\beta_z$. Essa estrutura preserva o princípio de marginalidade
(Nelder 1977): o termo linear principal deve estar presente sempre
que termos de interação ou expansão dele estiverem.

### 3.2 Inclinação por trecho

A derivada de $\log \text{LR}$ em relação a $s$ no trecho $j$
(isto é, em $s \in [k_{j-1}, k_j)$, com a convenção $k_0 = -\infty$ e
$k_{K+1} = +\infty$) é:

$$
m_j \;\equiv\; \frac{d \log \text{LR}}{ds}\bigg|_{s \in [k_{j-1}, k_j)} \;=\; \beta_z \;+\; \sum_{i=1}^{j-1} \beta_i.
$$

A inclinação no trecho $j$ é a soma cumulativa de $\beta_z$ com os
coeficientes das hinges que já "ativaram". A curva é um polinômio
piecewise de grau 1 com vértices em $\{k_1, \ldots, k_K\}$.

### 3.3 Validação pós-fit de crescimento estrito

A implementação não impõe restrições durante o ajuste — a regressão
logística é treinada sem restrições e o sistema **valida os slopes
cumulativos após o fit**. Fits que produzem trechos não-crescentes
são rejeitados por default.

Formalmente, exige-se:

$$
m_j \;>\; m_{\min} \quad \forall j \in \{1, \ldots, K+1\},
$$

onde $m_{\min} \geq 0$ é um limiar numérico (`min_allowed_slope`,
default $10^{-8}$). Embora uma função monotônica não-decrescente
possa conter platôs ($m_j = 0$), o FaceAudit adota crescimento
estrito como política operacional para evitar trechos sem
discriminação e instabilidades numéricas.

Quando a regressão logística sobre $\boldsymbol{\phi}(s)$ viola essa
restrição (tipicamente em trechos com pouco suporte estatístico, onde
o $\beta_j$ pode ficar fortemente negativo), o calibrador é rejeitado
com diagnóstico detalhado por trecho (§7.4).

### 3.4 Standardização interna

O calibrador no FaceAudit é treinado via `sklearn.Pipeline` composto
por: `PiecewiseLinearScoreTransformer` → `StandardScaler` →
`LogisticRegression`. A `StandardScaler` aprende $\mu_j, \sigma_j$ das
features expandidas em treino e standardiza:

$$
\tilde\phi_j(s) \;=\; \frac{\phi_j(s) - \mu_j}{\sigma_j}.
$$

Os coeficientes aprendidos $\beta_j^{\text{std}}$ vivem na escala
standardizada. Para auditoria e cálculo de inclinações na escala
bruta do score, recupera-se:

$$
\beta_j^{\text{raw}} \;=\; \beta_j^{\text{std}} / \sigma_j, \qquad
\beta_0^{\text{raw}} \;=\; \beta_0^{\text{std}} \;-\; \sum_j \beta_j^{\text{std}} \cdot \mu_j / \sigma_j.
$$

Esta conversão é feita após o `fit` e os coeficientes em escala bruta
são persistidos no metadata.

### 3.5 Ponto de cruzamento efetivo de log-LR = 0

O âncora $s_0$ é o ponto onde o LogReg auxiliar linear cruza log-LR =
0 (§5.1). Em geral, **não coincide** com o zero do calibrador
piecewise final, porque o piecewise tem inclinações diferentes do
auxiliar e intercepto efetivo distinto.

O zero do piecewise é determinado segmento por segmento. Em cada
segmento $r$, seja $A_r = \{j : \delta_j < z\}$ o conjunto de hinges
ativas no intervalo. A função pode ser escrita como

$$
f_r(z) \;=\; c_r \;+\; m_r \cdot z,
$$

com

$$
m_r \;=\; \beta_z \;+\; \sum_{j \in A_r} \beta_j, \qquad
c_r \;=\; \beta_0^{\text{raw}} \;-\; \sum_{j \in A_r} \beta_j \cdot \delta_j.
$$

A raiz candidata no segmento é $z_r = -c_r / m_r$. A raiz é aceita
apenas quando $z_r$ pertence ao intervalo do segmento. Sob
crescimento estrito ($m_r > 0$ para todo $r$), existe no máximo uma
raiz válida. O valor persistido é

$$
s_{LR = 0} \;=\; s_0 + z_r,
$$

acompanhado do índice do segmento onde foi encontrado
(`effective_log_lr_zero_crossing_segment_index`) e de uma flag
indicando se cai dentro do range observado de treino
(`effective_log_lr_zero_crossing_inside_training_range`).

> **Nota histórica.** Versões anteriores do componente reportavam
> $s_{LR=0} = s_0 - \beta_0^{\text{raw}} / m_{K+1}$, fórmula correta
> apenas no caso degenerado de zero knots ou knots todos em $\delta =
> 0$. O algoritmo geral por segmento foi introduzido em maio de 2026.

---

## 4. Treinamento sem leakage

### 4.1 LogReg auxiliar e knots dentro do `Pipeline`

O `PiecewiseLinearScoreTransformer` é um transformer
`sklearn`-compatível inserido em um `sklearn.Pipeline`. Ao ser
chamado dentro de `GridSearchCV`, seu `fit(X_train, y_train)` é
re-executado em cada fold com apenas os dados de treino do fold:

- Estimação do âncora $s_0$ via LogReg auxiliar (§5.1) ocorre **dentro
  do fit do transformer**. Cada fold tem seu próprio $s_0$ estimado
  apenas com seus dados de treino.
- Conversão `aux_log_lr → raw` em `KnotScale.AUX_LOG_LR` usa o
  $\alpha_{\text{aux}}$ daquele fold.
- Cálculo dos knots em `KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS` (§6.2) usa
  medianas calculadas apenas no subset de treino do fold.

Nenhuma dessas operações lê dados do fold de validação. A propriedade
fold-aware é garantida pela própria mecânica do `Pipeline` (ver
documentação do scikit-learn sobre `Pipeline` e `GridSearchCV`).

### 4.2 Agrupamento por identidade

A propriedade fold-aware acima previne leakage de $s_0$ e dos knots
entre treino e validação **do fold**, mas não previne leakage por
correlação entre pares. Em comparação biométrica, múltiplos pares
podem compartilhar a mesma identidade (e.g. par A=face1×face2 e par
B=face1×face3 compartilham face1). Se folds forem construídos por par
(via `StratifiedKFold` padrão), face1 pode aparecer simultaneamente
no treino e na validação de um mesmo fold.

Há uma dificuldade adicional específica a datasets de pares: cada
**par impostor contém duas identidades distintas**. O
`StratifiedGroupKFold` do scikit-learn aceita apenas um `group` por
linha; atribuir `groups = probe_identity` impede leakage pela
identidade probe, mas não evita que a identidade de referência
apareça em outro fold. Atribuir `groups = unordered_pair(id_a, id_b)`
evita repetir o mesmo par mas ainda permite que cada identidade
apareça em múltiplos folds via pares diferentes.

O protocolo correto é **particionar identidades antes da geração dos
pares**:

1. dividir o universo de identidades em partições disjuntas;
2. manter todas as capturas de cada identidade na mesma partição;
3. gerar pares genuínos e impostores **exclusivamente dentro de cada
   partição** — nunca entre identidades de partições diferentes;
4. em comparações cross-condition (e.g. frontal × lateral), dividir
   simultaneamente os datasets das duas condições preservando
   correspondência de identidade;
5. persistir e verificar empiricamente que não há identidades
   compartilhadas entre partições ($|V \cap C| = 0$).

No FaceAudit, a instrução `PAIRED_DATASET_SPLIT` (§8.4) implementa
essa propriedade canonicamente: divide pares (frontal, lateral) por
identidade, com `grouping_strategy = PRESERVE_LOOKALIKES` mantendo
clusters de identidades **previamente identificadas** como lookalikes
no mesmo subset — reduzindo o risco associado a esses clusters
conhecidos, sem garantir que todos os lookalikes reais tenham sido
detectados.

Quando pares já estiverem materializados em um
`SimilarityScoreDataset`, o calibrador detecta automaticamente as
identidades injetadas pelo executor via `self.metadata`
(`train_probe_identities` / `train_reference_identities`, V2.5) e
escolhe o splitter:

- **`StratifiedGroupKFold`** com `groups = probe_identity` quando
  as identidades estão disponíveis (V2.8). O probe é disjunto entre
  folds; pares Hp ficam totalmente disjuntos por identidade. Pares
  Hd têm probe disjunto, mas a identidade de **referência** ainda
  pode aparecer em múltiplos folds. O metadata registra
  `cv_identity_disjointness_level = "probe_only"`,
  `cv_identity_disjointness_verified = False`.
- **`IdentityDisjointPairKFold`** quando o config seleciona
  `STRICT_IDENTITY` (V2.10). Folds totalmente disjuntos por
  identidade; pares boundary excluídos apenas no fold corrente
  (§4.4). Metadata registra
  `cv_identity_disjointness_level = "full"`,
  `cv_identity_disjointness_verified = True`.
- **`StratifiedKFold`** quando as identidades não estão disponíveis
  (caso raro: input dict sem identidades, ou executor anterior à V2.5).
  Metadata registra `cv_identity_disjointness_level = "none"`,
  `cv_identity_disjointness_verified = False`.

**Validação empírica V2.8.** No holdout BFW split B (pipeline PLDA
SHRINK CROSS), a troca para `StratifiedGroupKFold` produziu o
seguinte efeito na seleção interna de $C$ e no $C_{\text{llr}}$
operacional:

| Calibrador | V2.7 (par) | V2.8 (grouped) | Δ $C_{\text{llr}}$ |
|---|---|---|---|
| Linear puro | 0,04739 (C=0,001) | 0,04739 (C=0,001) | 0 % |
| MANUAL 7-KNOT (`[−15, −12, −9, −6, −3, 0, +6]`) | 0,04234 (C=0,1) | **0,04126** (C=**0,01**) | **−2,6 %** |
| AUTO 2 knots | 0,04654 (C=0,1) | 0,04654 (C=0,1) | 0 % |
| BI GAUSS | 0,05965 | 0,05965 | 0 % (não usa GridSearch) |

A configuração MANUAL 7-KNOT foi a única onde o ranking do CV mudou ao
remover o leakage parcial — o splitter agrupado preferiu
regularização mais forte (C=0,01) e essa escolha generalizou
melhor para o holdout deste split específico. Linear puro já
estava em C extremamente regularizado; AUTO 2 knots não dependeu
de correlações entre pares no C ótimo deste experimento. O
comportamento é consistente com a hipótese de que o CV por par
superestimou a capacidade de generalização da configuração menos
regularizada. A análise de robustez em §4.5 (bootstrap por
identidade, 200 réplicas com multiplicidades preservadas)
corrobora empiricamente a vantagem de GROUPED sobre PAIR neste
corpus (PAIR perde em 56 % das réplicas vs 2 % de wins, mediana de
$\Delta C_{\text{llr}}$ = +0,0011). A adoção do splitter agrupado
como default V2.8 é sustentada por essa evidência empírica + a
motivação metodológica original; sua replicação em outros corpora
permanece desejável antes de generalização ampla.

**Limitação residual da V2.8.** O probe é disjunto entre folds,
mas a identidade de referência dos pares Hd ainda pode cruzar —
um par Hd $(p, r)$ tem **duas** identidades; agrupar só por
$p$ não impede que $r$ apareça em folds distintos. A V2.9 (§4.3)
tentou eliminar esse leakage residual com um filtro global de Hd;
a V2.10 (§4.4) substituiu a abordagem por máscaras dinâmicas por
fold + refit final em D_cal cheio.

### 4.3 V2.9 (histórica): filtro global estrito por identidade — substituída

> **Status**: substituída pela V2.10 (§4.4). Esta seção permanece
> como registro da evolução do componente.

Quando ativada (opt-in via metadata na V2.9), a V2.9 substituía o
`StratifiedGroupKFold` por uma construção `PredefinedSplit` com
filtragem global:

1. **Partição prévia das identidades**: identidades únicas (probe ∪
   reference) embaralhadas com `random_state=42` e distribuídas
   round-robin em $K$ grupos disjuntos.
2. **Filtragem global**:
   - Pares Hp ($p = r$) ficavam no grupo de $p$;
   - Pares Hd intra-grupo ficavam no grupo comum;
   - Pares Hd **cross-grupo** eram **descartados** globalmente do
     treino, do CV e do refit final.
3. **`PredefinedSplit`** sobre o subset filtrado.

**Limitação identificada da V2.9.** No exemplo BFW, 53 332 de 64 620
pares Hd cross-grupo (≈82 %) eram descartados permanentemente — o
conjunto de calibração efetivo caía de 66 339 para 13 007 pares
(≈19,6 %). Isso confundia dois efeitos: redução de leakage e perda
de amostra. A própria curva calibrada final era treinada apenas
sobre os pares retidos, prejudicando inferência.

### 4.4 V2.10 (atual): folds estritos dinâmicos + refit em D_cal cheio

A V2.10 corrige o problema acima separando explicitamente
**seleção de hiperparâmetro** (sob disjunção identitária por fold)
de **fit final** (sobre todo o D_cal):

1. **Partição balanceada das identidades** em $K$ grupos via LPT
   scheduling (Graham 1969): identidades ordenadas por carga
   (somatório de aparições probe + reference) e atribuídas ao
   grupo com menor carga acumulada. Reduz dispersão dos folds
   quando identidades têm número desigual de capturas
   (no BFW, range de train_counts cai de 43 → 3).
2. **Máscaras dinâmicas por fold** (splitter customizado
   `IdentityDisjointPairKFold`):

   ```python
   for test_group in identity_groups:
       test_mask  = probe.isin(test_group) & ref.isin(test_group)
       train_mask = ~probe.isin(test_group) & ~ref.isin(test_group)
       # boundary (uma id dentro, outra fora): excluído deste fold
       # mas elegível em outros folds.
       yield np.where(train_mask)[0], np.where(test_mask)[0]
   ```

   Cada fold preserva ≈ $((K-1)/K)^2$ do conjunto Hd no treino
   (≈64 % com $K=5$) em vez dos ≈1/K do filtro global da V2.9.
3. **Refit final em todo $D_{\text{cal}}$** após a seleção de $C$
   — o splitter governa apenas a iteração de folds do
   `GridSearchCV`; o modelo operacional é treinado sobre o
   conjunto completo, incluindo pares boundary que foram
   excluídos de cada fold individual.

Disjunção total verificada: para cada fold,
$\text{identidades(train)} \cap \text{identidades(test)} = \emptyset$.

**Clusters de lookalikes como unidade atômica (V2.11.2).** Quando
o usuário injeta `metadata["lookalike_clusters"]` como uma lista
de listas de identidades, o splitter trata cada cluster como uma
**super-identidade** durante a atribuição a grupos — todas as
identidades de um cluster ficam no mesmo fold, com carga LPT
agregada. Isso alinha o CV interno ao princípio do
`PAIRED_DATASET_SPLIT` com `PRESERVE_LOOKALIKES` (§8.4): pares Hd
entre duas identidades de um cluster ficam sempre no mesmo fold,
evitando que a similaridade visual entre lookalikes seja "vazada"
do treino para o teste. Identidades não declaradas em nenhum
cluster funcionam como singletons. Metadados:
`cv_strict_lookalike_clusters_used`,
`cv_strict_n_lookalike_clusters`,
`cv_strict_n_clustered_identities` (§7.10).

> **Validação empírica preliminar (V2.12).** Em corpus BFW (36
> identidades), uma heurística sintética de detecção de clusters
> baseada em score médio Hd acima do percentil 95 produziu **1
> cluster gigante de 26 identidades** (BFW é demograficamente
> homogêneo: todos males adultos brancos no split testado, scores
> Hd globalmente similares; P95 captura ruído e não lookalikes
> reais). Em 50 réplicas de bootstrap, o calibrador com esses
> pseudo-clusters injetados **degradou** o $C_{\text{llr}}$
> holdout (mediana 0,0471 → 0,0710, W/T/L 7/1/42 — clusters perdem
> em 84 % das réplicas) e colapsou a seleção de $C$ no extremo de
> regularização ($C = 0{,}001$ em 48/50 reps, vs distribuição
> ampla {0,01..10} sem clusters). **Leitura prática**: a feature
> funciona corretamente — respeita atomicidade, reduz boundary
> excluído de ≈20 k para ≈11 k pares — mas é **muito sensível à
> qualidade dos clusters declarados**. Clusters bem-anotados
> (e.g. pares conhecidos identificados manualmente ou por
> pipeline upstream com supervisão) são esperados ter 2–10
> identidades cada, com cobertura parcial do corpus. Clusters
> mal-definidos que agrupam a maioria das identidades
> **destroem** o calibrador. Recomendação: usar essa feature
> apenas quando há fonte externa confiável de clusters
> (anotações manuais, pipeline de detecção de lookalikes
> independente), nunca com heurísticas automáticas sobre os
> próprios scores de calibração.

**Configuração na V2.10.** Ativada via DTO (não mais via metadata):

```python
LogisticRegressionCalibrationModelConfig(
    ...,
    gridsearch_cv_mode=LogRegGridSearchCVMode.STRICT_IDENTITY,
)
```

Modos disponíveis: `AUTO` (default — escolhe GROUPED_PROBE quando
identidades estão disponíveis, cai em PAIR caso contrário), `PAIR`,
`GROUPED_PROBE`, `STRICT_IDENTITY`. O metadata persiste
`cv_split_mode_requested`, `cv_split_mode_applied` e
`cv_split_mode_fallback_reason` (§7.10).

**Comparativo PAIR / GROUPED / STRICT V2.10.** Holdout BFW split B,
pipeline PLDA SHRINK CROSS, 36 identidades de treino:

| Calibrador | PAIR ($C$, $C_{\text{llr}}$) | GROUPED ($C$, $C_{\text{llr}}$) | STRICT V2.10 ($C$, $C_{\text{llr}}$) |
|---|---|---|---|
| Linear puro | 0,001 ; 0,0474 | 0,001 ; 0,0474 | 0,001 ; 0,0474 |
| 1-KNOT S0 | 10 ; 0,0465 | 0,1 ; **0,0434** | 0,1 ; **0,0434** |
| AUTO 2 knots | 0,1 ; 0,0465 | 0,1 ; 0,0465 | 0,01 ; **0,0446** |

> **Nota sobre comparação com a tabela §4.2**: o calibrador
> **MANUAL 7-KNOT** da §4.2 V2.8 (Cllr 0,04234 → 0,04126) é uma
> configuração piecewise distinta do **1-KNOT S0** desta tabela.
> Os deltas relativos das duas tabelas **não devem ser combinados
> ou comparados diretamente** — são objetos diferentes.

A figura abaixo sobrepõe as três curvas de calibração (mesmo
calibrador piecewise 1-KNOT S0, mesmo train set, mesmo holdout),
variando apenas o splitter do CV interno:

![Comparação V2.7 / V2.8 / V2.10 do calibrador piecewise 1-KNOT S0](images/pwl_splitter_comparison.png)

Leitura: PAIR (vermelho, $C=10$) selecionou regularização fraca e
produziu slopes 0,214 / 0,266. GROUPED e STRICT V2.10 chegaram à
mesma escolha de hiperparâmetro ($C=0{,}1$) e, como o refit final
em ambos os modos é em todo o $D_{\text{cal}}$, produzem curvas
calibradas idênticas (slopes 0,166 / 0,290). No gráfico, as
linhas GROUPED e STRICT se sobrepõem perfeitamente — a
convergência dos dois modos no mesmo $C$ é coerente com a hipótese
de que neste split o leakage de pareamento favoreceu regularização
fraca demais, e tanto o agrupamento por probe quanto a disjunção
total identificaram esse viés. A análise de bootstrap em §4.5
(200 réplicas com multiplicidades preservadas) corrobora a
vantagem de GROUPED sobre PAIR (PAIR perde em 56 % das réplicas,
ganha em 2 %); STRICT_IDENTITY permanece estatisticamente
equivalente a GROUPED_PROBE neste corpus.

**Health checks por fold (V2.10).** O splitter
`IdentityDisjointPairKFold` calcula `fold_health` (persistido em
`cv_strict_fold_health`, §7.10) com critérios mínimos de suporte:
≥ 50 Hp e ≥ 50 Hd no train por fold, ≥ 10 Hp e ≥ 10 Hd no test,
boundary excluído < 50 % do total. `overall_status` ∈ {ok, warning,
critical}. Estes thresholds são **guardrails técnicos** para
detectar folds degenerados, não critérios de suficiência amostral
para uso pericial — a defensibilidade forense depende
adicionalmente de estabilidade dos parâmetros entre folds (§7.11),
cobertura por identidade, análise das caudas (§8.3) e validação
externa.

**Recomendação prática V2.10.**

- `STRICT_IDENTITY` é conceitualmente o modo mais defensável
  quando seus folds preservam suporte suficiente segundo os
  `fold_health` configurados (`overall_status = ok`). Em corpora
  pequenos, mesmo a exclusão dinâmica pode reduzir
  $((K-1)/K)^2 \cdot N_{\text{Hd}}$ a um nível em que o CV se
  torna instável — nesses casos, `GROUPED_PROBE` constitui um
  compromisso pragmático.
- A decisão deve ser baseada em **health checks por fold**
  (`fold_health.per_fold_status`), não em uma regra rígida de
  número absoluto de identidades.
- Para auditoria, persistir `cv_split_mode_requested`,
  `cv_split_mode_applied`, `cv_split_mode_fallback_reason` e
  `cv_strict_fold_health` no laudo permite justificar a escolha.

**Suporte a entradas multivariadas (V2.13).** Até a V2.12, o branch
linear do calibrador rejeitava `STRICT_IDENTITY` em entradas
multivariadas (`X.shape[1] > 1`, fluxo típico de
`SCORE_VECTORIZATION` alimentando o calibrador com múltiplas
features) e caía em fallback automático para `GROUPED_PROBE` — o
guarda foi posto por conservadorismo da V1 e não refletia uma
limitação estrutural. O `IdentityDisjointPairKFold` opera apenas
sobre `train_probe_identities` e `train_reference_identities`,
sendo **dimension-agnostic em X**. A partir da V2.13, o guarda foi
removido: estratégias forenses que combinam, por exemplo, score
cosseno bruto com hinge $h_+(s)$ e interações com raridade
$h_+(s) \cdot \tilde r_q$, $h_+(s) \cdot \tilde r_r$ podem ser
calibradas sob disjunção identitária total. Os mesmos metadados
(`cv_split_mode_applied = "STRICT_IDENTITY"`,
`cv_identity_disjointness_level = "full"`,
`cv_strict_*`) são persistidos no modo multivariado.

**Standardização per-fold (V2.13).** Anteriormente, quando
`standardize_features=True` em multivariada, a `StandardScaler`
era aplicada **antes** do CV — $\mu_j, \sigma_j$ vinham do
$D_{\text{cal}}$ completo, configurando leakage do test fold para
o train fold (a magnitude do leakage é pequena para μ/σ estáveis,
mas viola o princípio do CV). A partir da V2.13, a scaler vai
**dentro** do `sklearn.Pipeline`
`[StandardScaler → LogisticRegression]` que é usado como
estimator do `GridSearchCV`. μ/σ são aprendidos por fold no CV;
no refit final em $D_{\text{cal}}$ completo, a scaler é fitada
sobre todo o conjunto e seus parâmetros são extraídos para
`self._feature_mean` / `self._feature_std` — preservando o
contrato de inferência (`_features_to_X` aplica os mesmos μ/σ em
validação/casework). A modificação se aplica também ao branch
`use_gridsearch=False`; em 1D ou com `standardize_features=False`,
nada muda (μ/σ continuam `None`). O atributo `self.logreg`
permanece apontando para o `LogisticRegression` interno (não para
a Pipeline), preservando o contrato downstream que lê
`calibrator.logreg.coef_` para a denormalização de coeficientes
em escala bruta (executor `ScoreCalibrationInstructionExecutor`,
§7.5).

### 4.5 Análise exploratória de robustez por bootstrap de identidades

Para avaliar se a vantagem do GROUPED sobre o PAIR observada em
§4.2 / §4.4 reproduz sob reamostragem, executamos bootstrap por
identidade no domínio de calibração com o holdout BFW-B fixo.

**Protocolo (V2.11).** 200 réplicas. Em cada réplica:

1. amostra com **reposição** das 36 identidades únicas →
   contagem $m_i$ por identidade $i$;
2. **materialização** do subset preservando multiplicidades: cada
   par $(p, r)$ é replicado $m_p \cdot m_r$ vezes — é o bootstrap
   clássico, e não apenas subamostragem por presença/ausência;
3. ajuste do calibrador 1-KNOT S0 nos três modos (PAIR /
   GROUPED_PROBE / STRICT_IDENTITY V2.10) sobre o subset
   materializado;
4. avaliação no holdout BFW-B fixo (mesmo dataset em todas as
   réplicas);
5. comparação par-a-par com win/tie/loss usando tolerância
   $\varepsilon = 10^{-4}$.

| Modo | Mediana $C_{\text{llr}}$ | IC 95% | $C$ mais frequente |
|---|---|---|---|
| PAIR | 0,0492 | [0,0438, 0,0947] | 1,0 (133/200), 10 (66) |
| GROUPED_PROBE | 0,0474 | [0,0423, 0,0923] | 0,1 (90), 1,0 (77), 10 (30) |
| STRICT_IDENTITY (V2.10) | 0,0475 | [0,0424, 0,0923] | 1,0 (83), 0,1 (75), 10 (36) |

Diferenças par-a-par e contagens W/T/L ($\varepsilon = 10^{-4}$):

| Comparação | $\Delta C_{\text{llr}}$ mediana | IC 95% | Wins | Ties | Losses |
|---|---|---|---|---|---|
| PAIR − GROUPED | **+0,0011** | [−0,0000, +0,0108] | 5 (2 %) | 84 (42 %) | **111 (56 %)** |
| STRICT − GROUPED | +0,0000 | [−0,0073, +0,0079] | 28 (14 %) | 127 (64 %) | 45 (22 %) |
| STRICT − PAIR | −0,0009 | [−0,0097, +0,0024] | — | — | — |

**Leitura central V2.11.**

- **GROUPED_PROBE > PAIR (empírico)**: com 200 réplicas e
  bootstrap preservando multiplicidades, **PAIR perde em 56 % das
  réplicas e ganha em apenas 2 %**, com mediana de
  $\Delta C_{\text{llr}}$ de +0,0011 e limite inferior do IC 95 %
  encostando em zero (−0,0000). Esta é evidência mais consistente
  do que o resultado V2.10 anterior (com 30 reps sem
  multiplicidades, onde a diferença cruzava zero). Confirma —
  empiricamente neste corpus — a justificativa metodológica de
  preferir GROUPED_PROBE sobre PAIR.
- **STRICT_IDENTITY ≈ GROUPED_PROBE**: com mediana de
  $\Delta C_{\text{llr}}$ exatamente zero, IC 95 % amplo
  ([−0,0073, +0,0079]) e 64 % de empates dentro de $\varepsilon$,
  os dois modos são **estatisticamente equivalentes** neste
  corpus. STRICT_IDENTITY não trouxe vantagem empírica
  detectável sobre GROUPED_PROBE com $N_{\text{ids}} = 36$.
- **Distribuição de $C$**: PAIR escolhe $C \ge 1$ em 100 % das
  réplicas (regularização fraca); GROUPED e STRICT distribuem-se
  mais amplamente, com $C = 0{,}1$ presente em 45 % e 38 %
  respectivamente — reflete que o CV agrupado por identidade
  rejeita parte do regime $C$ alto.
- **Hipótese de trabalho**: a preferência por GROUPED_PROBE sobre
  PAIR é consistente com a teoria (eliminar leakage por
  correlação entre pares) e agora corroborada por evidência
  empírica neste corpus. A preferência por STRICT_IDENTITY sobre
  GROUPED_PROBE permanece **conceitual** (disjunção identitária
  total auditável + refit em D_cal cheio), não empiricamente
  diferenciada com $N_{\text{ids}}=36$.

**Limitações deste experimento.**

- **Holdout fixo (BFW-B).** O experimento mede sensibilidade do
  calibrador à composição de $D_{\text{cal}}$, não a incerteza
  total esperada do desempenho operacional. Variar também
  $D_{\text{val}}$ exigiria uma estrutura distinta (e.g. nested
  identity holdout). Foi tentada uma Análise B com partição
  identitária interna do conjunto disponível (70 % $D_{\text{cal}}$ +
  30 % $D_{\text{val}}$), mas os parquets BFW-A (treino) e BFW-B
  (val) já têm identidades disjuntas por construção (36 + 64,
  overlap = 0), tornando essa partição inviável neste corpus.
  Avaliação multi-domain exigiria corpora adicionais.
- **36 identidades** continua sendo um corpus pequeno. Mesmo com
  200 réplicas de bootstrap, o IC superior da distribuição de
  $C_{\text{llr}}$ ([0,094, 0,092]) reflete reamostragens onde
  poucas identidades ficaram representadas, gerando calibradores
  fracos.
- **Replicação cross-domínio**: a vantagem GROUPED > PAIR
  observada aqui precisa ser verificada em outros corpora
  (CFP, BFW completo, corpora com mais identidades) antes de
  generalização.

Metadados persistidos no modo STRICT_IDENTITY estão descritos em
§7.10 (contagens dinâmicas por fold, balance loads,
`fold_health` formalizado).

---

## 5. Âncora e knots

### 5.1 O ponto âncora $s_0$

A expansão piecewise-linear é parametrizada pelo âncora $s_0$. Os knots
absolutos são $k_j = s_0 + \delta_j$. A escolha de $s_0$ define o que
significa "centrar a expansão". Há dois modos.

#### Estimação automática (`HingeAnchorMode.AUTO_LOGREG`)

Um LogReg auxiliar univariado é treinado sobre os mesmos $(s, y)$ do
treino do calibrador:

$$
\log \text{LR}_{\text{aux}}(s) \;=\; \alpha_{\text{aux}} \cdot s + \beta_{\text{aux}}.
$$

O ponto $s_0$ é definido como o $s$ tal que
$\log \text{LR}_{\text{aux}}(s_0) = 0$:

$$
s_0 \;=\; -\beta_{\text{aux}} \,/\, \alpha_{\text{aux}}.
$$

A inclinação $\alpha_{\text{aux}}$ tem interpretação útil: é o fator
de conversão entre score bruto e log-LR linear local. Esse fator é
usado pela escala `AUX_LOG_LR` (§5.2) e pelo posicionamento
`AUTO_FROM_MIRRORED_MEDIANS` (§6.2).

#### Valor fixo (`HingeAnchorMode.MANUAL`)

O usuário fornece $s_0$ diretamente via
`hinge_anchor_manual_value`. Útil para reproduzir calibradores
externos já auditados ou aplicar regras de domínio explícitas. Não
gera $\alpha_{\text{aux}}$ — escalas e modos que dependem dele ficam
restritos.

#### $s_0$ não é um threshold de decisão

Por relevância forense (Morrison et al. 2016, §1.4), vale a
distinção explícita entre os seguintes conceitos:

| Conceito | Significado |
|---|---|
| **$s_0$** (âncora da expansão) | Posição de uma dobra numa função contínua. Acima e abaixo de $s_0$, o log-LR continua sendo definido continuamente. |
| **Threshold binário** ($\tau$) | Corte que converte score em duas categorias (match / non-match). Descartado pelo FaceAudit por princípio. |
| **log-LR $= 0$** | Ponto onde $\text{LR} = 1$; evidência neutra entre $H_p$ e $H_d$. Valor de referência interpretativa, **não** corte de decisão. |
| **Threshold de EER** | Diagnóstico de desempenho discriminativo do modelo, usado em validação. Não usado para classificar resultados de caso. |
| **Decisão judicial ou investigativa** | Camada operacional posterior, externa ao calibrador, dependente de prior odds, custos relativos e regras do domínio jurídico. |

O calibrador produz um valor contínuo em qualquer ponto do domínio
de score. O usuário e o consumidor da LR (e.g. perito, juiz)
operam sobre esse valor graduado. A existência do âncora $s_0$ ou
do cruzamento efetivo (§3.5) não restaura uma estrutura binária.

### 5.2 Escala dos offsets (`KnotScale`)

| Valor | Significado |
|---|---|
| `RAW` | Offsets diretamente em unidades de score. Backward-compat com configs antigos. |
| `AUX_LOG_LR` (default em novos calibradores) | Offsets em unidades de log-LR do calibrador auxiliar (nats). Conversão $\delta_j^{\text{raw}} = \delta_j^{\text{aux\_log\_lr}} / \alpha_{\text{aux}}$ feita no fit. Exige `HingeAnchorMode.AUTO_LOGREG`. |

A escala `AUX_LOG_LR` reduz a dependência do domínio numérico do
score (cosseno em ±1 vs. PLDA em ±100). Não garante, no entanto,
que os knots resultantes caiam dentro do range observado do score:
em casos de $\alpha_{\text{aux}}$ muito pequeno, offsets numéricos
grandes em log-LR auxiliar podem mapear para valores raw fora do
domínio dos dados. O metadata persistido inclui o campo
`knot_inside_training_range` por knot para auditar essa condição.

A escala se chama `AUX_LOG_LR` para evidenciar que vive na escala do
**LogReg auxiliar**, não na curva piecewise final. O log-LR do
piecewise final pode divergir do log-LR auxiliar quando o piecewise
muda de inclinação.

### 5.3 Posicionamento (`KnotPlacement`)

Quem decide a posição dos knots.

| Valor | Significado |
|---|---|
| `MANUAL` | O usuário informa `knot_offsets`. **Padrão operacional em novos calibradores**: `knot_offsets=[0.0]` em `KnotScale.AUX_LOG_LR`, equivalente a uma única dobra em $s_0$. |
| `AUTO_FROM_MIRRORED_MEDIANS` | Heurística experimental que posiciona dois knots a partir das medianas auxiliares refletidas em torno de log-LR = 0 (§6.2). Deve ser habilitada explicitamente. |

Em `AUTO_FROM_MIRRORED_MEDIANS`, os campos `knot_offsets` e `knot_scale` do
DTO são ignorados; o transformer determina ambos automaticamente. A
UI esconde esses campos quando AUTO está selecionado.

---

## 6. Posicionamento dos knots

### 6.1 Modo MANUAL

O usuário escreve `knot_offsets` no DTO. Exemplos com interpretação
clara em `KnotScale.AUX_LOG_LR`:

| `knot_offsets` (nats) | Interpretação |
|---|---|
| `[0.0]` | Um knot em $s_0$ (ponto neutro do auxiliar). |
| `[-ln(10), 0, +ln(10)]` $\approx$ `[-2.30, 0, +2.30]` | Três knots: LR auxiliar 1:10 contra Hp, neutro, e 10:1 a favor de Hp. |
| `[-ln(100), 0, +ln(100)]` $\approx$ `[-4.61, 0, +4.61]` | Três knots em LR auxiliar 1:100, neutro, e 100:1. |

A escolha entre alternativas pré-definidas deve ser feita via
cross-validation interna ou dev set, **nunca pelo $C_{\text{llr}}$
medido no mesmo holdout usado para reportar a métrica final** — esse
procedimento constitui tuning sobre o test set.

A figura abaixo ilustra um calibrador piecewise com
`knot_offsets = [0.0]` (um único knot em $s_0$) treinado e validado
sobre o mesmo domínio (BFW split, com identidades disjuntas entre os
subsets). A curva piecewise (azul) tem dois segmentos com
inclinações `[0.166, 0.291]` — slope mais íngreme acima de $s_0$,
permitindo que scores PLDA mais altos produzam aumento mais rápido
do log-LR. $s_0$ é a âncora estimada pelo calibrador auxiliar como
ponto aproximado de evidência neutra; neste experimento, ela
coincide aproximadamente com a região de transição entre os regimes
predominantes de Hd e Hp (isto não é garantido matematicamente, mas
foi observado neste experimento e é uma hipótese plausível a
investigar em outros scores PLDA — não uma propriedade
matematicamente garantida). A linha
tracejada laranja é a calibração afim ótima a posteriori, com
`slope < 1` indicando sobreconfiança residual da curva piecewise. As barras vermelhas e verdes na
base mostram a distribuição empírica de impostores e genuínos.
Resultado (V2.8 default, `gridsearch_cv_mode = GROUPED_PROBE`,
`neg_cllr_balanced`, `c_grid = [0.001, 0.01, 0.1, 1.0, 10.0]`,
$C$ selecionado $= 0{,}1$):
$C_{\text{llr}} = 0{,}0434$, $C_{\text{llr,affine}} = 0{,}0419$,
$C_{\text{llr,min}} = 0{,}0362$, `segment_slopes_raw = [0,1661, 0,2905]`
(185 797 impostores, 2 483 genuínos no holdout).

![Curva piecewise com 1 knot em s0](images/pwl_calibration_curve_singleknot.png)

### 6.2 Modo AUTO_FROM_MIRRORED_MEDIANS (heurística experimental)

**Status.** `AUTO_FROM_MIRRORED_MEDIANS` é uma heurística automática,
determinística e ajustada separadamente em cada fold do CV,
proposta no FaceAudit para o posicionamento parcimonioso de dois
knots. Não consta da literatura citada. Deve ser tratada como
**protocolo experimental** até que sua estabilidade seja demonstrada
em validação externa representativa do domínio de aplicação.

**Nomenclatura.** A heurística não usa diretamente as medianas
log-LR auxiliares das classes — usa-as **refletidas em torno de
log-LR = 0**:
$\boldsymbol\delta = \mathrm{sorted}(\,[-\hat\ell_p, -\hat\ell_d]\,)$.
Em distribuições aproximadamente simétricas
($\hat\ell_d \approx -\hat\ell_p$), as medianas refletidas coincidem
aproximadamente com as medianas observadas, e a distinção é
cosmética; em distribuições assimétricas, o espelhamento é uma
decisão de regularização não-trivial. A partir da V2.6, o nome
canônico do membro do enum é `AUTO_FROM_MIRRORED_MEDIANS`. O nome
legado `AUTO_FROM_MEDIANS` continua aceito como alias de entrada por
`from_str()` e, para não quebrar JSONs e pickles anteriores, continua
sendo o **valor serializado** do membro (`to_str()` devolve
`"AUTO_FROM_MEDIANS"` no JSON da config e em `piecewise_knot_placement`).
O que distingue os artefatos novos é
`knot_placement_protocol_version = "auto_from_mirrored_medians_v1"`.

**Default operacional.** Desde a V2.4, o construtor
`PiecewiseLinearCalibrationConfig` usa `KnotPlacement.MANUAL` com
`knot_offsets=[0.0]` em `KnotScale.AUX_LOG_LR` como padrão — a
configuração mais simples e estabelecida, equivalente a uma dobra
única no ponto âncora $s_0$. A escolha refletiu a evidência empírica
reunida neste documento (§6.2 abaixo e §8): no mesmo split, a dobra
única produziu Cllr menor que `AUTO_FROM_MIRRORED_MEDIANS`. O modo
automático deve ser habilitado explicitamente como **opt-in
experimental** quando o usuário queira testá-lo num domínio
específico.

#### Definição

A heurística posiciona dois knots em

$$
\boldsymbol\delta \;=\; \mathrm{sorted}\big(\big[\, -\, \hat\ell_p, \;\; -\, \hat\ell_d \,\big]\big),
$$

onde $\hat\ell_p$ e $\hat\ell_d$ são as medianas log-LR auxiliares de
cada classe estimadas no fold de treino:

$$
\hat\ell_p \;=\; \alpha_{\text{aux}} \cdot (\text{med}(s \mid y = 1) - s_0), \qquad
\hat\ell_d \;=\; \alpha_{\text{aux}} \cdot (\text{med}(s \mid y = 0) - s_0).
$$

A fórmula não exige treinar um segundo calibrador completo nem ter um
dev set separado; usa o próprio LogReg auxiliar já treinado para
$s_0$.

#### Intuição geométrica

Cada knot em $-\hat\ell_p$ ou $-\hat\ell_d$ posiciona-se no espelho em
torno de zero da mediana log-LR da classe correspondente. A heurística
pretende aproximar uma **região intermediária de transição** entre os
regimes de evidência. Essa interpretação é intuitiva mas não constitui
garantia matemática de que o intervalo entre knots corresponda
exatamente à região de sobreposição das classes.

#### Fold-awareness

Em `AUTO_FROM_MIRRORED_MEDIANS`, o cálculo das medianas acontece dentro do
`PiecewiseLinearScoreTransformer.fit`, executado uma vez por fold do
`GridSearchCV`. Cada fold tem seu próprio $\alpha_{\text{aux}}, s_0,
\hat\ell_p, \hat\ell_d$ e, portanto, seus próprios knots, computados
estritamente sobre seus dados de treino. Não há leakage dos folds de
validação para a definição dos knots. (Subsiste o risco de leakage por
correlação entre pares — ver §4.2.)

#### Exemplo de curva treinada por AUTO_FROM_MIRRORED_MEDIANS

A figura abaixo mostra um calibrador piecewise produzido por
`AUTO_FROM_MIRRORED_MEDIANS` sobre o mesmo dataset usado em §6.1. Dois knots
foram posicionados nos espelhos das medianas log-LR auxiliares de
cada classe (em $-89{,}19$ e $+13{,}68$ na escala bruta), gerando
três segmentos com inclinações `[0.077, 0.221, 0.319]`. A curva
permaneceu estritamente crescente. O trecho à direita do segundo
knot, entretanto, possui **cobertura assimétrica por classe**: há
genuínos, mas aparentemente nenhum impostor observado no treino. A
inclinação positiva desse trecho deve ser interpretada com cautela,
pois fica fracamente identificada pelos dados locais e depende mais
fortemente da continuidade paramétrica da curva, dos demais trechos
e da regularização L2 (`C = 0.1` foi suficiente neste fit). Ver
§10.2 e §9.3.

![Curva piecewise AUTO 2 knots](images/pwl_calibration_curve_autoknots.png)

#### Avaliação empírica comparada à dobra única em $s_0$

No mesmo split (BFW-A → BFW-B, mesmo pipeline PLDA SHRINK CROSS,
scorer `neg_cllr_balanced` V2.6), as duas configurações produziram:

| Posicionamento | $C_{\text{llr}}$ | $C_{\text{llr,affine}}$ |
|---|---|---|
| Uma dobra em $s_0$ (`MANUAL`, `knot_offsets=[0.0]`) | **0,0434** | **0,0419** |
| Dois knots refletidos (`AUTO_FROM_MIRRORED_MEDIANS`) | 0,0465 | 0,0437 |

Com `gridsearch_cv_mode = GROUPED_PROBE` (V2.8 default) e c_grid
amplo, a dobra única em $s_0$ produziu Cllr 0,003 menor que o modo
automático neste experimento — a complexidade adicional dos dois
knots não se traduziu em ganho operacional. O modo automático
apresenta $C_{\text{llr,affine}}$ marginalmente pior ($0{,}0437$ vs $0{,}0419$),
indicando que a forma da curva está ligeiramente mais distante da
forma ideal a posteriori, mas a diferença é pequena. O resultado
mantém a recomendação de que `AUTO_FROM_MIRRORED_MEDIANS` seja
tratado como heurística experimental e avaliado comparativamente,
não adotado como configuração universalmente superior. A dobra única
em $s_0$ continua sendo a escolha mais parcimoniosa para deployment
operacional padrão.

#### Limitação intrínseca: medianas do treino vs. holdout

Em estratégias com mismatch entre as distribuições de treino e
holdout (e.g. PLDA cross-condition entre populações diferentes), as
medianas $\hat\ell_p, \hat\ell_d$ computadas no treino divergem das do
holdout. Os knots ótimos para minimizar $C_{\text{llr}}$ no holdout
podem cair fora da posição escolhida por AUTO. Mesmo nesses casos,
AUTO não introduz leakage e o $C_{\text{llr}}$ reportado é honesto.

#### Health checks recomendados

Antes de aceitar um calibrador produzido por `AUTO_FROM_MIRRORED_MEDIANS`,
verificar no metadata:

- `hinge_anchor_aux_coef > 0` — auxiliar não degenerou.
- `piecewise_auto_log_lr_hp_p50 > 0` e
  `piecewise_auto_log_lr_hd_p50 < 0` — as medianas auxiliares estão em
  lados opostos do zero.
- `knot_inside_training_range = [true, true]` — todos os knots caem
  dentro do range observado de scores.
- `segment_counts_total[r]` razoável em todos os trechos (e.g. ≥ 30).
- **Cobertura por classe em cada trecho**: `segment_counts_hp[r] > 0` e
  `segment_counts_hd[r] > 0`. Quando uma das classes está totalmente
  ausente num trecho (e.g. cauda direita pura Hp ou cauda esquerda
  pura Hd), a inclinação local fica fracamente identificada pelos
  dados daquela região. Sua estimativa passa a depender mais
  intensamente da continuidade paramétrica da curva, dos demais
  trechos e da regularização L2. Nessa condição, coeficientes
  instáveis e inclinações negativas tornam-se mais prováveis,
  especialmente com regularização fraca. Esse risco **pode ocorrer
  com maior frequência** no modo automático quando os knots
  refletidos caem em regiões de cauda com cobertura assimétrica por
  classe.
- `segment_support_warnings` vazio.
- `is_monotonic = true`.

Se as medianas não estiverem em lados opostos do zero, isso indica que
o LogReg auxiliar não estratifica adequadamente as classes em torno do
ponto neutro neste domínio — `AUTO_FROM_MIRRORED_MEDIANS` pode não ser
adequado. O metadata `piecewise_auto_log_lr_hp_p50` /
`piecewise_auto_log_lr_hd_p50` permite essa verificação ex-post.

#### Caso de fallback

A heurística posiciona dois knots fixos. Em distribuições com regimes
abruptos próximos a $s_0$ (caso observado em cosseno, onde $s_0$ pode
ficar próximo do threshold de decisão associado à região de
equilíbrio entre erros, e as distribuições $H_p$ e $H_d$ são
aproximadamente simétricas em torno dele), a curva ótima pode ter uma
única dobra em $z = 0$ em vez de duas em $-\hat\ell_p$ e $-\hat\ell_d$.
Para essa situação, `MANUAL` com `knot_offsets = [0.0]` em
`KnotScale.AUX_LOG_LR` é o equivalente direto.

---

## 7. Diagnósticos persistidos

Após o fit, o calibrador armazena em `self.metadata` o conjunto
abaixo de campos. Todos são serializados no pickle e ficam disponíveis
para auditoria forense e reprodução. Esta seção é o **schema do
contrato de auditoria** do componente.

### 7.1 Configuração geral

- `calibration_basis_mode`: `"PIECEWISE_LINEAR"` ou `"NONE"`.
- `piecewise_enabled`: bool.
- `knot_placement_protocol_version`: identificador da versão da
  heurística usada (e.g. `"auto_from_mirrored_medians_v1"` desde
  V2.6; `"auto_from_medians_v1"` em pickles V2.0–V2.5;
  `"manual_v1"`).

### 7.2 Convenção de log-LR

- `log_lr_base`: `"natural"` — base do logaritmo.
- `log_lr_computation_method`: `"decision_function"` — método usado
  para extrair log-LR do estimador.
- `positive_class_label`: rótulo numérico da classe positiva no
  treino.
- `effective_training_prior_hp`, `effective_training_prior_hd`:
  prior efetivo de cada classe no treino. Com
  `class_weight="balanced"`, ambos = 0.5.
- `prior_log_odds_correction`: $\log(\pi/(1-\pi))$ que deve ser
  subtraído de `decision_function` para obter log-LR. Com prior
  balanceado = 0.0.

### 7.3 Âncora $s_0$

- `hinge_anchor_mode`: `"AUTO_LOGREG"` ou `"MANUAL"`.
- `hinge_anchor_s0`: $s_0$ aprendido ou informado.
- `hinge_anchor_aux_coef`, `hinge_anchor_aux_intercept`,
  `hinge_anchor_aux_C`: parâmetros do LogReg auxiliar (None em modo
  MANUAL).
- `hinge_anchor_training_method`: `"auto_logreg"` ou `"manual"`.
- `hinge_anchor_inside_training_range`: bool — $s_0$ cai dentro do
  range observado dos scores de treino?

### 7.4 Knots e suporte

- `piecewise_knot_scale`: `"RAW"` ou `"AUX_LOG_LR"`.
- `piecewise_knot_placement`: `"MANUAL"` ou `"AUTO_FROM_MEDIANS"`
  (string serializada — o valor `AUTO_FROM_MEDIANS` é preservado
  para o membro `AUTO_FROM_MIRRORED_MEDIANS` mesmo após o rename do
  enum na V2.6; ver `knot_placement_protocol_version` em §7.1).
- `piecewise_knot_offsets`: offsets em escala de score (campo de
  compatibilidade; mesmo conteúdo de `piecewise_knot_offsets_raw`).
- `piecewise_knot_offsets_user`: offsets na escala do usuário.
- `piecewise_knot_offsets_raw`: offsets em escala de score (sempre).
- `piecewise_knots_absolute`: posições absolutas $k_j$ em escala de
  score.
- `piecewise_auto_log_lr_hp_p50`, `piecewise_auto_log_lr_hd_p50`:
  medianas log-LR computadas no fit (None em modo MANUAL).
- `knot_inside_training_range`: lista bool — cada knot cai dentro do
  range observado?
- `fraction_scores_above_each_knot`: fração da amostra de treino
  acima de cada knot.
- `segment_counts_total`, `segment_counts_hp`, `segment_counts_hd`:
  contagem de pares por trecho.
- `segment_fractions_total`: fração da amostra de treino em cada
  trecho.
- `segment_support_warnings`: lista de avisos quando algum trecho
  tem suporte abaixo dos limiares operacionais.
- `segment_unique_identities_hp`: número de identidades distintas
  (probe_id) nos pares Hp do trecho. None quando o executor não
  passa identidades (input não é `SimilarityScoreDataset`).
- `segment_unique_identities_hd`: número de identidades distintas
  (probe_id ∪ reference_id) nos pares Hd do trecho. Permite
  detectar caudas com muitos pares mas poucas identidades
  dominantes — medida de suporte estatístico mais robusta que a
  contagem de pares.

### 7.5 Coeficientes

- `calibrator_coef_standardized`, `calibrator_intercept_standardized`:
  parâmetros na escala da `StandardScaler` interna.
- `calibrator_coef_raw`, `calibrator_intercept_raw`: parâmetros em
  escala de score bruta (§3.4).

### 7.6 Inclinações e monotonicidade

- `segment_slopes_raw`: lista $[m_1, m_2, \ldots, m_{K+1}]$ na
  escala de score.
- `is_monotonic`: bool.
- `min_allowed_slope`: $m_{\min}$ usado na verificação.
- `monotonicity_check`: descreve a política aplicada (
  `"strict_increasing_post_fit"`).

> **Relação com `ConditionalMonotonicityRule`** (regra plugável,
> `docs/conditional_monotonicity_rule_design.md`): a regra cobre as
> **15 arquiteturas declarativas** de `LR_ARCHITECTURE_VECTORIZATION`
> via `LRArchitectureDerivativeProvider` e, por despacho automático
> (`piecewise_enabled` no metadata, sem `architecture_state`), também o
> **piecewise univariado descrito neste documento**, via
> `PiecewiseLinearDerivativeProvider`: esse provider lê
> `piecewise_knots_absolute` e `segment_slopes_raw` e produz um
> `MonotonicityRegionResult` por segmento ($K+1$ segmentos), com
> `slope_value` constante. A verificação interna do
> `LogisticRegressionCalibrationModel` (`require_monotonicity` +
> `monotonicity_policy`, campos `segment_slopes_raw` e `is_monotonic`
> acima) continua sendo a que roda no treino. A auditoria automática
> pós-refit do executor de `CALIBRATION_MODEL_TRAINING` só é disparada
> quando o calibrador tem `architecture_state`; para o piecewise
> univariado, a regra só roda se for declarada como regra de validação
> de uma instrução.

### 7.7 Ponto de cruzamento efetivo

- `effective_log_lr_zero_crossing`: $s_{LR=0}$ (None se nenhum
  segmento contém zero válido).
- `effective_log_lr_zero_crossing_segment_index`: índice do segmento
  onde a raiz cai (0-indexado).
- `effective_log_lr_zero_crossing_inside_training_range`: bool.

### 7.8 Range observado

- `raw_score_min_training`, `raw_score_max_training`: extremos do
  score no treino.
- `raw_log_lr_min_training`, `raw_log_lr_max_training`: extremos do
  log-LR estimado pelo próprio piecewise nos pontos de treino.

### 7.9 Auxiliar

- `pipeline_feature_names`: nomes descritivos das features expandidas
  (e.g. `["z=score-s0", "max(z, 0)", "max(z - +2.0000, 0)", ...]`).

### 7.10 Seleção de $C$ sob restrição de monotonicidade (V2.5)

Quando `use_gridsearch=True`, o calibrador persiste o resultado da
seleção descrita em §9.3:

- `piecewise_selected_c`: $C$ escolhido (melhor candidato válido por
  CV score).
- `piecewise_selection_policy`: identificador da política aplicada
  (`"best_valid_candidate_by_cv_score"`).
- `piecewise_valid_c_values`: lista de $C$ cuja curva final passou em
  monotonicidade estrita.
- `piecewise_valid_c_cv_scores`: CV scores correspondentes (na
  função objetivo escolhida — ver `piecewise_cv_scoring` abaixo),
  paralelos a `piecewise_valid_c_values`.
- `piecewise_invalid_c_values`: lista de $C$ rejeitados.
- `piecewise_invalid_c_reasons`: motivo textual por $C$ rejeitado
  (inclui slopes computados e índices dos trechos ofensores),
  paralelo a `piecewise_invalid_c_values`.
- `piecewise_cv_scoring`: identificador da função objetivo do CV
  (`"neg_cllr_balanced"` padrão a partir de V2.6, ou
  `"neg_log_loss"` / `"roc_auc"` para escolhas explícitas).
- `piecewise_cv_scoring_log_base`: base do logaritmo na função
  objetivo (`"bits"` em `neg_cllr_balanced`; `"natural"` em
  `neg_log_loss`).
- `piecewise_cv_scoring_class_aggregation`: como Hp e Hd são
  agregados (`"equal_class_weight"` em `neg_cllr_balanced`;
  `"per_sample"` em `neg_log_loss`).
- `cv_splitter_type`: `"IdentityDisjointPairKFold"` (V2.10 com
  modo STRICT_IDENTITY), `"StratifiedGroupKFold"` (V2.8 GROUPED_PROBE
  quando identidades estão disponíveis), `"StratifiedKFold"`
  (V2.7 PAIR — fallback quando o input não traz identidades, e.g.
  dict raw). Valores legados V2.9 como
  `"PredefinedSplit_StrictByIdentity"` não são mais emitidos.
- `cv_splitter_random_seed`: seed do splitter (42 fixo, garante
  reprodutibilidade).
- `cv_fold_count`: número de folds efetivamente usado. Pode ser
  reduzido em relação a `cv_folds` da config quando: (a) a classe
  minoritária tem poucas amostras (`StratifiedKFold`), (b) o
  número de identidades únicas da classe minoritária é menor que
  `cv_folds` (`StratifiedGroupKFold`), ou (c) o número de
  identidades únicas é menor que `cv_folds` no modo STRICT_IDENTITY.
- `cv_identity_disjointness_level` (V2.10): enum textual com nível
  de disjunção identitária garantido pelo splitter
  (`"full"` — STRICT_IDENTITY V2.10; `"probe_only"` — GROUPED_PROBE
  V2.8; `"none"` — PAIR V2.7).
- `cv_identity_disjointness_verified` (V2.10): bool puro indicando
  se a disjunção declarada foi verificada na construção dos folds.
  `True` em STRICT_IDENTITY. `False` em PAIR e em GROUPED_PROBE —
  esse último garante disjunção apenas parcial (probe), e o campo
  `cv_identity_disjointness_level = "probe_only"` registra essa
  natureza separadamente.
- `cv_split_mode_requested` (V2.10): valor do enum
  `LogRegGridSearchCVMode` requisitado pelo config
  (`AUTO`/`PAIR`/`GROUPED_PROBE`/`STRICT_IDENTITY`).
- `cv_split_mode_applied` (V2.10): valor do enum efetivamente usado
  após resolução de fallback (e.g. AUTO → GROUPED_PROBE quando há
  identidades). A partir da V2.13, STRICT_IDENTITY já não cai em
  fallback por input multivariado — o `IdentityDisjointPairKFold` é
  dimension-agnostic em X.
- `cv_split_mode_fallback_reason` (V2.10): descrição textual do
  motivo do fallback, ou `None` se o modo aplicado bateu com o
  requisitado.
- `cv_strict_assignment_strategy` (V2.10, só em STRICT): estratégia
  de atribuição de identidades a grupos —
  `"balanced"` (LPT scheduling — default V2.10) ou `"round_robin"`
  (legado V2.9).
- `cv_strict_random_state` (V2.11, só em STRICT): seed do
  embaralhamento prévio à atribuição (42 fixo).
- `cv_strict_id_to_group_mapping` (V2.11, só em STRICT): dict
  `{identity_str: group_index}` com a atribuição explícita de
  cada identidade ao seu grupo (fold de teste). Permite
  **reprodução exata dos folds** em uma análise externa sem
  precisar rerodar o LPT scheduling — basta carregar o mapping e
  aplicar as máscaras dinâmicas. Chaves stringificadas para
  serialização JSON.
- `cv_strict_lookalike_clusters_used` (V2.11.2, só em STRICT):
  bool indicando se clusters de lookalikes foram declarados via
  `metadata["lookalike_clusters"]` e aplicados na atribuição.
- `cv_strict_n_lookalike_clusters` (V2.11.2): número de clusters
  efetivamente aplicados (clusters com 0 ou 1 identidade efetiva
  no dataset são ignorados como triviais).
- `cv_strict_n_clustered_identities` (V2.11.2): soma das
  identidades efetivamente clusterizadas (ignora membros
  declarados mas ausentes do dataset).
- `cv_strict_n_unique_identities` (V2.10, só em STRICT): tamanho da
  união (probe ∪ reference) particionada em $K$ grupos.
- `cv_strict_n_identities_per_group` (V2.10, só em STRICT): vetor
  de $K$ ints com contagem de identidades por grupo.
- `cv_strict_balance_group_loads` (V2.10 balanced, só em STRICT):
  carga (somatório de aparições) por grupo após LPT scheduling.
- `cv_strict_train_counts_by_fold`,
  `cv_strict_test_counts_by_fold`,
  `cv_strict_excluded_boundary_counts_by_fold`,
  `cv_strict_train_retained_fraction_by_fold`,
  `cv_strict_test_retained_fraction_by_fold`,
  `cv_strict_train_hp_counts_by_fold`,
  `cv_strict_train_hd_counts_by_fold`,
  `cv_strict_test_hp_counts_by_fold`,
  `cv_strict_test_hd_counts_by_fold` (V2.10, só em STRICT):
  contagens dinâmicas por fold para auditoria.
- `cv_strict_fold_health` (V2.10, só em STRICT): dict com
  `thresholds`, `per_fold_status` ∈ {ok, warning, critical},
  `per_fold_violations` (lista de strings descritivas por fold) e
  `overall_status`. Critérios mínimos por fold:
  ≥ 50 Hp e ≥ 50 Hd no train, ≥ 10 Hp e ≥ 10 Hd no test, boundary
  excluído < 50% do total. Folds com qualquer contagem zero ficam
  `critical`; abaixo dos mínimos mas não-zero ficam `warning`.
  Critical loga 🚨, warning loga ⚠️ no logger_callback.

> **Natureza dos thresholds dos health checks.** Os limiares acima
> (≥ 50/10 Hp/Hd, < 50 % boundary) são **guardrails técnicos**
> para detectar folds degenerados de forma automática. Eles não
> constituem critérios estatísticos de **suficiência amostral para
> uso pericial** — a defensibilidade forense depende
> adicionalmente da estabilidade dos parâmetros entre folds
> (§7.11), cobertura por identidade e por pose, comportamento das
> caudas dos scores (§8.3), e validação externa em corpora
> independentes. `overall_status = ok` é condição necessária,
> não suficiente.

**Campos legados V2.9** (não emitidos pela V2.10):
`cv_strict_n_kept_total`, `cv_strict_n_dropped_hd`,
`cv_grouping_level`. Persistiam apenas no protocolo V2.9 do
filtro global e foram substituídos pelas contagens dinâmicas por
fold acima e por `cv_identity_disjointness_level`. Pickles V2.9
antigos ainda podem conter esses campos; novos artefatos não.

Auditoria forense: o laudo pode citar quais $C$ foram avaliados,
quais foram descartados como inviáveis (sem ajuste post-hoc do
grid), qual função objetivo orientou a escolha, e quais foram os
folds usados.

### 7.11 Estabilidade entre folds do CV (V2.5)

Após a seleção de $C$, o calibrador roda `cv.split` adicional com o
$C$ escolhido e coleta, em cada fold, $s_0$, $\alpha_{\text{aux}}$,
knots absolutos, segment slopes e monotonicidade do trecho final —
métricas que indicam a estabilidade da heurística sob variação de
amostra. Especialmente relevante para `AUTO_FROM_MIRRORED_MEDIANS`:
variância alta nos knots entre folds sugere que a heurística é
sensível ao split e seu valor operacional deve ser questionado.

> **Ressalva metodológica importante.** Os diagnósticos `cv_*`
> medem estabilidade entre folds do splitter configurado. Três
> regimes são possíveis na V2.10:
>
> - **STRICT_IDENTITY (V2.10, opt-in)** — `cv_splitter_type =
>   "IdentityDisjointPairKFold"`,
>   `cv_identity_disjointness_level = "full"`,
>   `cv_identity_disjointness_verified = True`. Identidades
>   particionadas em $K$ grupos balanceados via LPT scheduling;
>   pares Hd boundary excluídos apenas do fold corrente (não
>   globalmente); refit final em todo $D_{\text{cal}}$. Disjunção
>   total verificada por fold (§4.4).
> - **GROUPED_PROBE (V2.8 default)** —
>   `cv_splitter_type = "StratifiedGroupKFold"`,
>   `cv_identity_disjointness_level = "probe_only"`,
>   `cv_identity_disjointness_verified = False`. Probe disjunto
>   entre folds; a identidade de **referência** dos pares Hd pode
>   ainda aparecer em múltiplos folds (cada par Hd carrega duas
>   identidades e o splitter aceita apenas uma label de grupo por
>   linha). Cobertura parcial.
> - **PAIR (V2.7 fallback)** —
>   `cv_splitter_type = "StratifiedKFold"`,
>   `cv_identity_disjointness_level = "none"`,
>   `cv_identity_disjointness_verified = False`. Folds formados
>   por par e podem compartilhar identidades inteiramente.
>
> Em todos os casos, essas métricas não substituem reamostragem
> por identidade (§9.5) e a vantagem empírica entre splitters em
> corpora pequenos deve ser verificada por análise de robustez
> dedicada (§4.5).

Campos persistidos:

**Estabilidade dos parâmetros:**

- `cv_hinge_anchor_s0_values`: lista com $s_0$ de cada fold.
- `cv_s0_mean`, `cv_s0_std`: média e desvio padrão entre folds.
- `cv_aux_coef_values`, `cv_aux_coef_mean`, `cv_aux_coef_std`: idem
  para $\alpha_{\text{aux}}$.
- `cv_piecewise_knots_absolute_values`: lista de listas; cada
  elemento são os knots absolutos do fold correspondente.
- `cv_knot_means`, `cv_knot_stds`: estatísticas por knot ao longo
  dos folds (índices alinhados aos knots).
- `cv_segment_slopes_raw_values`: slopes por fold (lista de listas).
- `cv_segment_slopes_mean`, `cv_segment_slopes_std`: estatísticas
  por segmento ao longo dos folds.

**Monotonicidade por fold (V2.7):**

- `cv_fold_monotonicity_recorded`: bool (sempre `True` quando os
  diagnósticos foram computados com sucesso) — sinaliza que a
  política mínima ("monotonicidade do fit final") foi
  complementada por verificação fold-a-fold.
- `cv_fold_is_monotonic_values`: lista bool por fold — `True` se
  todos os segment slopes do fold são estritamente positivos.
- `cv_fold_non_monotonic_indices`: lista de listas — índices
  (0-based) dos segmentos não-monotônicos em cada fold.
- `cv_fold_min_segment_slopes`: lista — slope mínimo observado em
  cada fold (útil pra quantificar a margem de monotonicidade).

A política operacional atual exige monotonicidade **apenas no fit
final** (§9.3), tratando os fields acima como diagnóstico de
estabilidade. Em configuração pericial consolidada, pode-se
endurecer a política exigindo `all(cv_fold_is_monotonic_values)`
como pré-condição adicional para o candidato $C$ ser aceito —
evolução proposta como modo opt-in.

Custo: `cv_folds` fits adicionais sobre o pipeline (≈ duplica o
custo do GridSearch padrão).

---

## 8. Protocolo de validação: domain match

### 8.1 Motivação e taxonomia dos corpora

Um protocolo forense defensável requer que os corpora envolvidos
sejam distinguidos explicitamente:

- **$D_{\text{model}}$** — dados usados para treinar encoder,
  transformações e modelo de similaridade (PLDA, cosseno após
  whitening etc.).
- **$D_{\text{cal}}$** — dados usados para ajustar o calibrador.
- **$D_{\text{dev}}$** — dados opcionais usados para escolher
  protocolo, hiperparâmetros e variantes de calibrador.
- **$D_{\text{val}}$** — holdout externo final, mantido intocado até
  a avaliação consolidada.
- **$D_{\text{case}}$** — material do caso concreto em casework.

A função de calibração estimada pelo componente piecewise depende da
distribuição empírica $(s, y)$ em $D_{\text{cal}}$. Quando
$D_{\text{cal}}$ e $D_{\text{val}}$ vêm de **distribuições diferentes**
— e.g. $D_{\text{model}} = D_{\text{cal}} = \text{CFP}$ e
$D_{\text{val}} = \text{BFW}$ — os parâmetros do LogReg auxiliar
($\alpha_{\text{aux}}, s_0$), as medianas log-LR estimadas para
`AUTO_FROM_MIRRORED_MEDIANS`, e os coeficientes finais $\beta_j$
refletem a geometria de $D_{\text{cal}}$, não a de $D_{\text{val}}$.

O $C_{\text{llr}}$ medido nesse protocolo continua sendo uma medida
honesta — responde à pergunta "quão bem o calibrador treinado em
$D_{\text{cal}}$ transfere para $D_{\text{val}}$?". O que ele **não**
responde diretamente é "qual seria o desempenho esperado de um
calibrador ajustado especificamente para o domínio operacional
$D_{\text{val}}$?". Operacionalmente, em casework no domínio de
$D_{\text{val}}$, há ganho de calibração potencial não capturado.

Uma solução defensável, **quando existem dados representativos
suficientes no domínio operacional**, é particionar o dataset
operacional em subsets disjuntos por identidade. No FaceAudit, isso
é feito via `PAIRED_DATASET_SPLIT` com `grouping_strategy =
PRESERVE_LOOKALIKES` aplicado ao par (frontal, lateral).
Tipicamente: $D_{\text{cal}} = \text{BFW split A}$ (60 %),
$D_{\text{val}} = \text{BFW split B}$ (40 %), com identidades
disjuntas. Quando não há corpus operacional suficiente, o protocolo
cross-domain continua válido com a ressalva acima.

> **Ressalva metodológica importante.** Quando o mesmo
> $D_{\text{val}}$ é usado para **escolher** entre variantes de
> calibrador (e.g. comparar 1-knot em $s_0$ vs AUTO 2-knots vs BI
> GAUSS) e para **reportar** o $C_{\text{llr}}$ final, o conjunto
> está funcionando como $D_{\text{dev}}$, não como
> $D_{\text{val}}$. Os resultados nesse modo são **evidência
> exploratória** legítima, mas otimistas em relação ao desempenho
> esperado em casos novos. Para reportar desempenho final sem
> otimismo, é necessário um $D_{\text{val}}$ externo independente
> (e.g. um terceiro split do BFW reservado *a priori*) ou
> procedimento aninhado (nested CV) equivalente. A tabela §8.2 a
> seguir é, portanto, evidência exploratória para seleção do
> protocolo, não validação final do calibrador escolhido.

### 8.2 Evidência empírica

A tabela abaixo compara duas configurações de protocolo no mesmo
pipeline PLDA SHRINK CROSS, holdout BFW. Os valores no segundo bloco
(BFW split A → BFW split B) foram obtidos com a função objetivo
balanceada `neg_cllr_balanced` (padrão V2.6, §9.3), `c_grid =
[0,001, 0,01, 0,1, 1,0, 10,0]` e splitter agrupado por identidade
do probe (`StratifiedGroupKFold`, padrão V2.8, §4.2).

| Protocolo | Calibrador | $C_{\text{llr}}$ |
|---|---|---|
| Calibrador treinado em CFP, holdout BFW completo | Linear puro | 0,0631 |
| (mesma configuração de pipeline e PLDA) | 1-KNOT S0 | 0,0662 |
|  | AUTO 2 knots | 0,0631 |
|  | BI GAUSS | 0,0509 |
| Calibrador treinado em BFW split A (60%), holdout BFW split B (40%) — identidades disjuntas | Linear puro | **0,0474** |
| (mesma configuração de pipeline e PLDA) | 1-KNOT S0 | **0,0434** |
|  | MANUAL 7-KNOT (`[−15, −12, −9, −6, −3, 0, +6]`) | **0,0413** |
|  | AUTO 2 knots | 0,0465 |
|  | BI GAUSS | 0,0597 |

Três observações empíricas:

**(a) Magnitude do ganho.** Os calibradores LogReg melhoraram entre
24 % e 34 % em $C_{\text{llr}}$ apenas pela mudança de protocolo —
nenhum hiperparâmetro do calibrador foi alterado. O `Cllr_min`
permaneceu em faixa próxima nos dois cenários ($\approx 0{,}038$ no
BFW completo e $\approx 0{,}036$ no BFW split B), compatível com
discriminação potencial semelhante. Importante: os holdouts não são
idênticos (BFW completo vs. BFW split B disjunto por identidade), de
modo que essa comparação **não demonstra invariância exata** da
ordenação subjacente do PLDA. A queda observada em $C_{\text{llr}}$ é
predominantemente atribuível à calibração; uma decomposição precisa
exigiria avaliar ambos os calibradores sobre o **mesmo conjunto de
pares**.

**(b) Inversão do ranking entre calibradores.** No protocolo
cross-domain (CFP → BFW), BI GAUSS foi o melhor calibrador
($C_{\text{llr}} = 0{,}0509$) e o 1-KNOT S0 foi o pior
($0{,}0662$). No protocolo split (BFW-A → BFW-B), 1-KNOT S0 foi o
melhor calibrador piecewise nativo ($0{,}0434$, com GROUPED V2.8 default);
AUTO 2 knots ficou logo abaixo ($0{,}0465$); Linear puro também em
$0{,}0474$; BI GAUSS foi o pior ($0{,}0597$). Não existe um
calibrador universalmente superior — o ranking depende do protocolo
de validação. **Atenção**: a aparente vantagem do 1-KNOT S0 sobre
AUTO 2 knots neste split específico (Δ = 0,003) é da mesma ordem
de grandeza da variabilidade amostral medida em §4.4 — não deve
ser tratada como ordenação robusta.

**(c) Sample efficiency.** A regressão logística piecewise precisa
fixar 2 a 4 parâmetros e tolera amostras menores razoavelmente bem.
O calibrador BI GAUSS implementado no FaceAudit é o algoritmo de
**Morrison (2024) — Bi-Gaussianized calibration**: aplica uma
calibração inicial por LogReg ou KDE (variantes principais do
artigo), ou identidade (`NONE` — adaptação do FaceAudit, aplicável
quando os scores de entrada são tratados como log-LRs nominais sob
o modelo gerador, como pode ocorrer com determinadas implementações
de PLDA; mesmo nesses casos, a adequação empírica da escala deve
ser avaliada no domínio operacional, pois mismatch, regularização e
violações de gaussianidade podem justificar recalibração); estima a
CDF empírica das duas classes com pesos
$0{,}5/(N_d+1)$ e $0{,}5/(N_p+1)$ (no FaceAudit, `gx_method` escolhe
entre a CDF empírica linear por trechos do artigo, `EMPIRICAL_LINEAR`,
e uma versão suavizada por KDE, `KDE_SMOOTHED`, que é o default da
config); e mapeia o output via
$F^{-1}(G(x))$ onde $F$ é a mistura paramétrica
$0{,}5 \cdot \mathcal{N}(+\sigma^2/2, \sigma^2) + 0{,}5 \cdot
\mathcal{N}(-\sigma^2/2, \sigma^2)$ com $\sigma$ derivado do
$C_{\text{llr}}$ ou do EER do output inicial (Morrison 2024, §2).
A CDF empírica é sensível ao $N$ da amostra de treino —
especialmente nas caudas, onde poucas observações geram saltos
discretos grandes em $G(x)$. O artigo destaca a importância de usar
dados de treino representativos da população relevante e amostras
suficientemente grandes para evitar overfitting e falha de
generalização.

No BFW split A (36 identidades / ≈ 684 amostras totais), a CDF
empírica está mal-resolvida nas caudas — gera log-LRs operacionais
ruidosos na região dos genuínos extremos, exatamente onde o
$C_{\text{llr}}$ penaliza fortemente. Os calibradores LogReg são mais
robustos a esse regime de amostra pequena por suporem forma
paramétrica linear ou piecewise-linear em $z$.

### 8.3 Comportamento empírico das caudas

A figura abaixo apresenta os quantis por classe ($H_p$ em verde,
$H_d$ em vermelho) em escala log-LR pós-calibração, para um
calibrador piecewise treinado no protocolo cross-domain (CFP →
BFW). Os genuínos difíceis apresentam penalidade relevante: o
percentil 1 de $H_p$ cai em $-1{,}79$, ainda no lado de log-LR
favorável a $H_d$, enquanto o percentil 5 está em $+1{,}39$. Para
os impostores, P95 = $-3{,}96$ e P99 = $-2{,}02$. A margem entre
quantis extremos é positiva, mas pequena:

$$
-1{,}79 - (-2{,}02) \;=\; +0{,}23.
$$

Portanto, não há inversão entre esses quantis destacados. Há,
contudo, uma cauda genuína com sinal contrário (log-LR $< 0$ no P1),
responsável pela contribuição elevada ao $C_{\text{llr}}$ visível no
painel inferior do gráfico (≈ 3 bits no bin P1).

![Quantis por classe — calibrador em CFP](images/pwl_quantiles_cfp_train.png)

A figura seguinte apresenta os mesmos quantis para um calibrador
treinado em dados do domínio operacional (BFW-A → BFW-B). A cauda
genuína melhora substancialmente: P1 passa de $-1{,}79$ para
$-0{,}68$, e P5 de $+1{,}39$ para $+4{,}08$. A margem extrema
aumenta:

$$
-0{,}68 - (-1{,}21) \;=\; +0{,}53.
$$

A cauda impostora extrema, entretanto, aproxima-se de zero (P99:
$-2{,}02 \to -1{,}21$). O ganho global não representa melhora
uniforme de ambas as classes — decorre principalmente de melhor
calibração dos genuínos difíceis no domínio operacional, refletido
na queda da contribuição ao $C_{\text{llr}}$ no bin P1 de ≈ 3 bits
para < 2 bits.

![Quantis por classe — calibrador em BFW split](images/pwl_quantiles_bfw_split.png)

> **Notas sobre estado das figuras.**
>
> 1. **Figura BFW (segunda imagem)**: regenerada na V2.4 com a
>    semântica corrigida do aviso —
>    `⚠ cauda Hp com log-LR < 0 (n=N)` substituindo o antigo
>    `⚠ inversão`.
> 2. **Figura CFP (primeira imagem)**: gerada em versão anterior
>    (V2.2/V2.3) com o aviso legado `⚠ inversão` ainda exibido.
>    Tentativas posteriores de regenerar a figura com a nova
>    semântica usando os datasets de score disponíveis no projeto
>    (cosseno CFP in-sample, cross-domain CFP→BFW cosseno) não
>    reproduziram cauda Hp com log-LR negativo — os calibradores
>    treinados produzem P1(Hp) > 0 em todos esses cenários. A
>    figura original possivelmente foi gerada com PLDA CFP, cujo
>    parquet de scores não está mais disponível neste projeto.
>    Pelos números no texto da §8.3 (P1(Hp) $= -1{,}79$), o aviso
>    correto na figura CFP também seria
>    `⚠ cauda Hp com log-LR < 0`, não `⚠ inversão` — não há
>    cruzamento entre os quantis extremos destacados (margem
>    $+0{,}23$).
> 3. **Distinção semântica**: o widget de plotagem do FaceAudit
>    foi corrigido na V2.4 — a partir dessa versão, o marcador
>    indica explicitamente a presença de genuínos com log-LR < 0
>    (cauda Hp com sinal contrário) e o número de genuínos
>    afetados (`n=N`). "Inversão entre quantis extremos das
>    classes" é fenômeno distinto e não é mais sinalizado pelo
>    mesmo aviso.

### 8.4 Implementação no FaceAudit

A instrução `PAIRED_DATASET_SPLIT` divide simultaneamente os dois
datasets de uma comparação cross-pose (e.g. frontal e lateral)
mantendo identidades emparelhadas. Parâmetros relevantes:

- `split_ratio`: fração da identidade alocada para o subset A.
  Valores típicos: 0,5 a 0,7 dependendo do tamanho do dataset
  operacional.
- `grouping_strategy = PRESERVE_LOOKALIKES`: mantém clusters
  previamente identificados de identidades visualmente semelhantes no
  mesmo subset, **reduzindo o risco** de leakage por lookalikes
  conhecidos. Não garante que todos os lookalikes reais tenham sido
  encontrados nem elimina toda dependência visual cross-identidade.
- `random_seed`: fixar para reprodutibilidade entre execuções.
- `min_calibration_pairs`, `min_calibration_ids`, `min_samples_per_id`:
  pisos operacionais. Se o split não consegue gerar pares ou
  identidades suficientes para calibração robusta, a instrução
  falha cedo.

O subset designado para calibração entra no pipeline corrente como
input da instrução `SIMILARITY_SCORING` que alimenta o
`CALIBRATION_MODEL_TRAINING`. O subset designado para validação
entra como input do `SIMILARITY_SCORING` na fase de validação. O
pipeline e o modelo de similaridade permanecem treinados em A
(CFP, no exemplo) — apenas o calibrador é treinado em B (BFW split
A).

### 8.5 Reportabilidade

No laudo, o protocolo de validação deve estar documentado
explicitamente. Itens mínimos:

- Dataset operacional e como foi particionado (ratio, seed,
  grouping_strategy, número final de identidades em cada subset).
- Confirmação empírica de disjunção de identidades entre treino do
  calibrador e validação ($|V \cap C| = 0$).
- Tamanho efetivo da amostra de calibração (pares e identidades).
- Política sobre amostras com pose ausente: o split paired pode
  produzir identidades com apenas uma das poses representadas em um
  dos subsets, reduzindo o $N$ efetivo de pares cross-pose.
- Intervalos via bootstrap por identidade (§9.5) quando o suporte
  por classe ou pose é fino.

---

## 9. Uso operacional e reportabilidade

### 9.1 Distinção entre log-LR bruto do modelo e LR reportável

A curva piecewise é linear no último trecho e, portanto, extrapola
sem limite quando $s \to +\infty$ (assumindo slope final positivo) ou
$s \to -\infty$ (slope inicial positivo, log-LR → $-\infty$).

Isso é matematicamente correto, mas não implica que qualquer log-LR
produzido pelo calibrador seja defensável em laudo. Recomenda-se
distinguir, na arquitetura operacional:

- **`raw_model_log_lr`**: saída matemática não censurada do
  calibrador (o que o componente piecewise entrega).
- **`reportable_log_lr`**: valor eventualmente submetido a política
  conservadora de reportabilidade baseada em suporte empírico,
  validação externa, e diretrizes do laboratório/jurisdição.

A política de limite reportável não é definida pela expansão
piecewise-linear. Constitui uma camada operacional separada.

### 9.2 Extrapolação

Os campos `raw_score_min_training` e `raw_score_max_training`
permitem identificar se um par observado em validação ou casework
está dentro ou fora do range em que o calibrador foi estimado.
Recomenda-se documentar e tratar com cautela casos onde o score do
caso cai fora do `[raw_score_min_training, raw_score_max_training]` —
o log-LR nesses pontos é extrapolação linear, não interpolação.

### 9.3 Regularização L2 e seleção de `c_grid`

O calibrador é treinado com `LogisticRegression(C=...)` (inverso da
regularização L2). A escolha de `c_grid` no `GridSearchCV` afeta
diretamente o comportamento dos coeficientes das hinges:

- **`C` alto (≥ 10)** corresponde a regularização fraca. O LogReg fica
  flexível e tende a ajustar coeficientes individuais grandes em
  trechos com pouco contraste discriminativo. Em piecewise com
  cobertura assimétrica por classe (§10.2), isso pode produzir slopes
  negativos em trechos sem Hp ou sem Hd — violação da monotonicidade
  rejeitada por default.
- **`C` moderado a forte (0.001 a 1)** força coeficientes menores. A
  curva fica mais suave, e em trechos sem contraste o $\beta_j$ da
  hinge tende a zero, mantendo o slope efetivo do trecho próximo do
  slope do trecho anterior (positivo). Geralmente resolve violações
  de monotonicidade sem necessidade de reposicionar knots.

Um caso prático observado: com `c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]`
e knots posicionados por `AUTO_FROM_MIRRORED_MEDIANS` num PLDA com Hd ausente
no trecho direito, o GridSearch escolheu `C = 100` e produziu
$m_3 = -0{,}016$ (rejeição). Reduzindo o grid para `[0.01, 0.1]`, o
mesmo dataset produziu `C = 0.1` e $m_3 = +0{,}319$ (monotônica).

#### Monotonicidade como restrição de viabilidade (implementado em V2.5)

A solução acima — restringir manualmente o `c_grid` após observar a
falha — é operacionalmente conveniente mas metodologicamente frágil:
constitui um ajuste experimental orientado pelo resultado. Desde
V2.5, o FaceAudit trata **monotonicidade como restrição de
viabilidade do modelo** durante a seleção de $C$ via o método
`_select_monotonic_C` do `LogisticRegressionCalibrationModel`. O
fluxo é:

1. `c_grid` é definido *a priori* (e.g. `[0.001, 0.01, 0.1, 1.0, 10.0]`);
2. `GridSearchCV(refit=False)` calcula CV score de cada candidato
   usando o scorer balanceado descrito abaixo;
3. **todos** os candidatos são iterados em ordem decrescente de CV
   score; para cada um, é feito fit completo + verificação de
   monotonicidade (Opção A: avalia todos antes de selecionar, pra
   persistir lista completa de aprovações e rejeições);
4. o primeiro candidato cuja curva final é estritamente
   crescente em todos os trechos é selecionado como `best_C`;
5. todos os candidatos avaliados (válidos e inválidos) são
   persistidos no metadata para auditoria (§7.10);
6. quando **nenhum candidato** produz curva monotônica, o calibrador
   é rejeitado com erro listando cada $C$ testado e a violação
   detectada (com `monotonicity_policy = ERROR`; em `WARNING`, o
   erro vira alerta no log e é usado o $C$ de melhor CV score, mesmo
   não-monotônico).

Nessa varredura o limiar de crescimento estrito é fixo em $10^{-8}$;
o `min_allowed_slope` configurado é aplicado na validação do fit
final (§3.3).

Esse procedimento converte a rejeição por não-monotonicidade num
filtro automático, elimina ajuste post-hoc do `c_grid`, e mantém
auditabilidade — o laudo pode citar quais $C$ foram avaliados e
quais foram descartados como inviáveis.

Recomendação geral: usar `c_grid` moderado a forte como default
(`[0.001, 0.01, 0.1, 1.0]`). Valores de $C$ muito altos são
defensáveis apenas em datasets com cobertura simétrica robusta por
trecho.

#### Função objetivo do cross-validation (V2.6)

Como o número de pares impostores pode ser muito maior que o número
de pares genuínos (caso típico em comparação biométrica, com
$N_d / N_p$ entre 10 e 1000), a seleção de $C$ **não deve** usar o
`neg_log_loss` convencional do scikit-learn sem ponderação. Esse
scorer calcula a perda média por amostra; em datasets fortemente
desbalanceados, fica dominado pela classe majoritária. O $C$
selecionado pode ser ótimo para a perda média por par mas
sub-ótimo para o objetivo forense balanceado medido por
$C_{\text{llr}}$.

A partir de V2.6, o FaceAudit utiliza um scorer balanceado
equivalente a $-C_{\text{llr}}$ em base 2 (default da config
`LogisticRegressionCalibrationModelConfig.scoring =
NEG_CLLR_BALANCED`):

$$
\text{Cllr}_{\text{CV}} \;=\; \frac{1}{2}\left[\frac{1}{N_p}\sum_{i \in H_p}\log_2(1 + e^{-\ell_i}) \;+\; \frac{1}{N_d}\sum_{j \in H_d}\log_2(1 + e^{+\ell_j})\right],
$$

com $\ell_k = \text{decision\_function}(x_k)$ — calculado
diretamente do estimador via `decision_function` (não usa
`predict_proba` + clipping, evitando ceilings artificiais). As duas
classes recebem peso agregado igual a $0{,}5$ independentemente da
quantidade de pares. O método correspondente é
`LogisticRegressionCalibrationModel._neg_cllr_balanced_scorer`. O
scorer é registrado como callable diretamente no `GridSearchCV` —
o sklearn aceita callables com assinatura
`(estimator, X, y) -> float`.

A escolha é persistida em `piecewise_cv_scoring` (§7.10) para
auditoria. Se você deliberadamente preferir o `neg_log_loss`
clássico, basta setar `scoring = NEG_LOG_LOSS` no DTO; o metadata
registra a escolha.

### 9.4 Cllr_min e cal_loss como diagnósticos exploratórios

O `Cllr_min` (via PAVA in-sample) e `cal_loss = Cllr − Cllr_min`
calculados sobre o conjunto avaliado são **diagnósticos
exploratórios**, não medidas operacionais significativas do grau de
calibração. Morrison (2021) argumenta que métricas baseadas em
PAV/PAVA não devem ser tratadas como critério isolado para escolha
de sistema pericial em casework. Elas continuam úteis para diagnóstico
comparativo entre variantes do mesmo pipeline, mas não substituem:

- Validação externa em datasets independentes representativos do
  domínio de aplicação.
- Análise das caudas e da estabilidade do modelo.
- Avaliação da aplicabilidade ao caso específico.

### 9.5 Validação por bootstrap

Em uso forense, recomenda-se complementar o $C_{\text{llr}}$ pontual
com intervalos via bootstrap, preferencialmente reamostrando por
**identidade** (não por par) para preservar a estrutura de
correlação.

---

## 10. Limitações

### 10.1 Knots fora do domínio dos dados

Quando os knots absolutos $k_j = s_0 + \delta_j$ caem fora do range
observado dos scores, a feature $\max(z - \delta_j, 0)$ é constante
(zero em todos os pontos) ou linearmente colinear com $z$ em todos os
pontos. No primeiro caso, a feature não carrega informação e o
LogReg estima $\beta_j$ próximo de zero. No segundo caso, a feature
deixa de carregar informação independente e, na presença de
regularização L2, o coeficiente pode ser **reduzido, redistribuído
entre colunas colineares, ou tornar-se numericamente instável**.

O metadata `knot_inside_training_range` permite identificar esse
problema.

### 10.2 Suporte por trecho

Há duas situações distintas que comprometem a estabilidade da
inclinação num trecho, ambas detectadas e reportadas em
`segment_support_warnings`:

**(a) Suporte total baixo.** Quando o knot é colocado numa faixa
esparsa do score (cauda extrema), o número total de amostras no trecho
correspondente é pequeno. O coeficiente $\beta_j$ tem alta variância e
pode assumir valor fortemente negativo ao tentar acomodar ruído local,
violando a condição de monotonicidade do trecho seguinte.

**(b) Cobertura assimétrica por classe.** O suporte total pode ser
grande, mas uma das classes pode estar **totalmente ausente** no
trecho (e.g. cauda direita pura Hp, sem nenhuma amostra Hd; ou cauda
esquerda pura Hd, sem nenhuma amostra Hp). Nessa situação, a
inclinação local fica fracamente identificada pelos dados daquela
região; sua estimativa passa a depender mais intensamente da
continuidade paramétrica da curva, dos demais trechos e da
regularização L2. Sob regularização fraca ($C$ alto), $\beta_j$ pode
assumir valor ligeiramente negativo, levando a slope negativo no
trecho correspondente. Esse padrão pode ocorrer com maior frequência
em distribuições PLDA com classes bem separadas, particularmente
quando a heurística `AUTO_FROM_MIRRORED_MEDIANS` posiciona os knots
em regiões de cauda com cobertura assimétrica.

A mitigação preferencial para (b) é aumentar a regularização L2 via
`c_grid` mais restrito (§9.3). A mitigação alternativa, mais agressiva,
é reposicionar manualmente o knot para uma região onde ambas as
classes ainda têm representação.

Os campos `segment_counts_total`, `segment_counts_hp`,
`segment_counts_hd` e `segment_support_warnings` permitem identificar
ambas as situações após um fit bem-sucedido. Em fits rejeitados por
não-monotonicidade, a mensagem de erro exibe o suporte de cada classe
por trecho explicitamente.

### 10.3 Número de knots vs. graus de liberdade

Cada knot adiciona um parâmetro $\beta_j$ ao modelo. Com $K$ knots, o
calibrador tem $K + 2$ parâmetros (incluindo $\beta_0$ e $\beta_z$).
Em datasets pequenos ou desbalanceados, número grande de knots
favorece sobreajuste. O modo AUTO_FROM_MIRRORED_MEDIANS usa 2 knots (com
colapso defensivo a 1 quando as duas medianas coincidem
numericamente).

### 10.4 Compatibilidade com `SCORE_VECTORIZATION`

A expansão piecewise-linear assume score 1D. Em pipelines de
calibração multivariada (input `SCORE_VECTORIZATION` com $D > 1$),
ativar `mode=PIECEWISE_LINEAR` resulta em erro. Para casos
multivariados, usar `LR_ARCHITECTURE_VECTORIZATION` (arquiteturas com
termos hinge) ou `SCORE_INTERACTION_FEATURE` externo com calibrador
linear, ou aplicar piecewise apenas após reduzir o vetor a 1D.

### 10.5 Score sem tipicidade explícita

Conforme §1.1, o calibrador piecewise produz uma LR baseada em
score. Quando o score representa apenas similaridade, sem incorporar
tipicidade da face questionada em relação à população relevante, a
LR resultante deve ser interpretada como tal. Incluir tipicidade
requer modelagem específica (e.g. `TYPICALITY_PAIR_SCORING` no
FaceAudit) e fusão multivariada, não compensável por refinamento da
calibração 1D.

---

## 11. Relação com `SCORE_INTERACTION_FEATURE`

A instrução `SCORE_INTERACTION_FEATURE` do FaceAudit também
implementa um termo `HINGE_BASE` ($\max(s - s_0, 0)$) como feature
auxiliar para calibradores multivariados. A diferença é onde a hinge
vive:

- **Piecewise-linear (este documento):** hinge interno ao
  calibrador. A regressão logística vê $\boldsymbol\phi(s)$ em vez de
  $s$. Restrito a score 1D.
- **`SCORE_INTERACTION_FEATURE` (HINGE_BASE):** hinge é uma feature
  externa, materializada como nova coluna em
  `SimilarityScoreDataset`. Permite combinar com outras features
  (tipicidade, raridade) em calibrador multivariado via
  `SCORE_VECTORIZATION`. O $s_0$ pode ser MANUAL, AUTO_LOGREG (com a
  mesma fórmula deste documento) ou herdado de execução anterior.

Os dois caminhos são matematicamente equivalentes no caso 1D com
mesmo $s_0$, mesma regularização e mesma escala de standardização. A
**equivalência matemática não implica equivalência operacional**
durante a seleção de hiperparâmetros: no fluxo interno deste
documento, $s_0$ e os knots são aprendidos dentro de cada fold do
`Pipeline` (§4.1), garantindo que o CV avalie variações reais; no
fluxo externo, caso a hinge seja materializada **antes** do
`GridSearchCV` (e.g. via `SCORE_INTERACTION_FEATURE` rodado uma vez
sobre todos os dados), o posicionamento da dobra pode ter usado
informações do conjunto completo e introduz leakage sutil na seleção
de $C$. Essa é uma das razões para preferir a expansão interna em
calibração univariada.

O piecewise interno é a forma ergonômica para o caso 1D; o
`SCORE_INTERACTION_FEATURE` é a forma para combinações multivariadas
onde a hinge é uma entre várias features e o leakage da
materialização precisa ser gerenciado separadamente (e.g. via
particionamento por identidade upstream).

---

## 12. Referências

- **Brümmer, N., & du Preez, J. (2006).** *Application-independent
  evaluation of speaker detection.* Computer Speech & Language,
  20(2–3), 230–275. Introduz $C_{\text{llr}}$ como métrica de
  calibração e a regressão logística como calibrador padrão de
  log-LR.
- **Brümmer, N., & van Leeuwen, D. (2006).** *On calibration of
  language recognition scores.* IEEE Odyssey 2006: The Speaker and
  Language Recognition Workshop. Aprofunda a teoria do
  $C_{\text{llr}}$ e da calibração afim.
- **Brümmer, N., Swart, A., & van Leeuwen, D. (2014).** *A
  comparison of linear and non-linear calibrations for speaker
  recognition.* Odyssey 2014: The Speaker and Language Recognition
  Workshop. Compara calibrações lineares e não lineares e discute
  limitações de transformações afins em diferentes pontos
  operacionais.
- **Morrison, G. S. (2013).** *Tutorial on logistic-regression
  calibration and fusion: Converting a score to a likelihood ratio.*
  Australian Journal of Forensic Sciences, 45(2), 173–197.
- **Morrison, G. S. (2021).** *In the context of forensic casework,
  are there meaningful metrics of the degree of calibration?*
  Forensic Science International: Synergy, 3, 100157. Argumenta que
  métricas baseadas em PAV/PAVA não são medidas operacionais
  significativas do grau de calibração em casework.
- **Morrison, G. S., & Enzinger, E. (2018).** *Score based procedures
  for the calculation of forensic likelihood ratios — Scores should
  take account of both similarity and typicality.* Science & Justice,
  58(1), 47–58. Demonstra empiricamente que scores baseados apenas em
  similaridade ou diferença podem ser inadequados, enquanto scores
  que incorporam similaridade e tipicidade se comportam melhor.
- **Morrison, G. S., Kaye, D. H., Balding, D. J., Taylor, D., Dawid,
  P., Aitken, C. G. G., Gittelson, S., Zadora, G., Robertson, B.,
  Willis, S., Pope, S., Neil, M., Martire, K. A., Hepler, A., Gill,
  R. D., Jamieson, A., de Zoete, J., Ostrum, R. B., & Caliebe, A.
  (2016).** *A comment on the PCAST report: Skip the "match"/"non-match"
  stage.* Forensic Science International. <https://doi.org/10.1016/j.forsciint.2016.10.018>.
  Carta científica argumentando que métodos forenses que produzem
  dados continuamente valorados não devem ser submetidos a uma etapa
  binária de classificação. Fundamenta a recomendação de mapear
  scores diretamente em LR graduada. Referência conceitual da §1.4.
- **Morrison, G. S. (2024).** *Bi-Gaussianized calibration of
  likelihood ratios.* Law, Probability and Risk, 23(1), mgae004.
  Apresenta o algoritmo Bi-Gaussianized adotado no
  `BiGaussianizedCalibrationModel` do FaceAudit (mapeamento via
  CDF empírica das classes em direção a uma mistura paramétrica de
  duas Gaussianas calibradas via Cllr ou EER).
- **Nelder, J. A. (1977).** *A Reformulation of Linear Models.*
  Journal of the Royal Statistical Society. Series A, 140(1),
  48–63. Referência canônica do princípio de marginalidade/hierarquia
  em modelos com termos de interação ou expansão.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The
  Elements of Statistical Learning,* §5.2 — Piecewise polynomials
  and splines. Referência geral sobre expansão de basis functions
  piecewise-linear.
- **ENFSI (2015).** *Guideline for Evaluative Reporting in Forensic
  Science.* European Network of Forensic Science Institutes.
  Moldura geral de relato avaliativo, requisitos de transparência,
  robustez e auditabilidade na atribuição de LR em laudo.

## Apêndice — splitter e modos de CV do GridSearch (migrado do CLAUDE.md)

> Conteúdo migrado do `CLAUDE.md` da raiz na reestruturação de 2026-11. As regras permanentes continuam resumidas lá; aqui fica o detalhe.

- `faceaudit/services/identity_disjoint_pair_kfold.py` — `IdentityDisjointPairKFold(BaseCrossValidator)` — splitter customizado do GridSearch interno do `LogisticRegressionCalibrationModel` quando `gridsearch_cv_mode = STRICT_IDENTITY`. Particiona identidades únicas (probe ∪ reference) em K grupos balanceados via LPT scheduling (Graham 1969) ou round-robin legado. Em cada fold: `test = pares com ambas as ids no grupo g`; `train = pares com ambas as ids fora de g`; **boundary** (uma id dentro, outra fora) **excluído apenas daquele fold** (não globalmente — preserva ≈((K−1)/K)² do Hd no treino). Aceita `lookalike_clusters` opcional: identidades de um cluster ficam no mesmo fold como super-id, carga LPT agregada. Trivial clusters (0 ou 1 id efetiva) e ids órfãs descartados silenciosamente. `get_fold_stats(labels)` retorna `id_to_group_mapping` (reprodução exata) + contagens Hp/Hd dinâmicas por fold + `fold_health` formalizado (`critical`/`warning`/`ok`, thresholds ≥50 Hp/Hd train, ≥10 test, <50 % boundary excluded — guardrails técnicos, **não suficiência forense**). Documentado em `docs/piecewise_linear_calibration.md` §4.4. Testes em `tests/test_identity_disjoint_pair_kfold.py` (25 casos).
- `faceaudit/models/calibration/` — Calibration model configs + implementations (LogReg, KDE, BiGauss). Inclui `LogRegS0Config` — DTO leve usado como `s0_logreg_config_override` opcional em `SCORE_INTERACTION_FEATURE` (calibrador auxiliar univariado de s0). Defaults: `C=1.0, solver=liblinear, class_weight=balanced, max_iter=1000`. `LogisticRegressionCalibrationModelConfig.gridsearch_cv_mode: LogRegGridSearchCVMode` seleciona o splitter do GridSearchCV interno — `AUTO` (default — escolhe `GROUPED_PROBE` quando há identidades disponíveis, cai em `PAIR` caso contrário), `PAIR`, `GROUPED_PROBE`, `STRICT_IDENTITY`. Quando `STRICT_IDENTITY`, o calibrador lê `metadata["lookalike_clusters"]` opcional — clusters declarados são propagados ao `IdentityDisjointPairKFold` como super-ids. Persiste no metadata `cv_split_mode_{requested,applied,fallback_reason}`, `cv_identity_disjointness_{level,verified}` (level enum textual `full`/`probe_only`/`none`; verified bool puro), `cv_strict_*` (mapping id→fold, balance loads, contagens por fold, fold_health). Validação empírica em `docs/piecewise_linear_calibration.md` §4.5 (bootstrap 200 reps com multiplicidades preservadas) e §4.4 (validação de lookalike clusters).
- `faceaudit/enums/logreg_gridsearch_cv_mode.py` — `LogRegGridSearchCVMode` — `AUTO` (default), `PAIR`, `GROUPED_PROBE`, `STRICT_IDENTITY`. Governa o splitter usado pelo `GridSearchCV` interno do `LogisticRegressionCalibrationModel` na seleção de `C` sob restrição de monotonicidade. `AUTO` decide com base em `metadata["train_probe_identities"]` disponível; `STRICT_IDENTITY` ativa `IdentityDisjointPairKFold` com folds dinâmicos + refit final em D_cal cheio. `STRICT_IDENTITY` agora se aplica também a entradas multivariadas (`SCORE_VECTORIZATION` com D > 1) — o splitter é dimension-agnostic, opera só sobre identidades; o fallback automático para `GROUPED_PROBE` em multivariado foi removido. quando `standardize_features=True` e D > 1, a `StandardScaler` é embutida na `sklearn.Pipeline` `[StandardScaler → LogReg]` usada como estimator do `GridSearchCV` — μ/σ por feature aprendidos por fold (sem leakage); refit em D_cal popula `_feature_mean`/`_feature_std` para inferência. `from_str` aceita `"STRICT_BY_IDENTITY"` (legado) como alias para `STRICT_IDENTITY`. Distinto de `CalibratorCVSplitMode` (usado pelo CV diagnóstico opt-in `k_folds > 0`).
