# Calibração piecewise-linear de razão de verossimilhança

Este documento descreve a calibração piecewise-linear (PWL) implementada
no Faceaudit como extensão interna do calibrador de regressão logística.
Cobre a motivação estatística, a formulação matemática, a interpretação
geométrica das mudanças de inclinação, as opções de configuração
disponíveis na ferramenta e as considerações de defensibilidade forense.

---

## 1. Contexto: calibração de scores em comparação facial forense

Em um pipeline forense de comparação facial, cada par (probe, reference)
produz um score de similaridade $s$ a partir de um modelo de comparação
(cosseno entre embeddings, PLDA two-covariance, etc). O score $s$ é uma
estatística informativa, mas não é em si uma razão de verossimilhança
(*likelihood ratio*, LR). A apresentação de evidência em laudo requer
uma quantidade calibrada com a seguinte interpretação probabilística:

$$
\text{LR}(s) \;=\; \frac{p(s \mid H_p)}{p(s \mid H_d)},
$$

onde $H_p$ é a hipótese da promotoria ("a face questionada pertence ao
mesmo indivíduo do padrão") e $H_d$ é a hipótese da defesa ("são
indivíduos distintos"). A função $s \mapsto \log \text{LR}(s)$ é
chamada de **curva de calibração**. O calibrador é o estimador
treinado de $\log \text{LR}(s)$.

Brümmer & du Preez (2006) propõem a regressão logística como estimador
do log-LR a partir de scores e rótulos binários $y \in \{0, 1\}$ (0
para $H_d$, 1 para $H_p$). Sob certas hipóteses, o modelo linear
$\log \text{LR}(s) = \beta_z \cdot s + \beta_0$ é consistente quando
$P(H_p \mid s)$ é logística em $s$. A regressão logística é o
calibrador padrão na literatura forense de reconhecimento biométrico.

---

## 2. Limitações do calibrador linear puro

A linearidade $\log \text{LR}(s) = \beta_z \cdot s + \beta_0$ assume
inclinação constante em toda a faixa do score. Essa hipótese falha em
distribuições reais quando há **regimes distintos** de evidência em
faixas diferentes:

- Em modelos como o PLDA, a distribuição $p(s \mid H_d)$ tem pico
  numericamente distante do pico de $p(s \mid H_p)$. Há regiões onde só
  uma das classes tem massa relevante; nessas regiões, a razão de
  densidades muda de comportamento (Hd cai exponencialmente, Hp ainda
  tem massa).
- Em regiões com pouca sobreposição entre classes, o LogReg linear
  extrapola o slope estimado no centro da distribuição e produz
  log-LRs nas caudas que podem ser sub ou super-estimados em relação à
  razão de densidades empírica.
- Quando o calibrador usa `class_weight="balanced"` (caso do
  Faceaudit), o gradiente do log-loss é uniforme por classe, mas a
  flexibilidade na **forma** da curva continua restrita a uma reta.

A calibração piecewise-linear é uma extensão da regressão logística
que permite mudanças de inclinação em pontos pré-definidos do score
sem perder a propriedade de calibrador probabilístico.

---

## 3. Formulação matemática

### 3.1 Basis expansion piecewise-linear

Seja $s_0 \in \mathbb{R}$ um ponto âncora (definido em §4) e
$\delta_1 < \delta_2 < \cdots < \delta_K$ um conjunto de offsets
relativos ao âncora. Define-se a variável centrada $z = s - s_0$ e a
expansão piecewise-linear:

$$
\boldsymbol{\phi}(s) \;=\; \big[\, z, \; \max(z - \delta_1, 0), \; \max(z - \delta_2, 0), \; \ldots, \; \max(z - \delta_K, 0) \,\big] \in \mathbb{R}^{K+1}.
$$

A regressão logística é então treinada sobre $\boldsymbol{\phi}(s)$
em vez de $s$:

$$
\log \text{LR}(s) \;\approx\; \beta_0 \;+\; \beta_z \cdot z \;+\; \sum_{j=1}^{K} \beta_j \cdot \max(z - \delta_j, 0).
$$

Cada função $\max(z - \delta_j, 0)$ é uma "hinge function" que vale
zero abaixo do knot absoluto $k_j = s_0 + \delta_j$ e cresce
linearmente com inclinação 1 acima dele.

### 3.2 Inclinação por trecho

A derivada de $\log \text{LR}$ em relação ao score $s$ em um ponto
$s \in [k_{j-1}, k_j)$ é:

$$
\frac{d \log \text{LR}}{ds}\bigg|_{s \in [k_{j-1}, k_j)} \;=\; \beta_z \;+\; \sum_{i=1}^{j-1} \beta_i \;\equiv\; m_j.
$$

A inclinação no trecho $j$ é a soma cumulativa de $\beta_z$ com os
coeficientes das hinges que já "ativaram". A curva é um polinômio
piecewise de grau 1 com vértices em $\{k_1, \ldots, k_K\}$.

### 3.3 Restrição de monotonicidade

Uma curva de calibração forense não pode ser não-monotônica em $s$:
score maior deve corresponder a log-LR maior ou igual, sob pena de
inversão da direção da evidência em parte do domínio. A condição
formal é:

$$
m_j \;>\; m_{\min} \quad \forall j \in \{1, \ldots, K+1\},
$$

onde $m_{\min} \geq 0$ é um limiar numérico de monotonicidade
(`min_allowed_slope`, default $10^{-8}$).

Quando a regressão logística sobre $\boldsymbol{\phi}(s)$ produz uma
curva que viola essa restrição (tipicamente em trechos com pouco
suporte estatístico, onde o $\beta_j$ pode ficar fortemente negativo
e sobrepujar a soma cumulativa), o Faceaudit rejeita o fit por
default (§5.5).

### 3.4 Standardização interna

A regressão logística no Faceaudit é treinada via `sklearn.Pipeline`
composto por: `PiecewiseLinearScoreTransformer` → `StandardScaler` →
`LogisticRegression`. A `StandardScaler` aprende
$\mu_j, \sigma_j$ das features expandidas em treino e standardiza:

$$
\tilde\phi_j(s) \;=\; \frac{\phi_j(s) - \mu_j}{\sigma_j}.
$$

Os coeficientes aprendidos $\beta_j^{\text{std}}$ vivem na escala
standardizada. Para auditoria e cálculo de inclinações na escala bruta
do score, recupera-se:

$$
\beta_j^{\text{raw}} \;=\; \beta_j^{\text{std}} / \sigma_j, \qquad
\beta_0^{\text{raw}} \;=\; \beta_0^{\text{std}} \;-\; \sum_j \beta_j^{\text{std}} \cdot \mu_j / \sigma_j.
$$

Esta conversão é feita no calibrador após o `fit` e os coeficientes em
escala bruta são persistidos no metadata para uso em diagnóstico e
plot da curva analítica.

### 3.5 Ponto de cruzamento efetivo de log-LR = 0

O âncora $s_0$ é o ponto onde $\log \text{LR}$ do **LogReg auxiliar
linear** vale zero (cf. §4.1), mas em geral **não** coincide com o
zero do calibrador piecewise final. O zero efetivo é determinado
pelo trecho que contém o ponto. Para o trecho mais à direita (slope
$m_{K+1} = \beta_z + \sum_i \beta_i$):

$$
s_{LR = 0} \;=\; s_0 \;-\; \frac{\beta_0^{\text{raw}}}{m_{K+1}}.
$$

Este valor é persistido como `effective_log_lr_zero_crossing` para
documentação no laudo.

---

## 4. O ponto âncora $s_0$

A basis expansion é parametrizada pelo âncora $s_0$. Os knots
absolutos são $k_j = s_0 + \delta_j$. A escolha de $s_0$ define o que
significa "centrar a expansão". Há dois modos:

### 4.1 Estimação automática (`HingeAnchorMode.AUTO_LOGREG`)

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
usado pelas escalas `LOG_LR` (§5.3) e `AUTO_FROM_MEDIANS` (§5.4).

O LogReg auxiliar é treinado **dentro** do
`PiecewiseLinearScoreTransformer.fit`, o que o torna automaticamente
fold-aware em pipelines com `GridSearchCV`: cada fold re-estima seu
próprio $s_0$ usando apenas seus dados de treino, sem leakage do
holdout.

### 4.2 Valor fixo (`HingeAnchorMode.MANUAL`)

O usuário fornece $s_0$ diretamente via
`hinge_anchor_manual_value`. Útil para reproduzir calibradores
externos já auditados ou aplicar regras de domínio explícitas. Não
gera $\alpha_{\text{aux}}$ — escalas e modos que dependem dele ficam
restritos.

---

## 5. Opções de configuração no Faceaudit

A configuração do piecewise vive no DTO
`PiecewiseLinearCalibrationConfig`, embutido no
`LogisticRegressionCalibrationModelConfig` via
`piecewise_linear_config`. As subseções abaixo descrevem cada campo.

### 5.1 `CalibrationBasisMode`

Ativa ou desativa a basis expansion.

| Valor | Efeito |
|---|---|
| `NONE` (default) | Calibrador opera como LogReg linear puro. Backward-compat. |
| `PIECEWISE_LINEAR` | Ativa a basis expansion (§3). |

### 5.2 `HingeAnchorMode`

Como $s_0$ é determinado (§4).

| Valor | Efeito |
|---|---|
| `AUTO_LOGREG` (default) | Aux LogReg fold-aware estima $s_0$ e $\alpha_{\text{aux}}$. |
| `MANUAL` | Usa `hinge_anchor_manual_value` direto. |

### 5.3 `KnotScale`

Unidade em que `knot_offsets` é expresso.

| Valor | Significado |
|---|---|
| `RAW` | Offsets diretamente em unidades de score. Backward-compat com configs antigos. |
| `LOG_LR` (default em construções novas) | Offsets em unidades de log-LR auxiliar. Conversão $\delta_j^{\text{raw}} = \delta_j^{\text{log-LR}} / \alpha_{\text{aux}}$ feita no fit. Exige `HingeAnchorMode.AUTO_LOGREG`. |

A escala `LOG_LR` é independente do modelo de similaridade subjacente:
"+2 log-LR" significa "razão de verossimilhança de 100:1 a favor de
$H_p$" tanto em cosseno quanto em PLDA. A escala `RAW` requer
conhecer o domínio numérico do score (cosseno em ±1, PLDA tipicamente
em ±100).

### 5.4 `KnotPlacement`

Quem decide a posição dos knots.

| Valor | Significado |
|---|---|
| `MANUAL` | Usuário escreve `knot_offsets` na configuração. |
| `AUTO_FROM_MEDIANS` (default em construções novas) | Transformer popula `knot_offsets` no fit usando a heurística de §6. |

Em `AUTO_FROM_MEDIANS`, os campos `knot_offsets` e `knot_scale` do
DTO são ignorados; o transformer determina ambos automaticamente. A
UI esconde esses campos quando AUTO está selecionado.

### 5.5 `require_monotonicity` e `monotonicity_policy`

Controlam a verificação de monotonicidade (§3.3) após o fit final.

| `monotonicity_policy` | Comportamento se a curva viola $m_j > m_{\min}$ |
|---|---|
| `ERROR` (default) | Levanta `ValueError` com diagnóstico detalhado por trecho. Calibrador não é entregue. |
| `WARNING` | Emite log de aviso e mantém o fit. Reservado para diagnóstico. |

`require_monotonicity = False` desativa a checagem por inteiro (não
recomendado em uso forense).

### 5.6 `min_allowed_slope`

Limiar $m_{\min}$ abaixo do qual um slope é considerado violador
(default $10^{-8}$). Mantém imunidade a ruído numérico (slopes
positivos mas muito próximos de zero podem aparecer em fits
pesadamente regularizados).

### 5.7 Configuração do LogReg auxiliar (`LogRegS0Config`)

Os hiperparâmetros do LogReg auxiliar (usado para estimar
$\alpha_{\text{aux}}, s_0$) são embutidos em
`s0_logreg_config`. Defaults: $C = 1.0$, `solver=liblinear`,
`class_weight=balanced`, `max_iter=1000`. O `class_weight=balanced`
garante que o âncora não seja deslocado pelo desbalanceamento entre
classes.

---

## 6. A heurística do espelho das medianas log-LR

A heurística empírica para posicionar 2 knots, derivada da análise de
estratégias PLDA e cosseno do projeto, é:

$$
\boldsymbol\delta \;=\; \mathrm{sorted}\big(\big[\, -\, \text{med}(\log \text{LR} \mid H_p), \;\; -\, \text{med}(\log \text{LR} \mid H_d) \,\big]\big).
$$

**Interpretação geométrica.** Seja $\ell_p = \text{med}(\log \text{LR}
\mid H_p)$ e $\ell_d = \text{med}(\log \text{LR} \mid H_d)$. Cada knot
em $-\ell_p$ ou $-\ell_d$ posiciona-se no espelho em torno de zero
da mediana log-LR da classe correspondente. O segmento entre os dois
knots é a "zona de mistura", onde as duas classes têm massa
relevante; fora dela, há predominância de uma classe.

**Cálculo a partir do treino.** As medianas log-LR são calculadas
analiticamente a partir das medianas raw e do $\alpha_{\text{aux}}$:

$$
\ell_p \;=\; \alpha_{\text{aux}} \cdot (\text{med}(s \mid y = 1) - s_0), \qquad
\ell_d \;=\; \alpha_{\text{aux}} \cdot (\text{med}(s \mid y = 0) - s_0).
$$

Essa fórmula não exige treinar um segundo calibrador completo nem
ter um dev set separado; usa o próprio LogReg auxiliar já treinado
para $s_0$.

**Fold-awareness.** Em `AUTO_FROM_MEDIANS`, esse cálculo acontece
dentro do `PiecewiseLinearScoreTransformer.fit`, que é chamado uma
vez por fold do `GridSearchCV`. Cada fold tem seu próprio
$\alpha_{\text{aux}}, s_0, \ell_p, \ell_d$ e, portanto, seus próprios
knots, computados estritamente sobre seus dados de treino. Não há
leakage do holdout do fold para a definição dos knots.

**Quando a heurística falha.** A heurística posiciona dois knots
fixos. Em distribuições com regimes "abruptos" próximos a $s_0$
(caso típico do cosseno, onde $s_0 \approx \text{EER}$ e as
distribuições $H_p$ e $H_d$ são aproximadamente simétricas em torno
dele), a curva ótima pode ter uma única dobra em $z = 0$ em vez de
duas em $-\ell_p$ e $-\ell_d$. A configuração `MANUAL` com
`knot_offsets = [0.0]` em `KnotScale.LOG_LR` cobre esse caso.

---

## 7. Defensibilidade forense

### 7.1 Protocolo recomendado: `AUTO_FROM_MEDIANS`

O modo `AUTO_FROM_MEDIANS` é o protocolo forense recomendado pelas
seguintes razões:

1. **Ausência de leakage.** Os knots são determinados exclusivamente
   a partir do conjunto de treino do calibrador. O conjunto de
   validação (holdout) permanece independente para avaliação de
   $C_{\text{llr}}$ (Brümmer & van Leeuwen 2006).
2. **Determinismo.** Dado o mesmo conjunto de treino e a mesma
   semente de inicialização, o procedimento gera os mesmos knots.
3. **Documentabilidade.** O critério cabe em uma única sentença no
   laudo: "os knots foram posicionados em
   $\mathrm{sorted}([-\hat\ell_p, -\hat\ell_d])$ a partir das
   medianas log-LR auxiliares estimadas fold-aware sobre o dataset de
   calibração". Todos os valores numéricos relevantes são persistidos
   no metadata do calibrador (§9).
4. **Sem dependência operacional.** Não exige rodar uma "estratégia
   gêmea" linear sobre o holdout, procedimento que constituiria
   leakage.

### 7.2 Uso de `MANUAL`

O modo `MANUAL` é defensável forensicamente quando:

- O usuário tem um **dev set independente do holdout** e
  computa $\ell_p, \ell_d$ sobre ele.
- O usuário reproduz um calibrador previamente auditado.
- O usuário aplica regra explícita de domínio (e.g. "knot no
  $z = 0$" como hipótese a ser testada).

Não é defensável quando os knots são escolhidos olhando estatísticas
do mesmo conjunto onde $C_{\text{llr}}$ é reportado — esse
procedimento constitui tuning de hiperparâmetros no test set e o
ganho observado em $C_{\text{llr}}$ não generaliza para um dataset
independente.

### 7.3 Limitação intrínseca de `AUTO_FROM_MEDIANS`

Em estratégias com forte mismatch entre as distribuições de treino e
holdout (caso típico: PLDA cross-condition entre populações
diferentes), as medianas $\ell_p, \ell_d$ do treino divergem das do
holdout. Os knots ótimos para minimizar $C_{\text{llr}}$ no holdout
podem cair fora da posição escolhida por AUTO. Mesmo nesses casos,
AUTO não introduz leakage e, portanto, o $C_{\text{llr}}$ reportado é
honesto. Uma forma defensável de capturar o ganho é separar treino,
dev e holdout, usando MANUAL com medianas do dev.

---

## 8. Limitações e armadilhas

### 8.1 Knots fora do domínio dos dados

Quando os knots absolutos $k_j = s_0 + \delta_j$ caem fora do range
observado dos scores, a feature $\max(z - \delta_j, 0)$ é constante
(igual a zero ou linearmente colinear com $z$) sobre todos os pontos
de treino. O LogReg estima coeficientes próximos de zero para essas
features e o modelo colapsa para o LogReg linear puro. O risco é
particularmente relevante quando offsets são escritos em
`KnotScale.RAW` para modelos com scores em pequena faixa numérica
(cosseno em ±1), onde valores como ±10 caem fora do domínio. A
escala `KnotScale.LOG_LR` evita esse problema por construção.

### 8.2 Knots em regiões de baixo suporte

Quando um knot é colocado em uma faixa esparsa do score (e.g. cauda
direita extrema), o número de amostras no trecho correspondente é
pequeno. O coeficiente $\beta_j$ associado tem alta variância e pode
assumir valor fortemente negativo ao tentar acomodar ruído local,
violando a condição de monotonicidade do trecho seguinte. Quando
isso ocorre com `MonotonicityPolicy.ERROR`, o fit é rejeitado com
diagnóstico detalhado por trecho indicando a suspeita.

### 8.3 Número de knots vs. graus de liberdade

Cada knot adiciona um parâmetro $\beta_j$ ao modelo. Com $K$ knots, o
calibrador tem $K + 2$ parâmetros (incluindo $\beta_0$ e $\beta_z$).
Em datasets pequenos ou desbalanceados, número grande de knots
favorece sobreajuste. O modo AUTO usa 2 knots sempre (com colapso
defensivo a 1 quando as duas medianas coincidem numericamente).

### 8.4 Compatibilidade com `SCORE_VECTORIZATION`

A expansão piecewise-linear assume score 1D. Em pipelines de
calibração multivariada (input `SCORE_VECTORIZATION` com $D > 1$),
ativar `mode=PIECEWISE_LINEAR` resulta em erro. Para casos
multivariados, usar `SCORE_INTERACTION_FEATURE` externo com
calibrador linear, ou aplicar piecewise apenas após reduzir o vetor a
1D.

---

## 9. Diagnósticos persistidos

Após o fit, o calibrador armazena em `self.metadata` o conjunto
abaixo de campos para auditoria forense e reprodução:

**Configuração:**
- `calibration_basis_mode`: "PIECEWISE_LINEAR" ou "NONE".
- `piecewise_enabled`: bool.
- `piecewise_knot_placement`: "MANUAL" ou "AUTO_FROM_MEDIANS".
- `piecewise_knot_scale`: "RAW" ou "LOG_LR".

**Âncora:**
- `hinge_anchor_mode`: "AUTO_LOGREG" ou "MANUAL".
- `hinge_anchor_s0`: $s_0$ aprendido ou informado.
- `hinge_anchor_aux_coef`, `hinge_anchor_aux_intercept`,
  `hinge_anchor_aux_C`: parâmetros do LogReg auxiliar (None em modo
  MANUAL).
- `hinge_anchor_training_method`: "auto_logreg" ou "manual".
- `hinge_anchor_inside_training_range`: bool indicando se $s_0$ cai
  dentro do range observado dos scores de treino.

**Knots:**
- `piecewise_knot_offsets_user`: offsets na escala do usuário
  (`LOG_LR` ou `RAW`).
- `piecewise_knot_offsets_raw`: offsets em escala de score (sempre).
- `piecewise_knots_absolute`: posições absolutas $k_j$ em escala de
  score.

**Medianas (modo AUTO):**
- `piecewise_auto_log_lr_hp_p50`, `piecewise_auto_log_lr_hd_p50`:
  $\ell_p, \ell_d$ computados no fit (None em modo MANUAL).

**Coeficientes:**
- `calibrator_coef_standardized`, `calibrator_intercept_standardized`:
  parâmetros na escala da `StandardScaler` interna.
- `calibrator_coef_raw`, `calibrator_intercept_raw`: parâmetros em
  escala de score bruta (§3.4).

**Slopes e monotonicidade:**
- `segment_slopes_raw`: lista $[m_1, m_2, \ldots, m_{K+1}]$ na
  escala de score.
- `is_monotonic`: bool.
- `min_allowed_slope`: $m_{\min}$ usado.

**Ponto de evidência neutra:**
- `effective_log_lr_zero_crossing`: $s_{LR=0}$ no trecho mais à
  direita (§3.5).

**Nomes das features:**
- `pipeline_feature_names`: lista descritiva como
  `["z=score-s0", "max(z, 0)", "max(z - +2.0000, 0)", ...]` para
  inspeção e debug.

---

## 10. Referências

- **Brümmer, N., & du Preez, J. (2006).** *Application-independent
  evaluation of speaker detection.* Computer Speech & Language,
  20(2–3), 230–275. Introduz $C_{\text{llr}}$ como métrica de
  calibração e a regressão logística como calibrador padrão de
  log-LR.
- **Brümmer, N., & van Leeuwen, D. (2006).** *On calibration of
  language recognition scores.* IEEE Odyssey 2006: The Speaker and
  Language Recognition Workshop. Aprofunda a teoria do
  $C_{\text{llr}}$ e da calibração afim.
- **Morrison, G. S. (2021).** *In the context of forensic casework,
  are there meaningful metrics of the degree of calibration?*
  Science & Justice, 61(1), 102–113. Discute a relação entre PAVA
  in-sample e medidas operacionais de calibração; relevante para a
  interpretação do `Cllr_min` e do `cal_loss` reportados pelo
  Faceaudit.
- **Tukey, J. W. (1977).** *Exploratory Data Analysis.* Princípio da
  hierarquia em modelos com termos de interação: incluir o termo
  principal sempre que se inclui o termo de interação. Aplica-se ao
  desenho da expansão piecewise (sempre incluir $z$ como primeira
  feature).
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The
  Elements of Statistical Learning,* §5.2 — Piecewise polynomials
  and splines. Referência geral sobre expansão de basis functions
  piecewise-linear.

---

## 11. Apêndice: relação com expansão SCORE_INTERACTION_FEATURE

A instrução `SCORE_INTERACTION_FEATURE` do Faceaudit também
implementa um termo `HINGE_BASE` (`max(s - s_0, 0)`) como feature
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

Os dois caminhos são equivalentes matematicamente no caso 1D com
mesmo $s_0$ e mesma regularização. O piecewise interno é a forma
ergonômica para o caso 1D; o `SCORE_INTERACTION_FEATURE` é a forma
para combinações multivariadas onde a hinge é uma entre várias
features.
