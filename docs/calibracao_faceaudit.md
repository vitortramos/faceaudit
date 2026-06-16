# Calibração de scores em razões de verossimilhança no FaceAudit

> **Status:** documento técnico de referência interna.
> **Escopo:** calibração univariada de scores contínuos em log-razão de
> verossimilhança (log-LR), avaliação, seleção e auditoria.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** repositório FaceAudit no commit `c39919f`
> (branch `claude/read-claude-md-SgzIV`). Defaults, fórmulas e enums
> conferidos em `faceaudit/models/calibration/`, `faceaudit/enums/`,
> `faceaudit/services/cllr_math.py`, `faceaudit/services/validation_math_service.py`,
> `faceaudit/protocols/calibration_model_protocol.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a calibração se encaixa no FaceAudit](#1-onde-a-calibração-se-encaixa-no-faceaudit)
- [2. Conceitos-base e notação](#2-conceitos-base-e-notação)
- [3. Como avaliar discriminação, calibração e transferência](#3-como-avaliar-discriminação-calibração-e-transferência)
- [4. Família discriminativa — regressão logística](#4-família-discriminativa--regressão-logística)
- [5. Família generativa — KDE](#5-família-generativa--kde)
- [6. Família bi-gaussianizada](#6-família-bi-gaussianizada)
- [7. Seleção, validação e prevenção de leakage](#7-seleção-validação-e-prevenção-de-leakage)
- [8. Calibração no pipeline do FaceAudit](#8-calibração-no-pipeline-do-faceaudit)
- [9. Como escolher o método apropriado](#9-como-escolher-o-método-apropriado)
- [10. Caudas, extrapolação e reportabilidade](#10-caudas-extrapolação-e-reportabilidade)
- [11. Auditoria e rastreabilidade](#11-auditoria-e-rastreabilidade)
- [12. Boas práticas resumidas](#12-boas-práticas-resumidas)
- [13. Perguntas frequentes](#13-perguntas-frequentes)
- [14. Glossário](#14-glossário)
- [15. Referências](#15-referências)
- [Apêndice A — Fórmulas de referência](#apêndice-a--fórmulas-de-referência)
- [Apêndice B — LogisticRegressionCalibrationModelConfig](#apêndice-b--logisticregressioncalibrationmodelconfig)
- [Apêndice C — PiecewiseLinearCalibrationConfig](#apêndice-c--piecewiselinearcalibrationconfig)
- [Apêndice D — KDECalibrationModelConfig](#apêndice-d--kdecalibrationmodelconfig)
- [Apêndice E — BiGaussianizedCalibrationModelConfig](#apêndice-e--bigaussianizedcalibrationmodelconfig)
- [Apêndice F — Enums](#apêndice-f--enums)
- [Apêndice G — Checklist de revisão documental](#apêndice-g--checklist-de-revisão-documental)
- [Apêndice H — Checklist de validação operacional](#apêndice-h--checklist-de-validação-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica geral sobre a calibração de scores no
FaceAudit. Reúne, num único arquivo, os fundamentos matemáticos, as
implementações disponíveis, as configurações operacionais e os controles
metodológicos necessários para uso defensável em laudo.

### 0.2 Público-alvo

Peritos, desenvolvedores e revisores técnicos. O texto é didático na exposição
inicial de cada tópico, formal na formulação matemática e exaustivo nas
tabelas de configuração.

### 0.3 Escopo

**Incluído:**

- scores contínuos univariados;
- conversão em log-razão de verossimilhança (log-LR);
- métricas de discriminação, calibração e transferência;
- seleção e validação de calibradores;
- integração no pipeline do FaceAudit;
- auditoria e rastreabilidade.

**Fora do escopo:**

- engenharia detalhada de embeddings;
- treinamento completo do backend de similaridade;
- fusão multivariada score × raridade (`SCORE_INTERACTION_FEATURE`,
  `S0Mode`, `ScoreInteractionMode`) — sobre isso, ver os módulos
  correspondentes em `faceaudit/executors/instructions/` e
  `faceaudit/enums/`;
- formulação específica das proposições de cada exame pericial.

### 0.4 Documentos complementares

Este documento é uma **referência guarda-chuva**. Para profundidade
metodológica, manutenção interna e evidência empírica, consulte:

| Documento | Escopo | Relação com este |
|---|---|---|
| `docs/calibracao_faceaudit.md` (este arquivo) | fundamentos gerais, visão integrada das famílias de calibradores, configurações, métricas e diagnósticos, protocolo de seleção e validação, integração com o pipeline, rastreabilidade | guarda-chuva |
| `docs/piecewise_linear_calibration.md` | comportamento interno da expansão piecewise-linear, parâmetros, regras de implementação, validações específicas | especificação interna |
| `calibracao_piecewise_linear_faceaudit_relatorio_integrado.md` | fundamentação forense, função hinge, âncora e escalas, escolha de knots, risco de cherry-picking, parcimônia, benchmark sintético, estudo de caso, protocolo operacional, perguntas críticas previsíveis | aprofundamento técnico-forense da piecewise-linear |
| `experiments/calibration/RELATORIO_comparacao_calibradores.md` | comparação empírica, cenários sintéticos com oráculo conhecido, limitações observadas | evidência experimental |
| `docs/vetorizacao_scores_faceaudit.md` (§1.6) + `CLAUDE.md` (seção "Arquitetura LR declarativa") | árvore de decisão `SCORE_VECTORIZATION` × `LR_ARCHITECTURE_VECTORIZATION` (V2.14) e descrição do modo `parameter_mode=FIT_PER_FOLD` consumido por este calibrador | substituto declarativo para arquiteturas LR canônicas + caminho per-fold do LogReg |

Quando a piecewise-linear for tratada aqui, será apenas em nível resumido,
remetendo ao relatório integrado para o aprofundamento.

---

## 1. Onde a calibração se encaixa no FaceAudit

### 1.1 Fluxo conceitual

A calibração é a etapa que transforma o score contínuo produzido por um
modelo de similaridade em uma estimativa contínua de log-LR.

```text
imagem
  ↓
embedding facial
  ↓
backend de similaridade (cosseno, PLDA, ...)
  ↓
score bruto contínuo
  ↓
calibrador
  ↓
log-LR contínuo
  ↓
avaliação, validação e relato
```

### 1.2 Score bruto não é evidência calibrada

O score bruto normalmente preserva a **ordenação relativa** entre os pares —
pares mais similares tendem a receber scores maiores. Sua magnitude, porém,
não possui necessariamente interpretação direta como força de evidência em
escala forense.

Um cosseno de 0,72 não significa, por si só, “evidência forte para mesma
pessoa”. Para que essa afirmação seja sustentada, é preciso transformar o
score em uma medida graduada de suporte às proposições do exame. Essa
transformação é a calibração.

### 1.3 Calibração não é classificação

Calibrar não é classificar. O FaceAudit produz, na saída, um valor contínuo
(o log-LR), não uma decisão binária `match` / `non-match`.

A dicotomização precoce introduz o chamado **cliff-edge effect**: dois scores
quase idênticos, mas em lados opostos de um limiar, recebem conclusões
qualitativamente distintas; e dois scores muito diferentes, ambos acima do
limiar, recebem a mesma conclusão. A saída contínua evita essa descontinuidade
artificial.

Em consequência, **os knots da expansão piecewise-linear não são limiares
classificatórios**. Eles apenas alteram localmente a inclinação da curva de
calibração — a curva continua contínua, sem saltos, e o número emitido segue
contínuo.

### 1.4 O calibrador não cria informação ausente do score

Esse ponto precisa ser explicitado para evitar interpretação excessiva.

Se o score de entrada representa apenas similaridade (a chance de duas imagens
serem visualmente parecidas), o calibrador produz uma **LR baseada em score
de similaridade**. Uma transformação calibradora pode corrigir a forma do
mapeamento `score → log-LR`, mas **não acrescenta, por si só**, informação de
tipicidade da face questionada em relação à população relevante.

Quando tipicidade ou raridade forem relevantes (por exemplo, uma face muito
incomum aumenta o valor probatório de uma similaridade média), devem ser
incorporadas por modelagem multivariada específica. Refinar a calibração
univariada não substitui essa etapa.

A fronteira com modelos multivariados — em particular o módulo
`SCORE_INTERACTION_FEATURE` — está documentada nos próprios módulos em
`faceaudit/executors/instructions/` e em `faceaudit/enums/`.

---

## 2. Conceitos-base e notação

### 2.1 Proposições

Toda comparação forense é avaliada sob duas proposições mutuamente
excludentes:

$$
H_p:\text{as imagens comparadas provêm do mesmo indivíduo,}
$$

$$
H_d:\text{as imagens comparadas provêm de indivíduos distintos.}
$$

### 2.2 Razão de verossimilhança baseada em score

Quando o sistema produz um score contínuo $s$, a razão de verossimilhança
baseada nesse score é

$$
\mathrm{LR}(s)=\frac{p(s\mid H_p)}{p(s\mid H_d)}.
$$

### 2.3 Escala logarítmica

O FaceAudit trabalha na **escala logarítmica natural** (em **nats**):

$$
\log\mathrm{LR}(s)=\log p(s\mid H_p)-\log p(s\mid H_d).
$$

A escala logarítmica é simétrica em torno de zero (evidência neutra) e
numericamente estável.

> **Nota sobre bases.** O log-LR é em nats (log natural). A métrica de
> qualidade Cllr (próxima seção) é reportada em **bits** (log₂). A conversão
> é direta: $\log_{10}\mathrm{LR}=\log\mathrm{LR}/\ln 10\approx\log\mathrm{LR}/2{,}303$.

### 2.4 Interpretação numérica

Tabela de leitura:

| $\log\mathrm{LR}$ | LR aproximada | Leitura |
|---|---|---|
| $0$ | $1$ | suporte equivalente; evidência neutra |
| $+2{,}30$ | $10$ | suporte aproximado de $10{:}1$ para $H_p$ |
| $-2{,}30$ | $0{,}1$ | suporte aproximado de $10{:}1$ para $H_d$ |
| $+4{,}61$ | $100$ | suporte aproximado de $100{:}1$ para $H_p$ |
| $-4{,}61$ | $0{,}01$ | suporte aproximado de $100{:}1$ para $H_d$ |

> **Exemplo.** Um calibrador recebe um cosseno $s=0{,}62$ e devolve
> $\log\mathrm{LR}=+3{,}0$. Isso corresponde a $\mathrm{LR}=e^{3}\approx 20$:
> a similaridade observada é cerca de 20 vezes mais esperada se as duas
> imagens forem da mesma pessoa do que se forem de pessoas diferentes. **É
> esse número, e não o `0,62` bruto, que entra no laudo.**

### 2.5 Rótulos internos

No `SimilarityScoreDataset`, a convenção de rótulos é fixa e aparece em todo
o pipeline:

| Rótulo | Significado |
|---|---|
| `1` | par genuíno (mesma identidade); hipótese $H_p$ |
| `0` | par impostor (identidades diferentes); hipótese $H_d$ |
| `-1` / `None` | ground truth desconhecido (uso em casework) |

A inferência de `same_person` em `SimilarityScoreSample.__post_init__` exige
que ambas as identidades estejam presentes; ausência é tratada como “verdade
desconhecida”, não como `False`.

### 2.6 Monotonicidade

Quando o backend de similaridade está orientado de modo que score maior
indica maior similaridade, a curva calibrada $s\mapsto\log\mathrm{LR}(s)$
deve ser **monotônica não decrescente**. Caso contrário, dois pares com
scores quase iguais poderiam receber log-LRs em direções opostas, e a saída
seria incoerente.

A monotonicidade é uma propriedade exigida; ela não decorre automaticamente
da forma funcional escolhida (ver §4.2.9 e §5.5).

---

## 3. Como avaliar discriminação, calibração e transferência

### 3.1 Três perguntas distintas

A avaliação de um sistema de comparação não se reduz a uma única métrica.
Existem três perguntas diferentes, e métricas diferentes endereçam cada uma
delas:

| Dimensão | Pergunta |
|---|---|
| Discriminação | O score ordena adequadamente genuínos e impostores? |
| Calibração | A magnitude do log-LR é compatível com os dados? |
| Transferência | O comportamento se mantém fora do conjunto usado no ajuste ou na seleção? |

Confundir essas dimensões é uma das principais fontes de mal-entendido em
discussões sobre desempenho. Em particular, **uma EER baixa não garante uma
calibração honesta**: um sistema pode ordenar muito bem e ainda assim emitir
log-LRs sistematicamente inflados ou tímidos.

### 3.2 EER (Equal Error Rate)

A EER é o ponto em que a taxa de falsos positivos iguala a taxa de falsos
negativos quando se varia um limiar de decisão sobre o score.

- **Finalidade:** medir o poder de discriminação bruto, na ausência de
  ajuste de calibração.
- **Interpretação:** EER menor indica melhor separação entre as distribuições
  de genuínos e impostores.
- **Limitação:** depende de um limiar; ignora a forma da curva fora do ponto
  de igualdade.
- **Por que não mede calibração:** transformações monotônicas crescentes do
  score (incluindo qualquer recalibração monotônica) não alteram o ranking,
  logo não alteram a EER. Dois sistemas com a mesma EER podem ter
  calibrações muito diferentes.

### 3.3 AUC (Area Under the ROC Curve)

A AUC mede a probabilidade de um par genuíno aleatório receber score maior
que um par impostor aleatório.

- **Finalidade:** medir o poder de ranking de modo integrado sobre todos os
  limiares possíveis.
- **Invariância:** AUC é invariante sob qualquer transformação monotônica
  crescente do score. Por isso, **AUC não diferencia entre log-LRs bem
  calibrados e log-LRs mal calibrados**, desde que a ordenação seja a mesma.
- **Limitação:** não responde à pergunta forense central (“o número emitido
  é honesto como força de evidência?”).

### 3.4 Cllr (Cost of Log-Likelihood-Ratio)

O Cllr (Brümmer & du Preez, 2006) é a métrica padrão-ouro de calibração
forense de log-LRs.

**Fórmula** (implementada exatamente em `faceaudit/services/cllr_math.py`):

$$
C_{\mathrm{llr}}=\frac{1}{2}\left[
  \frac{1}{N_p}\sum_{i\in H_p}\log_2\!\left(1+e^{-\ell_i}\right)
  +\frac{1}{N_d}\sum_{j\in H_d}\log_2\!\left(1+e^{+\ell_j}\right)
\right],
$$

onde $\ell_i$ é o log-LR do par $i$ (em nats), $N_p$ e $N_d$ são os números
de pares genuínos e impostores.

**Lendo a fórmula:**

- O **termo $H_p$** penaliza pares genuínos cujo log-LR é negativo (o
  sistema apontou para a hipótese errada). Se o log-LR genuíno for muito
  positivo, $\log_2(1+e^{-\ell})\approx 0$ e o custo é mínimo.
- O **termo $H_d$** penaliza pares impostores cujo log-LR é positivo.
- O **fator $1/2$** confere **peso igual** às duas classes. Isso é crucial
  porque, em biometria, $N_d\gg N_p$ tipicamente. Sem o balanceamento, a
  classe majoritária dominaria a métrica.
- A **base $\log_2$** torna o resultado expresso em bits. No código, o
  cálculo é feito em nats com `np.logaddexp(0, x) = ln(1 + eˣ)` (estável
  contra overflow) e dividido por $\ln 2$.

**Pontos de referência:**

| Sistema | $C_{\mathrm{llr}}$ | Interpretação |
|---|---|---|
| Inútil (sempre $\log\mathrm{LR}=0$, $\mathrm{LR}=1$) | $1{,}0$ bit | não informa nada |
| Perfeito | $\to 0$ | informativo e bem calibrado |
| Pior que inútil (LRs sistematicamente errados) | $>1{,}0$ | enganoso |

> **Exemplo.** Um sistema que devolve $\log\mathrm{LR}=0$ para tudo: cada
> termo vale $\log_2(1+e^{0})=\log_2 2=1$, portanto
> $C_{\mathrm{llr}}=0{,}5\cdot(1+1)=1{,}0$ bit. **Por isso Cllr = 1 é a
> “linha de inutilidade”**: qualquer calibrador útil precisa ficar abaixo.

A função `compute_cllr` também retorna `cllr_gen` e `cllr_imp` (as médias por
classe **sem** o fator $1/2$), úteis como diagnósticos isolados. Por
convenção do código, **esses dois termos não somam ao Cllr** — só o Cllr
total é uma métrica única.

### 3.5 Cllr_min (via PAVA)

O `ValidationMathService` recalcula o Cllr depois de aplicar **regressão
isotônica (PAVA)** aos log-LRs — a melhor recalibração monotônica possível
**ajustada no próprio conjunto avaliado**. O resultado é denotado
$C_{\mathrm{llr,min}}$.

**Utilidade:** diagnóstico exploratório. O gap entre Cllr e Cllr_min indica
quanto se perde por má calibração (em oposição à perda por falta de poder
discriminativo).

> ⚠️ **Cautela (Morrison 2021).** Como a PAVA é ajustada no mesmo conjunto
> que está sendo avaliado, $C_{\mathrm{llr,min}}$ é um **oráculo otimista
> in-sample**, não uma medida operacional de calibração nem um teto teórico
> real. É útil em desenvolvimento e comparação inter-estratégias, mas **não
> deve ser apresentado isoladamente em laudo** como grau de calibração.

### 3.6 cal_loss

$$
\mathrm{cal\_loss}=C_{\mathrm{llr}}-C_{\mathrm{llr,min}}
$$

Trata-se de um diagnóstico auxiliar: quantifica o gap em relação ao oráculo
PAVA in-sample. Sujeito à mesma ressalva da §3.5; não deve ser usado
isoladamente como critério de seleção operacional.

### 3.7 Cllr_affine (Brümmer)

O `ValidationMathService` também ajusta a melhor recalibração **afim**

$$
\log\mathrm{LR}_{\mathrm{recal}}=a\cdot\log\mathrm{LR}+b
$$

via regressão logística 1D quase sem regularização e classes balanceadas
(em código: `LogisticRegression(solver="lbfgs", class_weight="balanced",
max_iter=1000, C=1e6)`), e mede o Cllr resultante. Retorna
`(cllr_affine, slope a, intercept b)`. Em caso de separação perfeita, o
fallback é neutro `(cllr_original, 1, 0)`.

**Diagnóstico direto a partir de $(a, b)$:**

| Condição | Interpretação |
|---|---|
| $a\approx 1,\ b\approx 0$ | sistema bem calibrado afinamente |
| $a>1$ | LRs **excessivamente confiantes** (precisam ser puxados para zero) |
| $a<1$ | LRs **tímidos** (precisam ser amplificados) |
| $b\ne 0$ | viés sistemático numa direção |

> **Convenção exata.** A docstring do código (`_compute_affine_calibration`)
> deixa explícito que `a > 1` corresponde a *over-confident* e `a < 1` a
> *under-confident*. Esta documentação adota a mesma convenção.

### 3.8 ECE (Empirical Cross-Entropy)

A ECE é o custo de entropia cruzada calculado ao longo de uma grade de
*priors* (no FaceAudit: 100 pontos de log-odds a priori entre $-3{,}5$ e
$+3{,}5$, em log₁₀). Para cada *prior* da grade, compara-se a ECE real à ECE
mínima (obtida pela PAVA).

A partir da curva, derivam-se três escalares de diagnóstico:

- `ece_calibration_loss_area`: integral trapezoidal do gap não-negativo
  (área entre a curva real e a curva mínima);
- `ece_max_gap`: pico do gap;
- `ece_argmax_prior_log_odds`: o prior em que esse pico ocorre.

**Limitações:** depende da grade de priors e da definição do mínimo PAVA
(in-sample). Usar como diagnóstico, junto às demais métricas.

### 3.9 Transfer gap

A diferença sistemática entre desempenho **interno** (no conjunto usado para
ajustar e selecionar) e **externo** (em um conjunto independente, intocado).
Um transfer gap pequeno indica que a calibração se mantém fora do conjunto
de seleção; um gap grande sugere sobreajuste ou mudança de domínio.

Operacionalmente, o transfer gap é avaliado entre $D_{\mathrm{dev}}$ e
$D_{\mathrm{val}}$, conforme definido em §7.1.

### 3.10 Tippett plots e inspeção das caudas

Os Tippett plots mostram, para cada classe, a fração acumulada de log-LRs
acima/abaixo de cada valor. Eles permitem inspecionar visualmente o
comportamento das caudas — onde o ruído amostral é mais alto e onde o risco
de produzir LRs extremos pouco sustentados é maior.

Ao interpretar caudas, é imprescindível considerar **o suporte amostral em
cada região** (quantos pares de cada classe efetivamente cobrem a faixa).
Inferências sobre caudas com pouco suporte são fracas.

---

## 4. Família discriminativa — regressão logística

Esta família compreende dois métodos relacionados: a regressão logística
linear (4.1), e a sua extensão **aninhada** com expansão piecewise-linear
monotônica (4.2). A piecewise não é uma família independente: ela contém a
linear como caso particular (ver §4.2.10).

### 4.1 Regressão logística linear

#### 4.1.1 Intuição

A regressão logística linear aprende uma **reta** que mapeia o score bruto
diretamente para o log-LR. É o modelo mais parcimonioso, estável e
defensável.

#### 4.1.2 Formulação

$$
\log\mathrm{LR}(s)=\beta_0+\beta_1 s.
$$

A inclinação $\beta_1$ é constante em toda a faixa do score.

#### 4.1.3 Interpretação dos coeficientes

- $\beta_1$ é a inclinação: quanto o log-LR muda por unidade de score. Se o
  score está orientado de modo que valores maiores indicam maior similaridade,
  espera-se $\beta_1>0$.
- O ponto $s^\star=-\beta_0/\beta_1$ é onde $\log\mathrm{LR}=0$ (a evidência
  neutra). Não é um limiar de decisão; é apenas o cruzamento da reta com
  zero na escala calibrada.

#### 4.1.4 Exemplo numérico

Suponha que o ajuste sobre um conjunto de calibração com scores cosseno
tenha produzido $\beta_0=-4$ e $\beta_1=8$:

$$
\log\mathrm{LR}(s)=-4+8s.
$$

Avaliando em três pontos representativos:

| $s$ | $\log\mathrm{LR}$ | LR aproximada | Leitura |
|---|---|---|---|
| $0{,}25$ | $-2$ | $\sim 1{:}7$ | favorece $H_d$ |
| $0{,}50$ | $0$ | $1{:}1$ | evidência neutra |
| $0{,}75$ | $+2$ | $\sim 7{:}1$ | favorece $H_p$ |

#### 4.1.5 Convenção adotada no FaceAudit

A implementação utiliza
`LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000)`.
A leitura matemática precisa é a seguinte.

A função `decision_function` da regressão logística devolve o **logit da
probabilidade posterior** sob o prior implícito do treinamento. Para que
esse valor possa ser lido **diretamente como log-LR**, é necessário que o
prior usado no treino corresponda a $1{:}1$ entre $H_p$ e $H_d$ (convenção
de Brümmer). O parâmetro `class_weight="balanced"` reponderá efetivamente
as amostras de modo a corresponder a essa condição. Sob outras
configurações, o logit conteria uma correção de prior não nula que precisaria
ser subtraída.

O FaceAudit **não** aplica essa subtração silenciosamente. Em vez disso, ele
verifica explicitamente: a função `_estimator_prior_log_odds_correction`
retorna $0$ apenas quando `class_weight == "balanced"`; em qualquer outra
configuração, ela lança `RuntimeError`. Isso é deliberado: evita que um log-LR
contaminado por prior seja entregue por engano. **Não se deve, portanto,
afirmar de forma geral que `class_weight="balanced"` é, sozinho, suficiente
para garantir a interpretação como log-LR — a garantia decorre da combinação
desse balanceamento com a checagem explícita no código.**

Sob essa convenção:

- a saída do calibrador é o log-LR **sem clipping** (a docstring de
  `calibrate_scores` explicita que um clip antigo de $\pm 34{,}5$ foi
  removido — qualquer clip operacional deve ser política externa, ver §10);
- a regularização L2 é controlada pelo parâmetro $C$ (inverso da força). O
  $C$ é selecionado por GridSearch por padrão (§4.1.6).

#### 4.1.6 Scoring (objetivo do GridSearch)

A escolha da métrica que orienta o GridSearch de $C$ tem efeito direto sobre
a qualidade da calibração entregue.

| Métrica (enum) | Comentário |
|---|---|
| `NEG_CLLR_BALANCED` (default) | Equivalente a $-C_{\mathrm{llr}}$. Atribui peso igual a $H_p$ e $H_d$ — **recomendado para calibração forense**. Não é dominada pela classe majoritária em datasets desbalanceados. |
| `NEG_LOG_LOSS` | log-loss média **por amostra**. Em datasets onde $N_d\gg N_p$ (caso típico em comparação biométrica), tende a ser dominada pela classe majoritária. Não recomendado para calibração forense. |
| `ROC_AUC` | Mede capacidade de ranking. Não premia calibração: dois modelos com a mesma curva ROC têm AUC idêntica, ainda que um seja mal calibrado. Útil quando o foco é separação, não LR honesto. |

#### 4.1.7 Modos de validação cruzada

A escolha de $C$ via GridSearch envolve uma decisão sutil mas importante:
**como dividir os dados em folds**. Se pares com **a mesma pessoa**
aparecerem tanto no treino quanto no teste de um mesmo fold, a estimativa
do desempenho fica otimista.

| Modo | Splitter | Disjunção de identidade | Quando usar |
|---|---|---|---|
| `AUTO` (default) | escolhe `GROUPED_PROBE` se há identidades de probe; senão `PAIR` | depende | default seguro |
| `PAIR` | `StratifiedKFold` | **nenhuma** | otimista; fallback quando não há identidades |
| `GROUPED_PROBE` | `StratifiedGroupKFold` por identidade do probe | só do probe (vazamento residual pela referência) | padrão V2.8 |
| `STRICT_IDENTITY` | `IdentityDisjointPairKFold` (custom) | **total** — identidades 100 % disjuntas, refit final em $D_{\mathrm{cal}}$ completo | mais defensável quando há identidades suficientes |

> **Alias legado:** `STRICT_BY_IDENTITY` é reconhecido como sinônimo de
> `STRICT_IDENTITY` por `from_str`.

> **Importante.** `gridsearch_cv_mode` (acima) é **distinto** de
> `cv_split_mode`. O primeiro governa o GridSearch interno de $C$. O segundo
> governa apenas a CV diagnóstica opcional (`k_folds > 0`), usada para
> estimar o Cllr fora-da-amostra sem afetar a seleção de hiperparâmetros.

#### 4.1.8 Parâmetros principais

A tabela completa de campos (com todos os defaults, restrições e referências
de código) está no **Apêndice B**. Apresentam-se aqui os campos essenciais
para o uso operacional:

| Campo | Default | Função |
|---|---|---|
| `use_gridsearch` | `True` | ativa a busca de $C$ |
| `c_grid` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | grade de $C$ testada |
| `cv_folds` | `5` | número de folds do GridSearch |
| `gridsearch_cv_mode` | `AUTO` | particionamento dos folds (§4.1.7) |
| `scoring` | `NEG_CLLR_BALANCED` | métrica do GridSearch (§4.1.6) |
| `piecewise_linear_config` | `mode=NONE` | expansão piecewise (§4.2); desativada por padrão |

#### 4.1.9 Caminho per-fold para arquiteturas LR (V2.14)

Quando o dataset vetorizado vem de `LR_ARCHITECTURE_VECTORIZATION` com
`parameter_mode=FIT_PER_FOLD`, o `CALIBRATION_MODEL_TRAINING` detecta a flag
`defer_to_calibrator_per_fold=True` no metadata e usa um caminho separado:

- Carrega o `LR_ARCHITECTURE_RAW_INPUT` cru (matriz `<f8` C-contiguous read-only
  com colunas `[base, r_F, r_P]`) referenciado pelo dataset vetorizado.
- Constrói `Pipeline([ArchitectureVectorizerTransformer → StandardScaler →
  LogisticRegression])` e roda `GridSearchCV(cv=IdentityDisjointPairKFold(
  probe_identities=..., reference_identities=...))` sobre `c_grid`. O
  splitter aplica disjunção de identidades nos DOIS lados do par (§4.1.7
  `STRICT_IDENTITY`).
- O transformer ajusta s₀ via AUTO_LOGREG e r̄_F/r̄_P como média **APENAS
  sobre o `X_train` do fold** — eliminando o leakage residual do fluxo legado
  `FIT_FULL_CALIBRATION_LEGACY` (em que s₀/r̄ são estimados sobre todo D_cal
  ANTES da CV do calibrador).
- Após `refit=True`, o `best_estimator_.named_steps["arch_vec"].state_`
  reflete refit em D_cal completo. O state é desempacotado num
  `LogisticRegressionCalibrationModel` que preserva o contrato downstream
  (coeficientes acessíveis via `logreg.coef_`, scaler embutido populando
  `_feature_mean`/`_feature_std`, `best_C_` exposto).
- Metadata persistido: `selected_C`, `inner_cv_splitter=
  IdentityDisjointPairKFold`, `inner_cv_fold_count`,
  `strict_identity_disjoint=True`, `fit_per_fold=True`,
  `final_refit_on_full_calibration=True`, `architecture_state` (consumível por
  `INHERIT` em VALIDATION/CASEWORK), `standardized_coefficients`,
  `raw_scale_coefficients`, `raw_scale_intercept`, `vectorization_contract_
  fingerprint`.

Não-mudanças: para datasets que vêm de `SCORE_VECTORIZATION` puro ou de
`LR_ARCHITECTURE_VECTORIZATION` em `FIT_FULL_CALIBRATION_LEGACY`, o caminho
descrito em §4.1.1–§4.1.8 permanece — LogReg simples sobre a matriz
`score_vector_matrix` com os mesmos splitters do `gridsearch_cv_mode`.
Documentado em escopo em `CLAUDE.md` (seção "Arquitetura LR declarativa") e
nos testes `tests/test_per_fold_transformer.py`.

### 4.2 Expansão piecewise-linear monotônica

Esta seção fornece o resumo necessário ao posicionamento dentro do guia. O
aprofundamento metodológico — seleção de knots, anti-cherry-picking,
parcimônia, benchmark sintético, estudo de caso, perguntas críticas — está em
`calibracao_piecewise_linear_faceaudit_relatorio_integrado.md`. Os detalhes
internos de implementação estão em `docs/piecewise_linear_calibration.md`.

#### 4.2.1 Papel no documento

Aqui se trata da **forma funcional** e do **catálogo de configurações**. A
discussão metodológica completa fica nos documentos especializados.

#### 4.2.2 Régua articulada

A regressão logística linear desenha uma régua reta convertendo score em
log-LR; a inclinação é única. A calibração piecewise-linear instala
**dobradiças** nessa régua. A curva permanece **conectada** (contínua) em
cada dobradiça, mas pode passar a crescer mais rapidamente ou mais lentamente
a partir dali. Essas dobradiças são os **knots**.

#### 4.2.3 Função hinge

A peça matemática básica é a função hinge:

$$
h_k(s)=\max(s-k,0)=
\begin{cases}
0, & s\le k,\\
s-k, & s>k.
\end{cases}
$$

Antes do knot a feature vale zero; depois, cresce linearmente.

#### 4.2.4 Uma dobra

Com um único knot $k_1$:

$$
\log\mathrm{LR}(s)=\beta_0+\beta_1 s+\gamma_1\max(s-k_1,0).
$$

As inclinações por trecho são:

$$
m_0=\beta_1\quad(\text{antes do knot}),\qquad
m_1=\beta_1+\gamma_1\quad(\text{depois do knot}).
$$

Se $\gamma_1>0$, a curva fica mais inclinada depois da dobra; se
$\gamma_1<0$, a inclinação **diminui**. Uma hinge não serve apenas para
“aumentar” a inclinação — ela permite tanto aceleração quanto desaceleração.

#### 4.2.5 Duas dobras

Com dois knots $k_1<k_2$:

$$
\log\mathrm{LR}(s)=\beta_0+\beta_1 s+\gamma_1\max(s-k_1,0)+\gamma_2\max(s-k_2,0).
$$

Inclinações por trecho:

$$
m_0=\beta_1,\qquad m_1=\beta_1+\gamma_1,\qquad m_2=\beta_1+\gamma_1+\gamma_2.
$$

#### 4.2.6 Forma geral adotada no FaceAudit

Internamente, usa-se uma variável centrada na âncora $s_0$:

$$
z=s-s_0.
$$

Com offsets ordenados $\delta_1<\cdots<\delta_K$, a expansão é

$$
\boldsymbol{\phi}(s)=\Big[\,z,\ \max(z-\delta_1,0),\ \ldots,\ \max(z-\delta_K,0)\,\Big],
$$

e a regressão logística é treinada sobre $\boldsymbol{\phi}(s)$:

$$
\log\mathrm{LR}(s)\approx
\beta_0+\beta_z\,z+\sum_{j=1}^{K}\beta_j\,\max(z-\delta_j,0).
$$

Cada knot acrescenta um parâmetro. Com $K$ knots, o calibrador tem $K+2$
parâmetros principais (intercepto, inclinação base, um incremento por knot).
As inclinações por trecho (na escala bruta) são **somas cumulativas** dos
coeficientes:

$$
m_0=\beta_z,\qquad m_r=\beta_z+\sum_{j=1}^{r}\beta_j.
$$

#### 4.2.7 Âncora e escalas

A âncora $s_0$ é uma referência **geométrica**. Não é um threshold decisório.
No FaceAudit, $s_0$ pode ser estimada automaticamente por uma regressão
logística auxiliar univariada (`AUTO_LOGREG`, default) — caso em que
corresponde ao score bruto que o auxiliar associa a log-LR igual a zero —
ou informada manualmente (`MANUAL`).

Os offsets dos knots podem ser expressos em duas escalas:

| Escala | Tabela de leitura |
|---|---|
| `AUX_LOG_LR` (recomendada com `AUTO_LOGREG`) | em unidades de log-LR do auxiliar; cada $\ln 10\approx 2{,}30$ corresponde a $\mathrm{LR}=10$. Independe do modelo de similaridade. |
| `RAW` | diretamente em unidades do score (cosseno em $[-1, 1]$, PLDA em escala própria). Necessária com `MANUAL`. |

Tabela auxiliar de configurações típicas:

| Offsets em `AUX_LOG_LR` | Interpretação aproximada |
|---|---|
| `[0]` | uma dobra na região neutra do auxiliar ($\mathrm{LR}\approx 1$) |
| `[-ln(10), 0, +ln(10)]` | três dobras em torno de $1{:}10$, $1{:}1$ e $10{:}1$ |
| `[-ln(100), 0, +ln(100)]` | três dobras em torno de $1{:}100$, $1{:}1$ e $100{:}1$ |

A escala `AUX_LOG_LR` é a do calibrador auxiliar, **não** a da curva
piecewise final. Depois do ajuste das inclinações por trecho, o log-LR final
pode divergir da escala auxiliar.

> **Restrição importante.** `AUX_LOG_LR` exige `hinge_anchor_mode = AUTO_LOGREG`
> — sem o auxiliar não há `aux_coef` para converter offsets entre escalas.
> Com `MANUAL`, use `RAW`.

#### 4.2.8 Posicionamento dos knots

| Modo (`knot_placement`) | Comportamento | Estado |
|---|---|---|
| `MANUAL` (default) | o usuário escreve `knot_offsets` | consolidado |
| `AUTO_FROM_MIRRORED_MEDIANS` | o transformer popula automaticamente dois knots posicionados em medianas auxiliares refletidas em torno de log-LR = 0; exige `AUTO_LOGREG` | **experimental** — heurística não estabelecida na literatura; validar por domínio |

> O nome serializado de `AUTO_FROM_MIRRORED_MEDIANS` é a string legada
> `"AUTO_FROM_MEDIANS"` (preservada por compatibilidade de pickles).

#### 4.2.9 Continuidade e monotonicidade são conceitos distintos

A **continuidade** decorre automaticamente da basis hinge: não há saltos nos
knots.

A **monotonicidade** precisa ser verificada. Para que scores maiores não
produzam log-LRs menores, todas as inclinações acumuladas têm de permanecer
estritamente positivas:

$$
m_r>0\quad\text{para todo trecho }r.
$$

A política operacional `monotonicity_policy` controla a ação ao detectar
violação:

- `ERROR` (default) — **rejeita** o calibrador com erro listando as
  inclinações por trecho. Defensável: um calibrador não monotônico é
  incoerente.
- `WARNING` — permite com aviso forte. **Não usar em pipeline pericial**;
  serve apenas para diagnóstico (descobrir causas: multicolinearidade,
  regularização insuficiente, knots mal posicionados).

Quando o GridSearch está ativo, a seleção de $C$ é restrita: avalia-se cada
$C$ em ordem decrescente de score de CV e seleciona-se o primeiro que produza
curva estritamente monotônica. Se nenhum for monotônico, `ERROR` falha;
`WARNING` cai no melhor por CV.

##### Gate `CONDITIONAL_MONOTONICITY` para arquiteturas declarativas (V1.5)

A política `monotonicity_policy` descrita acima opera no `LogisticRegressionCalibrationModel`
**piecewise univariado clássico**. A partir da V1.5 da `ConditionalMonotonicityRule`,
calibradores treinados via `LR_ARCHITECTURE_VECTORIZATION` (qualquer das 15
arquiteturas — incluindo as ONE_FACE V2.14.1) recebem **auditoria adicional pós-refit**
embutida no `CalibrationModelTrainingInstructionExecutor`. A regra:

- Verifica `∂ log_LR / ∂ s > 0` em **todos os pares observados em $D_{\text{cal}}$**
  (`observed_rows`) e nos **cantos do retângulo $[P_5, P_{95}]$** das raridades
  observadas (`central_box`).
- Cobre arquiteturas com derivada multivariada que dependem de raridades
  $(\tilde r_F, \tilde r_P)$ ou $\tilde r_{\text{single}}$ — caso onde
  `monotonicity_policy` univariado não se aplica.
- Persiste `monotonicity_audit` no `calibrator.metadata` com status
  (`OK / WARNING / CRITICAL / NOT_AUDITED`), slopes por região, pior par observado
  e pior canto da caixa central.
- Reportado no CLI durante o `CALIBRATION_MODEL_TRAINING` (`🛡️ OK / ⚠ WARNING / ❌ CRITICAL`).

O suporte ao piecewise univariado legado (com $K$ knots arbitrários) pela mesma
regra está enfileirado como `PiecewiseLinearDerivativeProvider` na próxima
iteração. Detalhes em `docs/conditional_monotonicity_rule_design.md`.

#### 4.2.10 Parcimônia

Cada knot adiciona um parâmetro. Mais knots não significam, automaticamente,
melhor calibração. O benchmark sintético do FaceAudit (cenários com verdade
conhecida) mostra que **knots além do necessário não melhoram** e podem
**desestabilizar caudas** com pouco suporte (segmentos com poucos pares
produzem inclinações instáveis).

A regra de seleção operacional é:

> Usar a **menor complexidade** que demonstre ganho **reproduzível** fora
> da amostra utilizada para seleção, **com suporte suficiente em cada
> segmento**.

A regressão logística linear está **contida** na família piecewise: se os
coeficientes de hinge forem todos próximos de zero, a curva aproxima uma
reta. Isso facilita comparar ganho marginal e justifica chamar a piecewise
de extensão **aninhada**, não de família independente.

O FaceAudit persiste, por segmento, os health checks que sustentam essa
disciplina (`segment_counts_hp`, `segment_counts_hd`,
`segment_support_warnings`, `min_segment_slope`).

#### 4.2.11 Ponteiros para aprofundamento

- **Seleção de knots e prevenção de cherry-picking:**
  `calibracao_piecewise_linear_faceaudit_relatorio_integrado.md`, §7.
- **Benchmark sintético** (cenários com oráculo, escada de knots,
  parcimônia, sweep de N): mesmo relatório, §9; e
  `experiments/calibration/RELATORIO_comparacao_calibradores.md`.
- **Estudo de caso** (desenvolvimento × validação externa): mesmo
  relatório, §10.
- **Especificação interna e validações**: `docs/piecewise_linear_calibration.md`.

#### 4.2.12 Parâmetros principais

A tabela completa está no **Apêndice C**. Campos essenciais para uso:

| Campo | Default | Função |
|---|---|---|
| `mode` | `NONE` | `NONE` desativa; `PIECEWISE_LINEAR` ativa a expansão |
| `hinge_anchor_mode` | `AUTO_LOGREG` | como resolver $s_0$ |
| `knot_offsets` | `[0.0]` | offsets dos knots |
| `knot_scale` | `AUX_LOG_LR` (construtor); `RAW` (parser de pickles legados) | escala dos offsets |
| `knot_placement` | `MANUAL` | posicionamento manual ou automático (experimental) |
| `require_monotonicity` | `True` | verifica monotonicidade no fim do ajuste |
| `monotonicity_policy` | `ERROR` | ação ao detectar violação |
| `min_allowed_slope` | `1e-8` | inclinação mínima admissível por segmento |

> **Divergência de defaults a registrar.** O construtor do dataclass cria
> `knot_scale = AUX_LOG_LR`. O `from_dict` (usado ao desserializar pickles
> antigos) cai em `RAW` quando o campo está ausente — preserva compatibilidade
> com calibradores treinados antes da introdução do enum. A divergência é
> intencional e deve aparecer no Apêndice C.

---

## 5. Família generativa — KDE

### 5.1 Intuição

Em vez de assumir uma forma para o mapeamento `score → log-LR`, o KDE
**estima diretamente** as duas distribuições de score — uma sob $H_p$, outra
sob $H_d$ — e calcula a diferença entre as log-densidades.

### 5.2 Formulação matemática

$$
\widehat{\log\mathrm{LR}}(s)=\log\hat p(s\mid H_p)-\log\hat p(s\mid H_d),
$$

com $\hat p(\cdot\mid H_p)$ e $\hat p(\cdot\mid H_d)$ estimadas por *kernel
density estimation* em um espaço transformado (ver §5.4).

### 5.3 Clipping operacional

A implementação aplica, **sempre**, um clipping operacional sobre o log-LR
estimado:

$$
\log\mathrm{LR}_{\mathrm{op}}(s)=\mathrm{clip}\!\bigl(\widehat{\log\mathrm{LR}}(s),\,L_{\min},\,L_{\max}\bigr).
$$

No FaceAudit, $L_{\min}=-20$ e $L_{\max}=+20$ por default. O clip é aplicado
em dois lugares: na construção da curva interpolada de calibração (em escala
log-LR) **e** após o lookup na inferência (em escala LR, com $e^{L_{\min}}$ e
$e^{L_{\max}}$).

> **Distinção conceitual importante.** O clipping é uma **proteção
> operacional** que vive na implementação do KDE no FaceAudit; não é parte
> intrínseca do estimador KDE como objeto matemático. Em política
> operacional, esse clip deve ser entendido como decisão sobre **valor
> reportável** (ver §10), não como propriedade do método.

### 5.4 Normalização

Antes da estimação, scores são padronizados (z-score) com a média $\mu$ e o
desvio-padrão $\sigma$ aprendidos sobre o conjunto de treino concatenado
(genuínos $\cup$ impostores). Esses parâmetros são persistidos e reaplicados
na inferência.

> **Atenção.** A normalização do KDE no FaceAudit é **z-score**, **não**
> transformação de Fisher. O campo legado `fisher_eps` foi descartado e o
> parser silenciosamente o ignora. Tabelas antigas que descrevam Fisher para
> KDE no FaceAudit estão desatualizadas.

### 5.5 Bandwidth

O parâmetro $h$ (bandwidth) controla a suavidade do estimador. Há um trade-off
clássico:

- $h$ pequeno → densidade espinhada, baixo viés, **alta variância**, risco
  de sobreajuste e instabilidade nas caudas;
- $h$ grande → densidade suave, baixo variância, **alto viés**, risco de
  borrar detalhe relevante.

O FaceAudit oferece bandwidths separados para genuínos e impostores
(`bandwidth_genuine`, `bandwidth_impostor`). Por padrão (`use_gridsearch=True`),
cada um é otimizado por GridSearch no espaço z-score, sobre uma grade
log-espaçada com expansões automáticas caso o ótimo caia na borda.

### 5.6 Grade e interpolação

A curva $s\mapsto\log\mathrm{LR}(s)$ é precomputada sobre uma grade densa de
scores que cobre a faixa observada com margem de $\pm 5\sigma$, e
armazenada como interpolador linear (`scipy.interpolate.interp1d`). Em
inferência, scores fora da faixa coberta são clampados aos valores de borda
do interpolador. Esse comportamento de borda é uma **decisão operacional**
de extrapolação (ver §10).

### 5.7 Exemplo numérico

Suponha que, para $s=0{,}65$, os estimadores devolvam densidades
$\hat p(s\mid H_p)=3{,}0$ e $\hat p(s\mid H_d)=0{,}1$. Então

$$
\widehat{\log\mathrm{LR}}(0{,}65)=\log(3{,}0)-\log(0{,}1)=\log(30)\approx 3{,}40\ \text{nats},
$$

correspondendo a $\mathrm{LR}\approx 30$. Como $|3{,}40|<L_{\max}=20$, o
clipping não atua.

### 5.8 Limitações

- **Sensibilidade ao bandwidth**: bandwidths mal escolhidos podem distorcer
  as caudas;
- **Suporte insuficiente nas caudas**: regiões com poucas amostras geram
  estimativas instáveis de log-LR;
- **Extrapolação**: fora da faixa observada o interpolador clamps no valor
  de borda — é necessário tratar isso como decisão operacional explícita;
- **Baixa densidade**: scores em regiões pouco povoadas combinam densidade
  pequena no numerador e no denominador, e pequenas flutuações são
  amplificadas;
- **Assimetria de cobertura entre as classes**: se uma das classes tem
  cobertura ruim numa faixa, o log-LR estimado nessa faixa não é confiável.

### 5.9 Parâmetros principais

A tabela completa está no **Apêndice D**. Campos essenciais:

| Campo | Default | Função |
|---|---|---|
| `kernel` | `"gaussian"` | kernel do `KernelDensity` |
| `use_gridsearch` | `True` | otimiza bandwidths por GridSearch |
| `bandwidth_genuine` | `0.1` | bandwidth de genuínos (usado se GridSearch desligado) |
| `bandwidth_impostor` | `0.1` | bandwidth de impostores (idem) |
| `min_log_lr` | `-20.0` | clip inferior (**sempre ativo**) |
| `max_log_lr` | `+20.0` | clip superior (**sempre ativo**) |
| `interpolation_points` | `10000` | pontos da curva interpolada |
| `k_folds` | `0` | CV diagnóstica opcional |
| `cv_split_mode` | `STRATIFIED_PAIR` | splitter da CV diagnóstica |

---

## 6. Família bi-gaussianizada

### 6.1 Intuição

A bi-gaussianização (Morrison 2024) **deforma** os scores por meio de uma
transformação monotônica de modo que os log-LRs calibrados sigam as
distribuições esperadas para um sistema **idealmente calibrado**: duas
gaussianas simétricas em torno de zero, com médias $\pm\sigma^2/2$ e
variância $\sigma^2$. O parâmetro $\sigma$ controla quão confiantes os
log-LRs podem ser — sistemas que separam melhor recebem $\sigma$ maior.

### 6.2 Distribuições ideais

Para o sistema-alvo:

$$
\log\mathrm{LR}\mid H_p\ \sim\ \mathcal{N}\!\left(+\sigma^{2}/2,\ \sigma^{2}\right),
$$

$$
\log\mathrm{LR}\mid H_d\ \sim\ \mathcal{N}\!\left(-\sigma^{2}/2,\ \sigma^{2}\right).
$$

Esse formato com médias simétricas $\pm\sigma^2/2$ é a assinatura matemática
de um sistema de LR idealmente calibrado.

### 6.3 Transformação conceitual

$$
\log\mathrm{LR}(s)=F^{-1}\!\bigl(G(s)\bigr),
$$

onde:

- $G(s)$ é a **CDF empírica conjunta** dos scores de treino (construída
  combinando as distribuições das duas classes; ver §6.5);
- $F$ é a CDF da **mistura ideal** das duas gaussianas
  $\mathcal{N}(\pm\sigma^2/2,\sigma^2)$:
  $F(z)=\tfrac12\Phi\!\bigl((z-\sigma^2/2)/\sigma\bigr)+\tfrac12\Phi\!\bigl((z+\sigma^2/2)/\sigma\bigr)$;
- $F^{-1}$ é sua inversa (computada por interpolação ou por bracket de
  Brent, ver §6.5).

A transformação é monotônica por construção: $G$ e $F^{-1}$ são ambas
monotônicas. A monotonicidade da curva calibrada não precisa, portanto, ser
verificada como condição extra.

### 6.4 Estimação de $\sigma$

O parâmetro de escala $\sigma$ é determinado a partir de uma **métrica-alvo
medida sobre o treino** (`target_metric`). Dois modos:

**Modo EER** — forma fechada, sem calibrador inicial:

$$
\sigma=-2\,\Phi^{-1}(\mathrm{EER}).
$$

**Modo Cllr** — regressão empírica de Morrison; sendo $C_{\mathrm{llr}}$
medido sobre as saídas do **calibrador inicial** aplicado ao próprio treino:

$$
\sigma^{2}=-\frac{1}{c}\,\ln\!\left(\frac{1}{b}\,\ln C_{\mathrm{llr}}+1\right).
$$

**Valores das constantes (conferidos no código,
`bi_gaussianized_calibration_model.py`):**

$$
b=17{,}7,\qquad c=9{,}33\times 10^{-3}.
$$

Em ambos os modos, $\sigma^2$ é em seguida limitado por um cap numérico de
segurança ($1{,}0\times 10^{3}$) e pode ser encolhido por
`sigma_shrinkage_factor` $\in (0, 1]$ (encolhe LRs em direção a 1; cautela
operacional).

> **Cllr menor → $\sigma$ maior → LRs mais confiantes.** Cada sistema recebe
> autorização para ser tão confiante quanto seu desempenho mensurável
> justifica.

### 6.5 Configurações

A escolha das peças internas é a alavanca de calibração mais importante.

**Calibrador inicial** (`initial_calibrator`) — usado apenas para **estimar
$\sigma$** via a métrica; o mapa final sempre passa por $G$ e $F^{-1}$:

| Valor | Comportamento |
|---|---|
| `NONE` | assume que os scores **já** estão em escala log-LR (ex.: PLDA log-LR) |
| `LOGREG` | rápido; assume linearidade no espaço do score |
| `KDE` | mais robusto a outliers (recomendado pelo artigo) |

**Construção de $G(x)$** (`gx_method`):

| Valor | Construção |
|---|---|
| `EMPIRICAL_LINEAR` | CDF empírica piecewise-linear pura (default original do artigo) |
| `KDE_SMOOTHED` | $G(x)=\tfrac12 \hat F_g(x)+\tfrac12 \hat F_i(x)$ com CDFs suavizadas por kernel gaussiano. Default no FaceAudit. Mais suave → calibração mais generalizável |

O FaceAudit usa, no `KDE_SMOOTHED`, um **fator multiplicativo de bandwidth**
(`gx_kde_bandwidth_factor`, default $5{,}0$). Esta é a alavanca mais sensível:
valores maiores tornam $G$ mais suave e a calibração mais robusta a flutuações
amostrais. O valor $5{,}0$ é o ponto empiricamente validado no FaceAudit
(documentado no docstring da config).

**Métrica-alvo** (`target_metric`): `CLLR` (default) ou `EER`. Em `EER`, o
calibrador inicial não é utilizado.

**Inversão de $F$** (`inversion_method`):

| Valor | Característica |
|---|---|
| `INTERP1D` (default) | precomputa $F$ sobre grade e usa `interp1d`. Rápido |
| `BRENT_ROOT` | `scipy.optimize.brentq` por ponto. Preciso, lento |

**Estimação de $\sigma$** (`sigma_estimation_mode`):

| Valor | Característica |
|---|---|
| `IN_SAMPLE` | calibrador inicial ajustado e avaliado no mesmo dataset |
| `CROSS_FITTED` (default) | k-fold estratificado; reduz overfit do calibrador inicial |

O parâmetro `sigma_cv_folds` controla o $k$ desse k-fold interno e **não se
confunde** com `k_folds`, que é a CV diagnóstica do pipeline completo (ver
§3 e §8.2).

**Direção do score** (`score_direction`):

| Valor | Comportamento |
|---|---|
| `AUTO` (default) | detecta pela média das duas classes |
| `HIGHER_IS_GENUINE` | scores maiores indicam $H_p$ (cosseno, PLDA típico) |
| `LOWER_IS_GENUINE` | scores maiores indicam $H_d$ (distância) — inverte o sinal antes do mapeamento |

**Clipping final** (`enable_clipping`): por padrão **desligado**
(`False`). Há, porém, dois clamps internos sempre ativos:

- $u=G(s)$ é truncado em $[\varepsilon,1-\varepsilon]$ antes da inversão de
  $F$ (evita $\pm\infty$);
- `gx_clamp_to_training_range` (default `True`) projeta scores fora da faixa
  do treino na borda observada (não-extrapolação à la Morrison §2.7;
  produz platôs visíveis nas caudas — comportamento operacional, não
  patologia).

### 6.6 Exemplo conceitual

Considere dois sistemas A e B, ambos baseados em PLDA, mas com qualidades
distintas.

1. O calibrador inicial é aplicado em cada conjunto de treino, e mede-se o
   Cllr. Sistema A separa muito bem: Cllr $\approx 0{,}2$. Sistema B separa
   pouco: Cllr $\approx 0{,}8$.
2. Pela regressão $\sigma\leftrightarrow C_{\mathrm{llr}}$ (§6.4), A recebe
   $\sigma$ grande; B recebe $\sigma$ pequeno.
3. Na inferência, um score $s$ é submetido ao seguinte mapeamento:
   - $G(s)$ devolve sua posição percentual na CDF empírica conjunta;
   - $F^{-1}$ devolve o log-LR ideal correspondente a essa posição, sob a
     mistura $\mathcal{N}(\pm\sigma^2/2,\sigma^2)$.
4. Resultado: o sistema A, autorizado a ser mais confiante, pode produzir
   log-LRs em torno de $\pm 7$ para scores extremos; o sistema B, com
   $\sigma$ menor, fica mais perto de zero. Cada um é confiante apenas na
   medida em que seu desempenho justifica.

### 6.7 Limitações

- **Dependência do calibrador inicial** (modo CLLR): se a transformação
  inicial for inadequada, o $\sigma$ derivado também o será;
- **Sensibilidade às caudas** de $G$, especialmente com `EMPIRICAL_LINEAR` em
  conjuntos pequenos;
- **Suporte amostral**: ambos $G$ e a estimação de $\sigma$ dependem de
  cobertura razoável;
- **Extrapolação**: fora do treino, o clamp opera por design — registrar como
  política de reportabilidade (§10);
- **Diferenças entre comportamento teórico e empírico**: a regressão
  $\sigma\leftrightarrow C_{\mathrm{llr}}$ é empírica (constantes $b, c$
  ajustadas pelo autor); resultados podem divergir em domínios não
  representados no ajuste original;
- **Necessidade de validação no domínio**: o método possui motivação teórica
  própria, mas seu comportamento em cada domínio precisa ser confirmado
  empiricamente.

### 6.8 Estado metodológico

A bi-gaussianizada não deve ser tratada como solução universal. Possui
motivação teórica explícita e é uma alternativa legítima onde se aplica.
Resultados de comparação em corpora específicos delimitam **adequação ao
domínio**; não demonstram inadequação geral. Recomenda-se compará-la,
sempre, com a regressão logística como baseline.

### 6.9 Parâmetros principais

A tabela completa está no **Apêndice E**. Campos essenciais:

| Campo | Default | Função |
|---|---|---|
| `target_metric` | `CLLR` | métrica que define $\sigma$ |
| `initial_calibrator` | `LOGREG` | calibrador inicial p/ estimar $\sigma$ |
| `gx_method` | `KDE_SMOOTHED` | construção de $G(x)$ |
| `gx_kde_bandwidth_factor` | `5.0` | multiplicador da bandwidth de $G$ — alavanca crítica |
| `sigma_estimation_mode` | `CROSS_FITTED` | como estimar $C_{\mathrm{llr}}$ |
| `inversion_method` | `INTERP1D` | inversão de $F$ |
| `score_direction` | `AUTO` | direção do score |
| `enable_clipping` | `False` | clipping final do log-LR (desligado por padrão) |
| `gx_clamp_to_training_range` | `True` | clamp interno na faixa do treino |
| `k_folds` | `0` | CV diagnóstica opcional do pipeline completo |
| `cv_split_mode` | `STRATIFIED_PAIR` | splitter da CV diagnóstica |

---

## 7. Seleção, validação e prevenção de leakage

### 7.1 Separação funcional dos datasets

A defensabilidade de uma calibração depende, antes da escolha do método, da
**separação funcional rigorosa** entre os conjuntos usados em cada etapa.

| Conjunto | Finalidade |
|---|---|
| $D_{\mathrm{emb}}$ | treinamento do embedding (modelo facial) |
| $D_{\mathrm{backend}}$ | treinamento do backend de similaridade (ex.: PLDA) |
| $D_{\mathrm{cal}}$ | ajuste dos coeficientes do calibrador |
| $D_{\mathrm{dev}}$ | seleção da família e dos hiperparâmetros (knots, $C$, bandwidth, $\sigma$) |
| $D_{\mathrm{val}}$ | avaliação externa, **após congelamento de todas as decisões** |
| caso real | aplicação, **sem retuning** |

Identidades **não devem se cruzar entre conjuntos**. A regra prática: use
splits explícitos por identidade ao construir cada conjunto e documente
quais identidades estão em cada lado.

### 7.2 Separação por identidade

Split aleatório de pares **não é suficiente**: a mesma identidade pode
aparecer em vários pares e, portanto, em vários folds. Em comparação
biométrica, o leakage por identidade é a forma mais comum e mais sutil de
contaminação.

Em todas as CVs e splits funcionais, prefira agrupamento por identidade
(`STRICT_IDENTITY` ou `GROUPED_PROBE`, na seleção de $C$ da LogReg; ou
`STRATIFIED_GROUP` na CV diagnóstica), e documente o **nível de disjunção**
efetivamente alcançado (probe-only vs total).

### 7.3 Outras fontes de leakage

Listar e verificar:

- a mesma identidade entre treino e teste;
- a mesma imagem;
- crops derivados (variantes do mesmo arquivo);
- frames vizinhos (de vídeo);
- *augmentations* a partir do mesmo original;
- mesma sessão de captura;
- mesma fonte (mesmo dispositivo, mesma luz, mesmo ambiente);
- arquivos duplicados em corpora diferentes;
- sobreposição entre corpora (subconjuntos públicos reaproveitados).

### 7.4 Seleção do calibrador

Procedimento defensável:

1. Definir um **menu pequeno de candidatos** *a priori* (ex.: LogReg
   linear; LogReg piecewise com `[0]`; LogReg piecewise com `[-2,3, 0, +2,3]`
   em `AUX_LOG_LR`; KDE; BiGauss com `gx_kde_bandwidth_factor` em $\{3, 5,
   8\}$).
2. Ajustar cada candidato em $D_{\mathrm{cal}}$.
3. Avaliar cada candidato em $D_{\mathrm{dev}}$ (com health checks: suporte
   por segmento, monotonicidade, slopes, clipping efetivo).
4. Aplicar a **regra de parcimônia**: selecionar o candidato mais simples
   admissível dentro de uma margem prática de Cllr (ex.: $\varepsilon=0{,}001$
   bit). Detalhes em
   `experiments/calibration/RELATORIO_comparacao_calibradores.md` §3.5.
5. **Congelar** o calibrador (incluindo seu hash, paths e config completa).
6. Avaliar **uma única vez** em $D_{\mathrm{val}}$.
7. Aplicar ao caso.

Avaliar repetidamente em $D_{\mathrm{val}}$ e ajustar até obter um resultado
desejado é, por construção, uma forma de leakage do conjunto de validação no
processo de seleção.

### 7.5 Quando os dados são limitados

Em corpora pequenos, separar três conjuntos pode ser inviável. Uma
alternativa é **nested cross-validation por identidade**:

- *outer loop* por identidade para estimativa honesta de desempenho;
- *inner loop* por identidade para seleção de hiperparâmetros.

A regra essencial é manter identidades sempre disjuntas entre os níveis.

### 7.6 Bootstrap

Para intervalos de confiança:

- preferir **bootstrap por identidade** (resampla identidades, depois
  reconstrói pares) ou **bootstrap hierárquico**;
- **evitar bootstrap ingênuo de pares**: ele subestima a variância porque a
  mesma identidade aparece muitas vezes em pares correlacionados.

### 7.7 Sensibilidade

Registrar — para revisão crítica e auditoria — a variabilidade das decisões
do calibrador entre folds e seeds:

- variação dos slopes por trecho (piecewise);
- variação do $C$ selecionado;
- variação do $\sigma$ derivado (BiGauss);
- frequência com que cada candidato seria escolhido sob *resampling*;
- comportamento das caudas em cada repetição.

Uma calibração estável entre repetições é mais defensável que uma calibração
que troca de forma a cada *seed*.

---

## 8. Calibração no pipeline do FaceAudit

### 8.1 Visão geral

```text
SimilarityScoreDataset (raw_score por par + same_person)
        ↓
CALIBRATION_MODEL_TRAINING
        ↓
TRAINED_CALIBRATION_MODEL (persistido)
        ↓
SCORE_CALIBRATION
        ↓
SimilarityScoreDataset (com sample.metadata["log_lr"])
        ↓
STRATEGY_EVALUATION (Cllr, Cllr_min, Cllr_affine, ECE, EER, Tippett, ...)
```

### 8.2 `CALIBRATION_MODEL_TRAINING`

Treina um calibrador a partir de scores rotulados.

- **Entrada:** `input_resource_id` apontando para um
  `SimilarityScoreDataset` com `raw_score` e `same_person`. Identidades de
  probe/reference (quando presentes) são também extraídas — são consumidas
  pela CV de $C$ da LogReg piecewise para fold por identidade.
- **Seleção da configuração:** `calibration_model_type` $\in$
  $\{$`LOGISTIC_REGRESSION`, `KDE`, `BI_GAUSSIANIZED`$\}$ (default LogReg);
  só a config correspondente é usada (`logreg_config`, `kde_config`,
  `bigauss_config`).
- **Treinamento:** chamada `model.fit(scores, labels, logger_callback)`.
- **Validação cruzada diagnóstica:** se `k_folds >= 2`, é executada uma CV
  com hiperparâmetros **frozen** (modo `cv_hyperparameter_mode="FROZEN"`).
- **Telemetria:** Cllr in-sample sempre; Cllr fora-da-amostra se `k_folds >=
  2`. Para a LogReg piecewise, também são persistidas as inclinações por
  trecho, $s_0$, knot offsets em ambas as escalas, slopes por fold, e os
  *segment support warnings*.
- **Persistência:** um recurso `TRAINED_CALIBRATION_MODEL` é gravado
  (incluindo o pickle do modelo).

### 8.3 `SCORE_CALIBRATION`

Aplica um calibrador treinado.

- **Entrada:** o `TRAINED_CALIBRATION_MODEL` + um `SimilarityScoreDataset`
  com `raw_score`.
- **Operação:** `calibrate_scores(raw_scores)` em vetor; o resultado é
  populado no metadata de cada par: `sample.metadata["log_lr"]`.
- **Saída:** novo `SimilarityScoreDataset` (tipo `CALIBRATED_SCORES`).
  Mantém `is_casework` e a telemetria de calibração herdada.
- **Sem hiperparâmetros próprios.** Toda a configuração mora no calibrador
  treinado. Esta instrução é apenas o ato de aplicar.

### 8.4 Contrato comum (`CalibrationModelProtocol`)

Todo calibrador implementa a mesma interface:

```python
def fit(self, scores: np.ndarray, labels: np.ndarray,
        logger_callback: Callable[[str], None] = print) -> None: ...
def predict_proba(self, scores: np.ndarray) -> np.ndarray: ...   # (N, 2) com [P(Hd), P(Hp)]
def calibrate_scores(self, scores: np.ndarray) -> np.ndarray: ... # log-LR (nats)
def save(self, path: str) -> None: ...
def load(self, path: str) -> None: ...
```

E expõe:

- `name`, `is_fitted`, `requires_training` (propriedades);
- `metadata` (dicionário de telemetria e auditoria).

### 8.5 Fronteira de escopo

Quando a evidência do caso depende de mais do que similaridade — por
exemplo, quando tipicidade ou raridade da face questionada importam —, a
solução **não** é refinar a calibração univariada. É introduzir um modelo
multivariado (`SCORE_INTERACTION_FEATURE`, `SCORE_VECTORIZATION`, ou modelos
generativos próprios). Essa rota está fora do escopo deste documento;
consultar os módulos correspondentes em `faceaudit/executors/instructions/` e
em `faceaudit/enums/`.

---

## 9. Como escolher o método apropriado

Tabela orientativa (não substitui análise empírica no domínio):

| Situação | Estratégia inicial recomendada |
|---|---|
| Poucos dados ou ausência de evidência externa de não-linearidade | **regressão logística linear** (baseline obrigatório) |
| Mudança monotônica de inclinação reproduzível em $D_{\mathrm{dev}}$ | **piecewise-linear parcimoniosa** (1 a 3 knots, regra de parcimônia) |
| Densidades bem sustentadas + interesse em modelagem generativa | **KDE** |
| Aplicação compatível com a hipótese bi-gaussianizada + validação externa favorável | **bi-gaussianizada** |
| Segmentos com pouco suporte | **reduzir a complexidade** |
| Caudas instáveis | aplicar cautela, **limites e política de reportabilidade explícitas** |
| Resultados muito próximos entre candidatos | **preferir o mais simples** admissível |

Mensagem central:

> **Começar pelo modelo mais simples e promover complexidade somente quando
> o ganho for reproduzível, auditável e sustentado pelos dados.**

---

## 10. Caudas, extrapolação e reportabilidade

### 10.1 Valor bruto e valor reportável

Distinguir:

| Conceito | Definição |
|---|---|
| **Estimativa matemática** | o que o calibrador computa, sem qualquer recorte |
| **Faixa observada** | intervalo de scores efetivamente vistos em $D_{\mathrm{cal}}$ |
| **Faixa empiricamente sustentada** | subfaixa em que o suporte (densidade, contagem por classe) justifica leitura forense |
| **Política de relato** | regras explícitas que mapeiam a estimativa em valor reportável (clipping, limites, formato textual em laudo) |

### 10.2 Extrapolação

Fora do suporte amostral, todos os métodos extrapolam de algum modo:

- **LogReg linear:** extrapola linearmente a inclinação aprendida — log-LRs
  ilimitados em ambas as direções;
- **LogReg piecewise:** extrapola com a inclinação do último (ou do
  primeiro) segmento — pode ampliar incoerências se o último segmento tiver
  pouco suporte;
- **KDE:** clamps no valor de borda do interpolador (e clip $[L_{\min},
  L_{\max}]$ sempre ativo);
- **BiGauss:** `gx_clamp_to_training_range` (default `True`) projeta na
  borda observada de $G$.

Em todos os casos, **a extrapolação deve ser tratada como decisão
operacional explícita**, registrada e justificada — não como propriedade
silenciosa do método.

### 10.3 Política operacional

Recomenda-se que cada calibração entregue documente:

- **limites de reportabilidade** (faixa em que o log-LR é apresentado sem
  ressalva);
- **justificativa** dos limites (densidade de suporte; estabilidade entre
  folds);
- **versão** da política (mudanças devem ser versionadas);
- **diferenças entre calibradores** (cada família tem comportamento
  próprio nas caudas);
- **tratamento no relatório pericial** (texto-padrão para casos em que o
  log-LR extrapolaria).

---

## 11. Auditoria e rastreabilidade

Cada execução de calibração que vier a sustentar um laudo deve permitir
**reprodução fiel** anos depois. Para isso, persistir:

### 11.1 Identificação da execução

- `experiment_id`;
- versão do código (tag ou commit hash);
- data e hora (UTC);
- *seed* aleatória;
- ambiente (Python, dependências relevantes).

### 11.2 Genealogia dos corpora

- corpora de treinamento do embedding;
- corpora de treinamento do backend;
- $D_{\mathrm{cal}}$, $D_{\mathrm{dev}}$, $D_{\mathrm{val}}$;
- regras de split (por identidade, por sessão, etc.);
- **sobreposições verificadas e ausências documentadas**.

### 11.3 Configuração do calibrador

- família (`LogReg`, `KDE`, `BiGauss`);
- variante (linear / piecewise; bandwidth GridSearch / fixo; CLLR / EER);
- todos os parâmetros do dataclass;
- todos os enums com seus valores;
- defaults efetivos no momento da execução;
- hiperparâmetros selecionados (ex.: melhor $C$, bandwidths finais, $\sigma$);
- alternativas testadas e descartadas (lista do menu *a priori* +
  motivos).

### 11.4 Diagnósticos

- Cllr;
- Cllr_min;
- cal_loss;
- Cllr_affine, slope $a$, intercepto $b$;
- ECE (curva + escalares);
- EER, AUC;
- transfer gap entre $D_{\mathrm{dev}}$ e $D_{\mathrm{val}}$;
- curvas (Tippett, score×log-LR);
- *warnings* (incluindo `segment_support_warnings`);
- suporte por segmento e por classe;
- limites de reportabilidade efetivos.

### 11.5 Fonte documental dos defaults

Para cada configuração relevante, registrar:

- caminho do arquivo no repositório;
- classe ou enum;
- fábrica (se aplicável);
- parser (se aplicável);
- commit;
- data da revisão;
- eventuais divergências entre dataclass / parser / fábrica.

---

## 12. Boas práticas resumidas

1. **Manter a regressão logística linear como baseline obrigatório** em
   toda comparação.
2. **Preferir `NEG_CLLR_BALANCED`** quando o objetivo é calibrar log-LRs.
3. **Separar identidades entre folds e conjuntos funcionais**
   ($D_{\mathrm{cal}}$, $D_{\mathrm{dev}}$, $D_{\mathrm{val}}$).
4. **Não selecionar modelos com base no resultado do par examinado** —
   nenhuma decisão de calibração deve depender de ver o caso.
5. **Rejeitar candidatos piecewise não monotônicos** (`monotonicity_policy =
   ERROR` em pipeline pericial).
6. **Inspecionar suporte e cobertura por segmento** (`segment_counts_hp/hd`,
   `segment_support_warnings`).
7. **Interpretar Cllr_min e PAVA com cautela** — diagnóstico in-sample, não
   medida operacional (Morrison 2021).
8. **Não confundir discriminação com calibração** (EER e AUC não medem
   calibração).
9. **Documentar clipping, extrapolação e reportabilidade** como decisões
   operacionais explícitas.
10. **Congelar estratégia e parâmetros antes do uso no caso real**.
11. **Registrar alternativas testadas** (menu *a priori* + motivos do
    descarte).
12. **Versionar código, configuração, corpora e resultados** de modo que a
    execução seja reproduzível anos depois.

---

## 13. Perguntas frequentes

**Por que calibrar scores?**
Porque o score bruto não tem interpretação direta como força de evidência;
sua magnitude depende de escala, normalização e backend. A calibração
transforma o score numa medida graduada, comparável e auditável (o log-LR).

**Score alto significa necessariamente LR alta?**
Não automaticamente. Score e LR concordam em ordenação se a calibração for
monotônica, mas a **magnitude** do log-LR depende da curva ajustada e do
suporte amostral.

**Calibração é classificação?**
Não. Calibração produz um número contínuo (log-LR); classificação produz
uma decisão binária. A dicotomização precoce introduz cliff-edge effects.

**Boa EER garante boa calibração?**
Não. EER mede discriminação; é invariante sob qualquer transformação
monotônica crescente do score. Dois sistemas com a mesma EER podem ter
calibrações diametralmente opostas.

**Por que a regressão logística linear é o baseline?**
Pelo conjunto: poucos parâmetros, interpretação simples, estabilidade
amostral, integração com regularização L2, ampla literatura forense e
menor risco de sobreajuste quando a relação verdadeira é aproximadamente
afim.

**Quando a piecewise-linear é justificável?**
Quando há evidência *reproduzível em $D_{\mathrm{dev}}$* de mudança
monotônica de inclinação ao longo do domínio, com suporte amostral
suficiente nos segmentos relevantes. Aprofundamento em
`calibracao_piecewise_linear_faceaudit_relatorio_integrado.md` §7.

**Escolher knots conforme o domínio é cherry-picking?**
Não, se a escolha for feita por procedimento previamente definido, em
$D_{\mathrm{dev}}$ independente do caso, sobre um menu *a priori*. Torna-se
*cherry-picking* quando knots são escolhidos depois de observar o
resultado do caso ou depois de explorar o $D_{\mathrm{val}}$.

**Por que não usar sempre mais knots?**
Porque cada knot adiciona um parâmetro e potencial instabilidade. Knots em
regiões pouco povoadas geram inclinações ruidosas; o sobreajuste deteriora
a transferência. A regra é parcimônia (§4.2.10).

**O knot em zero é um threshold?**
Não. Um knot é uma posição em que a curva calibrada *pode* mudar de
inclinação. A saída segue contínua, e a decisão forense permanece sobre o
log-LR contínuo.

**Quando KDE pode ser útil?**
Quando há dados de calibração abundantes e bem distribuídos (inclusive nas
caudas), e a forma do mapeamento `score → log-LR` é claramente não-linear
de um modo que a piecewise não capture. Custo: maior sensibilidade ao
bandwidth e suporte de caudas.

**Bi-gaussianizada substitui os demais métodos?**
Não. É uma alternativa com motivação teórica própria, com vantagens em
domínios compatíveis, mas com sensibilidade a parâmetros (especialmente
`gx_kde_bandwidth_factor`) e a hipóteses sobre a forma das distribuições.
Compará-la sempre com o baseline.

**O calibrador cria informação de tipicidade?**
Não. Se o score representa apenas similaridade, o calibrador devolve uma LR
de similaridade. Tipicidade e raridade exigem modelagem específica (fora do
escopo deste documento).

**Como tratar LRs extremas?**
Como decisão operacional explícita: clipping (em KDE: sempre ativo; em
BiGauss: opcional + clamps internos; em LogReg: sem clip por padrão),
limites de reportabilidade, justificativa documentada. Ver §10.

**Cllr_min pode ser usado isoladamente como medida de calibração?**
Não. É diagnóstico in-sample. Reportá-lo isoladamente em laudo é
metodologicamente frágil (Morrison 2021). Ele auxilia comparações
internas, não substitui validação externa.

**Quais conjuntos podem compartilhar identidades?**
**Nenhum** dos conjuntos funcionais ($D_{\mathrm{emb}}$, $D_{\mathrm{backend}}$,
$D_{\mathrm{cal}}$, $D_{\mathrm{dev}}$, $D_{\mathrm{val}}$, caso real) deve
compartilhar identidades com nenhum outro. Sobreposições mais sutis (mesma
imagem, crops derivados, frames vizinhos, sessão) também devem ser verificadas
e documentadas.

**O que deve ser congelado antes do caso real?**
Tudo: a família escolhida, todos os parâmetros, os enums, o resultado da
seleção, o splitter, os limites de reportabilidade, a versão do código. Após
o congelamento, $D_{\mathrm{val}}$ é avaliado **uma vez**, e o sistema é
aplicado ao caso sem retuning.

---

## 14. Glossário

- **score** — valor contínuo produzido pelo backend de similaridade.
- **LR** — *likelihood ratio* baseada no score, $p(s\mid H_p)/p(s\mid H_d)$.
- **log-LR** — logaritmo natural da LR, em nats.
- **nat** — unidade do logaritmo natural.
- **bit** — unidade do logaritmo de base 2 (usada em Cllr).
- **calibração** — transformação que torna o número emitido coerente como
  força de evidência.
- **discriminação** — capacidade de ordenar genuínos acima de impostores.
- **EER** — *equal error rate*.
- **AUC** — *area under the ROC curve*.
- **Cllr** — *cost of log-likelihood-ratio* (Brümmer & du Preez, 2006).
- **Cllr_min** — Cllr após recalibração monotônica PAVA in-sample.
- **cal_loss** — $C_{\mathrm{llr}}-C_{\mathrm{llr,min}}$.
- **Cllr_affine** — Cllr após recalibração afim $a\cdot\ell + b$ ótima.
- **ECE** — *empirical cross-entropy*.
- **PAVA** — *pool adjacent violators algorithm* (regressão isotônica).
- **KDE** — *kernel density estimation*.
- **bandwidth** — largura do kernel no KDE.
- **knot** — dobra na curva piecewise-linear.
- **hinge** — função $\max(s-k, 0)$.
- **slope** — inclinação de um segmento da curva piecewise.
- **âncora ($s_0$)** — ponto de referência geométrica em torno do qual a
  expansão piecewise é centrada.
- **clipping** — restrição do log-LR a um intervalo $[L_{\min}, L_{\max}]$.
- **extrapolação** — comportamento do calibrador fora da faixa observada
  em $D_{\mathrm{cal}}$.
- **reportabilidade** — faixa de log-LR em que o sistema entrega valor sem
  ressalva.
- **holdout** — conjunto reservado para avaliação independente.
- **nested cross-validation** — CV com *outer loop* de avaliação e *inner
  loop* de seleção de hiperparâmetros.
- **leakage** — vazamento de informação entre conjuntos (identidades
  comuns, imagens duplicadas, crops, etc.).
- **tipicidade** — quão comum (ou rara) é uma face em relação à população
  relevante.
- **raridade** — complemento operacional da tipicidade.

---

## 15. Referências

- Brümmer, N., & du Preez, J. (2006). *Application-independent evaluation
  of speaker detection.* Computer Speech & Language.
- Morrison, G. S. (2013/2018). *Tutorial on logistic-regression calibration
  and fusion of speaker recognition scores* (e atualizações).
- Morrison, G. S. (2021). *Are there meaningful metrics of the degree of
  calibration?* Forensic Science International: Synergy.
- Morrison, G. S. (2024). *Bi-Gaussianized calibration of likelihood
  ratios.* Law, Probability and Risk.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
  Statistical Learning*, §5.2 (bases piecewise e splines).
- Referências internas do FaceAudit:
  - `docs/piecewise_linear_calibration.md` — especificação interna.
  - `calibracao_piecewise_linear_faceaudit_relatorio_integrado.md` —
    relatório técnico aprofundado.
  - `experiments/calibration/RELATORIO_comparacao_calibradores.md` —
    benchmark comparativo.

---

## Apêndice A — Fórmulas de referência

**Razão de verossimilhança baseada em score:**

$$
\mathrm{LR}(s)=\frac{p(s\mid H_p)}{p(s\mid H_d)},\qquad
\log\mathrm{LR}(s)=\log p(s\mid H_p)-\log p(s\mid H_d).
$$

**Cllr** (em bits):

$$
C_{\mathrm{llr}}=\frac{1}{2}\left[
  \frac{1}{N_p}\sum_{i\in H_p}\log_2\!\bigl(1+e^{-\ell_i}\bigr)
  +\frac{1}{N_d}\sum_{j\in H_d}\log_2\!\bigl(1+e^{+\ell_j}\bigr)
\right].
$$

**Cllr_min:** $C_{\mathrm{llr}}$ aplicado às saídas da regressão isotônica
(PAVA) ajustada nos próprios $\{\ell_i\}$ in-sample.

**cal_loss:** $\mathrm{cal\_loss}=C_{\mathrm{llr}}-C_{\mathrm{llr,min}}$.

**Cllr_affine (recalibração ótima afim):**

$$
\log\mathrm{LR}_{\mathrm{recal}}=a\cdot\log\mathrm{LR}+b,\qquad
C_{\mathrm{llr,affine}}=C_{\mathrm{llr}}(\{a\ell_i+b\}).
$$

**Regressão logística (linear):**

$$
\log\mathrm{LR}(s)=\beta_0+\beta_1 s.
$$

**Função hinge:** $h_k(s)=\max(s-k,0)$.

**Piecewise-linear (forma geral, $K$ knots, âncora $s_0$):**

$$
z=s-s_0,\qquad
\log\mathrm{LR}(s)=\beta_0+\beta_z z+\sum_{j=1}^{K}\beta_j\max(z-\delta_j,0).
$$

Inclinações por trecho: $m_0=\beta_z$, $m_r=\beta_z+\sum_{j=1}^{r}\beta_j$
para $r\ge 1$.

**KDE:**

$$
\widehat{\log\mathrm{LR}}(s)=\log\hat p(s\mid H_p)-\log\hat p(s\mid H_d),
\quad
\log\mathrm{LR}_{\mathrm{op}}(s)=\mathrm{clip}\!\bigl(\widehat{\log\mathrm{LR}}(s),\,L_{\min},\,L_{\max}\bigr).
$$

**Bi-gaussianizada:**

$$
\log\mathrm{LR}(s)=F^{-1}\!\bigl(G(s)\bigr),\quad
F(z)=\tfrac{1}{2}\Phi\!\Bigl(\tfrac{z-\sigma^2/2}{\sigma}\Bigr)+\tfrac{1}{2}\Phi\!\Bigl(\tfrac{z+\sigma^2/2}{\sigma}\Bigr).
$$

**Estimação de $\sigma$:**

$$
\sigma=-2\,\Phi^{-1}(\mathrm{EER}),
\qquad
\sigma^2=-\frac{1}{c}\ln\!\left(\frac{1}{b}\ln C_{\mathrm{llr}}+1\right),
$$

com $b=17{,}7$, $c=9{,}33\times 10^{-3}$.

---

## Apêndice B — `LogisticRegressionCalibrationModelConfig`

Arquivo: `faceaudit/models/calibration/logistic_regression_calibration_model_config.py`.

| Campo | Tipo | Default | Interpretação | Restrição |
|---|---|---|---|---|
| `C` | `float` | `1.0` | inverso da força de regularização L2 | `> 0`; ignorado se `use_gridsearch=True` |
| `use_gridsearch` | `bool` | `True` | ativa GridSearch de `C` | — |
| `c_grid` | `List[float]` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | grade de `C` testada | não-vazia, todos `> 0` quando GridSearch ativo |
| `cv_folds` | `int` | `5` | número de folds do GridSearch | `>= 2`; reduzido em runtime se a classe minoritária for pequena |
| `gridsearch_cv_mode` | `LogRegGridSearchCVMode` | `AUTO` | particionamento dos folds do GridSearch | ver Apêndice F |
| `scoring` | `LogRegGridScoringMetric` | `NEG_CLLR_BALANCED` | métrica do GridSearch | ver Apêndice F |
| `standardize_features` | `bool` | `False` | aplica z-score por feature em entrada multivariada | só age se `D > 1` |
| `k_folds` | `int` | `0` | CV diagnóstica opcional | `0` desliga; `1` é inválido; `>= 2` ativa |
| `cv_split_mode` | `CalibratorCVSplitMode` | `STRATIFIED_PAIR` | splitter da CV diagnóstica | distinta de `gridsearch_cv_mode` |
| `piecewise_linear_config` | `PiecewiseLinearCalibrationConfig` | `mode=NONE` | expansão piecewise interna | desativada por padrão |

Observações:

- A implementação usa `LogisticRegression(solver="lbfgs",
  class_weight="balanced", max_iter=1000)`. A condição `class_weight ==
  "balanced"` é checada explicitamente; qualquer outro valor levanta
  `RuntimeError`.
- O log-LR de saída é o `decision_function` em nats, **sem clipping**.

---

## Apêndice C — `PiecewiseLinearCalibrationConfig`

Arquivo: `faceaudit/models/calibration/piecewise_linear_calibration_config.py`.

| Campo | Tipo | Default | Interpretação | Restrição / observação |
|---|---|---|---|---|
| `mode` | `CalibrationBasisMode` | `NONE` | `NONE` desativa; `PIECEWISE_LINEAR` ativa | gate da expansão |
| `hinge_anchor_mode` | `HingeAnchorMode` | `AUTO_LOGREG` | como resolver $s_0$ | `MANUAL` exige `hinge_anchor_manual_value` |
| `hinge_anchor_manual_value` | `Optional[float]` | `None` | valor manual de $s_0$ | obrigatório se `MANUAL`; finito |
| `knot_offsets` | `List[float]` | `[0.0]` | offsets dos knots | ordenados, sem duplicatas, finitos |
| `knot_scale` | `KnotScale` | `AUX_LOG_LR` (construtor) | escala dos offsets | **`from_dict` cai em `RAW`** quando o campo está ausente em pickles antigos (divergência intencional) |
| `knot_placement` | `KnotPlacement` | `MANUAL` | posicionamento dos knots | `AUTO_FROM_MIRRORED_MEDIANS` é experimental e exige `AUTO_LOGREG` |
| `require_monotonicity` | `bool` | `True` | verifica monotonicidade pós-ajuste | — |
| `monotonicity_policy` | `MonotonicityPolicy` | `ERROR` | ação ao detectar violação | `WARNING` é experimental |
| `min_allowed_slope` | `float` | `1e-8` | inclinação mínima admissível por segmento | `>= 0` |

Restrições cruzadas:

- `KnotScale.AUX_LOG_LR` requer `HingeAnchorMode.AUTO_LOGREG`.
- `KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS` requer `HingeAnchorMode.AUTO_LOGREG`.
- `MANUAL` placement + `knot_offsets` vazio é erro.
- Em `AUTO_FROM_MIRRORED_MEDIANS` com `MANUAL` anchor: erro.

Serialização legada:

- O valor serializado de `KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS` é a
  string `"AUTO_FROM_MEDIANS"` (preservada para compatibilidade de pickles).

---

## Apêndice D — `KDECalibrationModelConfig`

Arquivo: `faceaudit/models/calibration/kde_calibration_model_config.py`.

| Campo | Tipo | Default | Interpretação | Restrição |
|---|---|---|---|---|
| `kernel` | `str` | `"gaussian"` | kernel do `KernelDensity` | não-vazio |
| `use_gridsearch` | `bool` | `True` | otimiza bandwidths por GridSearch no espaço z-score | — |
| `bandwidth_genuine` | `float` | `0.1` | bandwidth de genuínos (usado se GridSearch desligado) | `> 0` |
| `bandwidth_impostor` | `float` | `0.1` | bandwidth de impostores | `> 0` |
| `max_samples_for_gridsearch` | `int` | `10000` | cap de amostras no GridSearch | `> 0` |
| `max_stored_samples` | `int` | `50000` | cap de amostras por classe (downsampling do final fit) | `> 0` |
| `min_bandwidth_abs` | `float` | `1e-3` | bandwidth mínimo absoluto no espaço transformado | `> 0` |
| `max_grid_expansions` | `int` | `2` | expansões da grade se o melhor cair na borda | `>= 0` |
| `min_log_lr` | `float` | `-20.0` | clip inferior (**sempre ativo**) | `<` `max_log_lr` |
| `max_log_lr` | `float` | `+20.0` | clip superior (**sempre ativo**) | — |
| `interpolation_points` | `int` | `10000` | pontos da curva interpolada | `>= 100` |
| `random_seed` | `int` | `42` | semente do subamostreio | — |
| `k_folds` | `int` | `0` | CV diagnóstica opcional | `0` desliga; `1` é inválido; `>= 2` ativa |
| `cv_split_mode` | `CalibratorCVSplitMode` | `STRATIFIED_PAIR` | splitter da CV diagnóstica | — |

Observações:

- A normalização interna é **z-score**, com $(\mu,\sigma)$ aprendidos sobre
  o conjunto de treino concatenado. O campo legado `fisher_eps` foi
  descartado e o parser o ignora.
- O clipping de log-LR é aplicado tanto na construção do interpolador (em
  escala log-LR) quanto após o lookup na inferência (em escala LR, com
  $e^{L_{\min}}$ e $e^{L_{\max}}$).

---

## Apêndice E — `BiGaussianizedCalibrationModelConfig`

Arquivo: `faceaudit/models/calibration/bi_gaussianized_calibration_model_config.py`.

Constantes empíricas (em `bi_gaussianized_calibration_model.py`):
`_CLLR_FIT_B = 17.7`, `_CLLR_FIT_C = 9.33e-3`, `_SIGMA_SQ_CAP = 1.0e3`.

| Campo | Tipo | Default | Interpretação | Restrição |
|---|---|---|---|---|
| `target_metric` | `BiGaussTargetMetric` | `CLLR` | métrica que define $\sigma$ | `EER` ignora o calibrador inicial |
| `initial_calibrator` | `BiGaussInitialCalibrator` | `LOGREG` | calibrador inicial para estimar $\sigma$ via CLLR | irrelevante em `EER` |
| `inversion_method` | `BiGaussInversionMethod` | `INTERP1D` | método de inversão de $F$ | `BRENT_ROOT` é preciso, lento |
| `sigma_shrinkage_factor` | `float` | `1.0` | encolhe $\sigma$ derivado | em `(0, 1]` |
| `enable_clipping` | `bool` | `False` | clipping final do log-LR | desligado por padrão |
| `min_log_lr` | `float` | `-20.0` | clip inferior (só se `enable_clipping`) | — |
| `max_log_lr` | `float` | `+20.0` | clip superior (só se `enable_clipping`) | — |
| `interpolation_points` | `int` | `10000` | grid points de $F^{-1}$ (se `INTERP1D`) | `>= 100` |
| `brent_xtol` | `float` | `1e-6` | tolerância de Brent (se `BRENT_ROOT`) | `> 0` |
| `brent_search_range` | `float` | `50.0` | bracket simétrico para Brent (em nats) | `> 0` |
| `initial_logreg_C` | `float` | `1.0` | `C` da LogReg inicial | `> 0` |
| `initial_kde_bandwidth` | `float` | `0.0` | bandwidth do KDE inicial (`0` = Silverman) | `>= 0` |
| `initial_kde_max_samples_per_class` | `int` | `10000` | cap de amostras do KDE inicial | `>= 0`; `0` = sem subamostreio |
| `random_seed` | `int` | `42` | semente | — |
| `gx_method` | `BiGaussGxMethod` | `KDE_SMOOTHED` | construção de $G(x)$ | **alavanca de maior impacto** |
| `gx_kde_bandwidth_gen` | `float` | `0.0` | bandwidth do KDE de genuínos em $G$ (`0` = Silverman) | `>= 0` |
| `gx_kde_bandwidth_imp` | `float` | `0.0` | bandwidth do KDE de impostores em $G$ (`0` = Silverman) | `>= 0` |
| `gx_kde_bandwidth_factor` | `float` | `5.0` | multiplicador da bandwidth resolvida | `> 0`; ponto empiricamente validado no FaceAudit |
| `gx_clamp_to_training_range` | `bool` | `True` | clamp à faixa observada (não-extrapolação) | só se `KDE_SMOOTHED` |
| `gx_grid_points` | `int` | `2000` | grid points para cachear $G$ | `>= 100` |
| `sigma_estimation_mode` | `BiGaussSigmaEstimationMode` | `CROSS_FITTED` | como estimar $C_{\mathrm{llr}}$ | sem efeito se `initial=NONE` ou métrica `EER` |
| `sigma_cv_folds` | `int` | `5` | $k$ do k-fold no `CROSS_FITTED` | `>= 2`; reduzido em runtime para classes pequenas |
| `k_folds` | `int` | `0` | CV diagnóstica do pipeline completo | distinto de `sigma_cv_folds` |
| `cv_split_mode` | `CalibratorCVSplitMode` | `STRATIFIED_PAIR` | splitter da CV diagnóstica | — |
| `score_direction` | `BiGaussScoreDirection` | `AUTO` | direção do score | `LOWER_IS_GENUINE` inverte o sinal |

Observações:

- **`sigma_cv_folds` vs `k_folds`** — não confundir: o primeiro é interno à
  estimação de $\sigma^2$; o segundo é a CV diagnóstica do pipeline
  completo.
- **Clamps internos**: mesmo com `enable_clipping=False`, $u=G(s)$ é
  truncado em $[\varepsilon, 1-\varepsilon]$, e `gx_clamp_to_training_range`
  (default `True`) projeta scores fora do treino na borda.

---

## Apêndice F — Enums

Defaults consolidados (com notas sobre divergências entre construtor /
parser / fallback de `from_str`).

| Enum | Arquivo | Default | Observação |
|---|---|---|---|
| `CalibrationModelType` | `enums/calibration_model_type.py` | `LOGISTIC_REGRESSION` | — |
| `CalibrationBasisMode` | `enums/calibration_basis_mode.py` | `NONE` | gate da expansão piecewise |
| `HingeAnchorMode` | `enums/hinge_anchor_mode.py` | `AUTO_LOGREG` | — |
| `KnotScale` | `enums/knot_scale.py` | construtor: `AUX_LOG_LR` / parser legado: `RAW` | divergência intencional para compat |
| `KnotPlacement` | `enums/knot_placement.py` | `MANUAL` | `AUTO_FROM_MIRRORED_MEDIANS` é experimental; serializa como `"AUTO_FROM_MEDIANS"` |
| `MonotonicityPolicy` | `enums/monotonicity_policy.py` | `ERROR` | `WARNING` é experimental, não usar em pipeline pericial |
| `LogRegGridSearchCVMode` | `enums/logreg_gridsearch_cv_mode.py` | `AUTO` | alias `STRICT_BY_IDENTITY` → `STRICT_IDENTITY` |
| `LogRegGridScoringMetric` | `enums/logreg_grid_scoring_metric.py` | `NEG_CLLR_BALANCED` | recomendado para forense |
| `CalibratorCVSplitMode` | `enums/calibrator_cv_split_mode.py` | `STRATIFIED_PAIR` | da CV diagnóstica, distinta da do GridSearch |
| `KDEKernelType` | `enums/kde_kernel_type.py` | `GAUSSIAN` | string-valued |
| `BiGaussTargetMetric` | (config) | construtor: `CLLR` / `from_str` fallback: `CLLR` | — |
| `BiGaussInitialCalibrator` | (config) | construtor: `LOGREG` / `from_str` fallback: `NONE` | divergência: o default operacional é `LOGREG` |
| `BiGaussGxMethod` | (config) | construtor: `KDE_SMOOTHED` / `from_str` fallback: `EMPIRICAL_LINEAR` | divergência: o default operacional é `KDE_SMOOTHED` |
| `BiGaussInversionMethod` | (config) | `INTERP1D` | — |
| `BiGaussSigmaEstimationMode` | (config) | construtor: `CROSS_FITTED` / `from_str` fallback: `IN_SAMPLE` | divergência: o default operacional é `CROSS_FITTED` |
| `BiGaussScoreDirection` | (config) | `AUTO` | — |

Restrições cruzadas a destacar:

- `gridsearch_cv_mode` **≠** `cv_split_mode` (alvos diferentes: GridSearch de
  $C$ vs CV diagnóstica).
- `sigma_cv_folds` **≠** `k_folds` (interno à estimação de $\sigma^2$ vs CV
  diagnóstica do pipeline).
- `AUX_LOG_LR` exige `HingeAnchorMode.AUTO_LOGREG` (sem auxiliar não há
  `aux_coef`).
- A expansão piecewise depende de `CalibrationBasisMode.PIECEWISE_LINEAR`
  como gate explícito.
- Modos automáticos experimentais (`AUTO_FROM_MIRRORED_MEDIANS`, política
  `WARNING`) precisam ser rotulados como tais.

---

## Apêndice G — Checklist de revisão documental

Antes de circular o documento:

- [ ] Markdown válido (índice com âncoras, tabelas renderizadas, blocos de
      código consistentes);
- [ ] LaTeX (`$$ ... $$` e `$ ... $`) coerente em todas as fórmulas;
- [ ] fórmulas conferidas contra `cllr_math.compute_cllr`,
      `_compute_affine_calibration`, e `bi_gaussianized_calibration_model.py`;
- [ ] todos os defaults conferidos contra os dataclasses, parsers e fábricas
      correspondentes;
- [ ] caminhos do código registrados (Apêndices B–F);
- [ ] commit informado no cabeçalho;
- [ ] referências internas (`docs/`, `experiments/`) válidas;
- [ ] terminologia uniforme (LR, log-LR, Cllr, $H_p$, $H_d$);
- [ ] exemplos numéricos recalculados e consistentes;
- [ ] distinção explícita entre fundamento matemático, implementação,
      política metodológica e estado experimental;
- [ ] nenhuma afirmação excessiva sobre validação (limitar-se ao escopo dos
      dados de cada estudo citado);
- [ ] nenhuma alteração em código de produção.

---

## Apêndice H — Checklist de validação operacional

Para cada calibração que sustentar um laudo:

- [ ] corpora documentados ($D_{\mathrm{emb}}$, $D_{\mathrm{backend}}$,
      $D_{\mathrm{cal}}$, $D_{\mathrm{dev}}$, $D_{\mathrm{val}}$);
- [ ] split por identidade verificado em cada conjunto e entre conjuntos;
- [ ] ausência de leakage confirmada (imagens, crops, frames, sessões,
      duplicações de corpora);
- [ ] menu de candidatos *a priori* registrado;
- [ ] regra de seleção aplicada uniformemente (regra de parcimônia com
      $\varepsilon$ explícito);
- [ ] health checks por segmento avaliados (suporte, slopes, monotonicidade);
- [ ] política de clipping documentada (KDE: ativa por default; BiGauss:
      desligada por default + clamps internos; LogReg: sem clipping);
- [ ] política de reportabilidade documentada (limites, justificativa,
      versão);
- [ ] métricas computadas e versionadas (Cllr, Cllr_min, cal_loss,
      Cllr_affine, ECE, EER, AUC);
- [ ] bootstrap por identidade (ou hierárquico) para intervalos de
      confiança;
- [ ] nested CV documentada quando aplicada;
- [ ] rastreabilidade completa (commit, ambiente, *seed*, hash do
      calibrador persistido);
- [ ] congelamento explícito **antes** da avaliação em $D_{\mathrm{val}}$ e
      antes da aplicação ao caso.
