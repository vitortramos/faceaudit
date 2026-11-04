Crie um novo modelo de calibração completo na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, modelos de calibração implementam `CalibrationModelProtocol`. Eles recebem **scores brutos** (produzidos por um `SimilarityModelProtocol`) e os convertem em LRs calibrados.

Convenção de labels (sempre respeitada):
- `1` = genuíno / mesma identidade / hipótese Hp
- `0` = impostor / identidades diferentes / hipótese Hd

O `CalibrationModelTrainingInstructionExecutor` instancia e treina o calibrador. Aceita como entrada um `SimilarityScoreDataset` (score 1D ou vetorizado) ou um `dict {"scores": ..., "labels": ...}`. Ao adicionar um novo calibrador, é necessário atualizar a instrução, o executor e o editor inline da UI. Os calibradores existentes são `LogisticRegressionCalibrationModel`, `KDECalibrationModel` e `BiGaussianizedCalibrationModel`, em `faceaudit/models/calibration/`.

## Passos obrigatórios

### 1. Enum — `faceaudit/enums/calibration_model_type.py`
Adicione o novo membro ao enum `CalibrationModelType`. Cada membro é uma tupla `(value, label, label_pt, description)` — `value` igual ao nome do membro, `label` em inglês técnico, `label_pt` exibido na UI, `description` em PT-BR. O `value` é definido em `__new__` (não em `__init__`) para o pickle funcionar — mantenha o padrão. O `from_str()` é genérico (resolve por nome ou valor), então não precisa de caso novo.

### 2. Config DTO — `faceaudit/models/calibration/<nome>_calibration_model_config.py`
Crie o DTO de configuração do calibrador. Deve:
- Usar `@dataclass` e herdar de `JSONSerializableMixin`
- Declarar campos com `field()` e `metadata={"help": "..."}`
- Validar valores em `__post_init__` (rejeite valores inválidos imediatamente)
- Implementar `from_dict()` com hidratação de enums se necessário
- Manter configuração focada **apenas no calibrador** — sem misturar modelo de similaridade, pipeline ou mineração de pares

### 3. Implementação do calibrador — `faceaudit/models/calibration/<nome>_calibration_model.py`
Implemente `CalibrationModelProtocol`. A classe deve:

**Propriedades obrigatórias:**
- `name: str` — nome amigável
- `is_fitted: bool` — True após fit bem-sucedido
- `requires_training: bool` — sempre True para calibradores

**Métodos obrigatórios:**
- `fit(scores: np.ndarray, labels: np.ndarray, logger_callback: Callable[[str], None] = print) → None`
  - Valide arrays: mesma quantidade de amostras, labels apenas 0 e 1, ambas as classes presentes
  - Aceite scores como vetor 1D ou matriz (N, 1)
  - Logue progresso via `logger_callback` (N total, N genuínos, N impostores)
  - Defina `self._is_fitted = True` ao final
- `predict_proba(scores: np.ndarray) → np.ndarray`
  - Retorna shape (N, 2): coluna 0 = P(Hd), coluna 1 = P(Hp)
  - Compatível com convenção scikit-learn
- `calibrate_scores(scores: np.ndarray) → np.ndarray`
  - Retorna log-LR calibrado (1D array)
  - Implementação típica: `log(P_hp) - log(P_hd)` com clip numérico
- `save(path: str) → None` — persiste via pickle com `model_version` e `model_type` no payload dict
- `load(path: str) → None` — restaura a partir do pickle

**Padrões a seguir:**
- `MODEL_VERSION = 1` como constante de classe
- `_ensure_fitted()` chamado no início de `predict_proba` e `calibrate_scores`
- Payload do pickle como dict explícito (não pickle da instância inteira)
- Logs no formato `[{self.name}] mensagem`

### 4. Config embutida na instrução — `faceaudit/models/instructions/calibration_model_training_instruction.py`
Os hiperparâmetros do calibrador vivem **embutidos no DTO da instrução** (como `logreg_config`, `kde_config`, `bigauss_config`), não como recurso separado. Em `CalibrationModelTrainingInstruction`:
- Campo `novo_config: Optional[NovoCalibrationModelConfig] = field(default=None, metadata={"help": ...})`
- Property `uses_novo` (`calibration_model_type == CalibrationModelType.NOVO_TIPO`)
- Em `__post_init__`, criar o config default quando `uses_novo` e o campo estiver vazio
- Em `prune_unused_fields()`, zerar o campo quando o tipo não for o novo
- Em `from_dict()`, hidratar o dict do config para o DTO

### 5. Integração no executor — `faceaudit/executors/instructions/calibration_model_training_instruction_executor.py`
Adicione:
- Import do novo calibrador e config
- Bloco `elif instruction.uses_novo:` no `execute()` (ao lado dos ramos `uses_logistic_regression` / KDE / `uses_bigauss`)
- Validação de `isinstance(instruction.novo_config, NovoCalibrationModelConfig)` em `_validate_instruction_specifics()`
- Método `_train_novo_calibrator(scores, labels, config_payload, context)` seguindo o padrão de `_train_kde_calibrator` / `_train_bigauss_calibrator`
- Se o calibrador tiver telemetria a gravar no metadata, um ramo em `_attach_calibrator_telemetry`

Observação: a auditoria de monotonicidade (`_run_monotonicity_audit`) só roda quando há `architecture_state` ou piecewise-linear no metadata — para calibradores novos ela é pulada em silêncio, a menos que se implemente um provider de derivada (`MonotonicityDerivativeProviderProtocol`).

### 6. UI — editor inline do config
- Em `interface/tabs/strategy_manager_v2/strategy_editor/nested_dto_widget.py`, acrescente `("novo_config", "inline")` a `NESTED_DTO_ATTRS`.
- Em `instruction_config_panel.py` (laço sobre `NESTED_DTO_ATTRS`), acrescente o gate `if attr_name == "novo_config" and not getattr(instr, "uses_novo", False): continue`, como os existentes — sem ele o editor aparece para todo tipo de calibrador.
- Os `metadata={"help": "..."}` dos campos do config viram tooltips automaticamente.

## Verificação final
- [ ] Enum tem o novo membro `(value, label, label_pt, description)`
- [ ] Instrução tem o campo `novo_config`, a property `uses_novo` e o gate correspondente no config panel
- [ ] Config DTO tem validação em `__post_init__` e `from_dict()` correto
- [ ] Calibrador implementa todos os 5 membros do protocolo
- [ ] `fit()` valida labels (apenas 0 e 1, ambas as classes presentes)
- [ ] `predict_proba()` retorna shape (N, 2) com [P(Hd), P(Hp)]
- [ ] `calibrate_scores()` retorna log-LR com proteção numérica (clip/eps)
- [ ] `save()`/`load()` são simétricos e incluem `model_version`
- [ ] Executor instancia e treina o calibrador, registrando-o no contexto como `TRAINED_CALIBRATION_MODEL`
