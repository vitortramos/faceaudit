Crie um novo modelo de calibração completo na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, modelos de calibração implementam `CalibrationModelProtocol`. Eles recebem **scores brutos** (produzidos por um `SimilarityModelProtocol`) e os convertem em LRs calibrados.

Convenção de labels (sempre respeitada):
- `1` = genuíno / mesma identidade / hipótese Hp
- `0` = impostor / identidades diferentes / hipótese Hd

O `CalibrationModelTrainingInstructionExecutor` instancia e treina o calibrador. Aceita como entrada um `SimilarityScoreDataset` ou um `dict {"scores": ..., "labels": ...}`. Ao adicionar um novo calibrador, é necessário atualizar esse executor.

## Passos obrigatórios

### 1. Enum — `faceaudit/enums/calibration_model_type.py`
Adicione o novo valor ao enum `CalibrationModelType`. Verifique se o enum tem `from_str()` e adicione o caso correspondente.

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

### 4. Integração no executor — `faceaudit/executors/instructions/calibration_model_training_instruction_executor.py`
Adicione:
- Import do novo calibrador e config
- Property `uses_novo_calibrador` na instrução (se não existir, adicione também em `CalibrationModelTrainingInstruction`)
- Bloco `elif instruction.calibration_model_type == CalibrationModelType.NOVO_TIPO:` no `execute()`
- Validação do tipo correto do config payload em `_validate_instruction_specifics()`
- Método `_train_novo_calibrador(scores, labels, config_payload, context)` seguindo o padrão dos existentes

## Verificação final
- [ ] Enum tem o novo valor e `from_str()` o reconhece
- [ ] Config DTO tem validação em `__post_init__` e `from_dict()` correto
- [ ] Calibrador implementa todos os 5 membros do protocolo
- [ ] `fit()` valida labels (apenas 0 e 1, ambas as classes presentes)
- [ ] `predict_proba()` retorna shape (N, 2) com [P(Hd), P(Hp)]
- [ ] `calibrate_scores()` retorna log-LR com proteção numérica (clip/eps)
- [ ] `save()`/`load()` são simétricos e incluem `model_version`
- [ ] Executor instancia e treina o calibrador, registrando-o no contexto como `TRAINED_CALIBRATION_MODEL`
