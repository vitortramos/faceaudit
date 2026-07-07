Crie um novo modelo de similaridade completo na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, modelos de similaridade implementam `SimilarityModelProtocol` e produzem **scores brutos** — não LRs calibrados. A calibração é responsabilidade do `CalibrationModelProtocol`. Os dois se conectam via `SimilarityScoreDataset`.

O `SimilarityModelTrainingInstructionExecutor` instancia e treina o modelo baseado no `SimilarityModelType` declarado na instrução. Ao adicionar um novo modelo, é necessário atualizar esse executor também.

## Passos obrigatórios

### 1. Enum — `faceaudit/enums/similarity_model_type.py`
Adicione o novo valor ao enum `SimilarityModelType`. Verifique se o enum tem `from_str()` e adicione o caso correspondente.

### 2. Config DTO — `faceaudit/models/similarity/<nome>_similarity_model_config.py`
Se o modelo tiver hiperparâmetros, crie um DTO de configuração. Deve:
- Usar `@dataclass` e herdar de `JSONSerializableMixin`
- Declarar campos com `field()` e `metadata={"help": "..."}`
- Validar valores em `__post_init__`
- Implementar `from_dict()` hidratando enums se necessário
- Manter configuração focada **apenas no modelo** — sem misturar calibração, pipeline ou decisões de execução

### 3. Implementação do modelo — `faceaudit/models/similarity/<nome>_similarity_model.py`
Implemente `SimilarityModelProtocol`. A classe deve:

**Propriedades obrigatórias:**
- `name: str` — nome amigável
- `is_fitted: bool` — True após fit bem-sucedido
- `requires_training: bool` — True se exige fit real (ex: PLDA), False se é só preparação (ex: Cosine)

**Métodos obrigatórios:**
- `fit(dataset: EmbeddingDataset, logger_callback: Callable[[str], None] = print) → None`
  - Valide dataset (não nulo, não vazio)
  - Logue progresso via `logger_callback`
  - Defina `self._is_fitted = True` ao final
- `score_pair(probe: EmbeddingInfo, reference: EmbeddingInfo) → float` — score bruto entre dois embeddings
- `score_vectors(v1: np.ndarray, v2: np.ndarray) → float` — score bruto entre dois vetores 1D
- `score_matrix(probe_matrix: np.ndarray, ref_matrix: np.ndarray) → np.ndarray` — matriz completa (N_probe × N_ref)
- `save(path: str) → None` — persiste estado via pickle com `model_version` e `model_type` no payload
- `load(path: str) → None` — restaura estado a partir do pickle

**Padrões a seguir:**
- `MODEL_VERSION = 1` como constante de classe
- `_ensure_fitted()` chamado no início de score_pair/score_vectors/score_matrix
- Payload do pickle como dict explícito (não pickle da instância inteira)
- Logs no formato `[{self.name}] mensagem`

### 4. Integração no executor — `faceaudit/executors/instructions/similarity_model_training_instruction_executor.py`
Adicione:
- Import do novo modelo e config
- Property `uses_novo_modelo` na instrução (se ainda não existir, adicione também em `SimilarityModelTrainingInstruction`)
- Bloco `elif instruction.uses_novo_modelo:` no `execute()` chamando um método `_prepare_novo_modelo()`
- Validação correspondente em `_validate_instruction_specifics()`
- Método `_prepare_novo_modelo(dataset, config_payload, context)` seguindo o padrão dos existentes

## Verificação final
- [ ] Enum tem o novo valor e `from_str()` o reconhece
- [ ] Config DTO tem validação em `__post_init__` e `from_dict()` correto
- [ ] Modelo implementa todos os 7 membros do protocolo
- [ ] `save()`/`load()` são simétricos e incluem `model_version`
- [ ] Executor instancia e treina o modelo, registrando-o no contexto como `TRAINED_SIMILARITY_MODEL`
- [ ] Validação no executor verifica tipo correto do config payload
