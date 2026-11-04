Crie um novo modelo de similaridade completo na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, modelos de similaridade implementam `SimilarityModelProtocol` e produzem **scores brutos** — não LRs calibrados. A calibração é responsabilidade do `CalibrationModelProtocol`. Os dois se conectam via `SimilarityScoreDataset`.

O `SimilarityModelTrainingInstructionExecutor` instancia e treina o modelo baseado no `SimilarityModelType` declarado na instrução. Ao adicionar um novo modelo, é necessário atualizar a instrução, o executor e (se houver config) o editor inline da UI. Os modelos existentes são `CosineSimilarityModel` e `PLDASimilarityModel` (variantes SIMPLE / TWO_COVARIANCE / SHRINKAGE_TWO_COVARIANCE), em `faceaudit/models/similarity/`.

## Passos obrigatórios

### 1. Enum — `faceaudit/enums/similarity_model_type.py`
Adicione o novo membro ao enum `SimilarityModelType` (hoje `COSINE` e `PLDA`). Cada membro é uma tupla `(value, label, label_pt, description)` — `value` igual ao nome do membro, `label` em inglês técnico, `label_pt` exibido na UI, `description` em PT-BR. O `from_str()` é genérico (resolve por nome ou valor), então não precisa de caso novo.

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

### 4. Instrução — `faceaudit/models/instructions/similarity_model_training_instruction.py`
Os hiperparâmetros ficam **embutidos no DTO da instrução**. Hoje o campo `similarity_model_config` é tipado como `Optional[PLDASimilarityModelConfig]` e só é usado pelo PLDA; para um modelo novo com config, acrescente um campo próprio (ex.: `novo_config: Optional[NovoSimilarityModelConfig]`) em vez de reaproveitar aquele. Em `SimilarityModelTrainingInstruction`:
- Property `uses_novo_modelo` (ao lado de `uses_cosine` / `uses_plda`)
- `__post_init__` cria o config default quando `uses_novo_modelo` e o campo estiver vazio; `prune_unused_fields()` zera o campo nos outros tipos; `from_dict()` hidrata o dict do config
- `get_resource_field_specs()` / `_sync_declared_io()`: inclua o dataset de treino se o modelo exigir fit; `_ensure_default_validation_rules()` se houver regras padrão (o PLDA usa `FISHER` + `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES`)

### 5. Integração no executor — `faceaudit/executors/instructions/similarity_model_training_instruction_executor.py`
Adicione:
- Import do novo modelo e config
- Bloco `elif instruction.uses_novo_modelo:` no `execute()` (ao lado de `_prepare_cosine_similarity_model` / `_train_plda_similarity_model`)
- Validação correspondente em `_validate_instruction_specifics()` (config com `isinstance`, dataset presente no contexto)
- Método `_prepare_novo_modelo(dataset, config_payload, context)` (ou `_train_novo_modelo`) seguindo o padrão dos existentes

### 6. UI — editor inline do config (se houver)
- Acrescente `("novo_config", "inline")` a `NESTED_DTO_ATTRS` em `interface/tabs/strategy_manager_v2/strategy_editor/nested_dto_widget.py`.
- No laço sobre `NESTED_DTO_ATTRS` em `instruction_config_panel.py`, acrescente o gate `if attr_name == "novo_config" and not getattr(instr, "uses_novo_modelo", False): continue`.
- Campos que aparecem/somem conforme outro campo: `get_hidden_fields()` + `REACTIVE_FIELDS` no DTO (padrão do `PLDASimilarityModelConfig`).

## Verificação final
- [ ] Enum tem o novo membro `(value, label, label_pt, description)`
- [ ] Instrução tem a property `uses_novo_modelo` e, se houver config, o campo próprio + gate no config panel
- [ ] Config DTO tem validação em `__post_init__` e `from_dict()` correto
- [ ] Modelo implementa todos os 7 membros do protocolo
- [ ] `save()`/`load()` são simétricos e incluem `model_version`
- [ ] Executor instancia e treina o modelo, registrando-o no contexto como `TRAINED_SIMILARITY_MODEL`
- [ ] Validação no executor verifica tipo correto do config payload
