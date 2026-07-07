Crie uma nova transformação de embeddings completa no FaceAudit para: $ARGUMENTS

## Contexto

Na arquitetura do FaceAudit, transformações de embeddings implementam `EmbeddingTransformationProtocol` e são registradas em `TransformationType`. O `EmbeddingTransformationPipeline` (nova arquitetura) as instancia via `_STEP_CLASS_MAP`, usando o DTO de configuração injetado pelo `TransformatorConfig`. A matemática é sempre delegada ao `EmbeddingMathService`.

Cada transformação pode ser **stateful** (aprende parâmetros no `fit()`) ou **stateless** (apenas aplica uma operação fixa). O `get_state()`/`set_state()` é obrigatório mesmo em stateless — o modo DUAL do pipeline depende dele para sincronizar os ramos Source e Target.

## Passos obrigatórios

### 1. Enum de tipo fundamental — `faceaudit/enums/step_type.py`
Adicione um novo valor ao enum `StepType` em UPPER_SNAKE_CASE (ex: `MY_STEP = "MY_STEP"`).

### 2. Config DTO — `faceaudit/models/<nome>_config.py`
**Pule este passo se a transformação for stateless sem parâmetros** (como `L2Normalization` ou `CenteringTransformation`) — nesse caso use `dto_class=None` no enum (passo 3).

Se a transformação tiver parâmetros configuráveis, crie o DTO:
- Usar `@dataclass` e herdar de `JSONSerializableMixin`
- Declarar campos com `field()` e `metadata={"help": "..."}`
- Validar valores em `__post_init__` (rejeite valores inválidos imediatamente)
- Implementar `from_dict()` hidratando enums com `from_str()` se necessário
- Manter o DTO focado **apenas nos parâmetros da transformação**

### 3. Enum rico — `faceaudit/enums/transformation_type.py`
Adicione o novo membro ao enum `TransformationType`. O construtor exige **exatamente 10 argumentos** na ordem:

```python
NOME_TRANSFORMATION = (
    "NOME_TRANSFORMATION",        # value — string de serialização (deve bater com o nome do membro)
    "Nome Legível",               # label — exibido na UI
    "Abrev",                      # summary_name — abreviação para o resumo do pipeline
    "Descrição da operação.",     # description — tooltip/ajuda
    "🔤",                         # icon — emoji para a UI
    False,                        # supports_dimension_reduction — True se reduz dimensão (ex: Whitening, LDA)
    StepType.MY_STEP,             # step_type — valor criado no passo 1
    False,                        # requires_axis_recalculation — True se muda a escala do espaço (ex: LDA)
    lambda p: "",                 # param_formatter — lambda que recebe dict e retorna string curta para o sumário
    NomeConfig                    # dto_class — classe do DTO do passo 2, ou None se stateless sem parâmetros
)
```

Se houver DTO, adicione o import no topo do arquivo (junto dos outros DTOs importados). O `dto_class` é o que alimenta o factory em `TransformatorConfig.from_dict()` — sem ele a desserialização do JSON quebra silenciosamente.

### 4. Implementação — `faceaudit/transformators/<nome>_transformation.py`
Crie a classe implementando `EmbeddingTransformationProtocol`.

**Propriedades obrigatórias:**
- `name: str` — nome amigável para logs e auditoria
- `transformation_type: TransformationType` — retorna o valor criado no passo 3
- `requires_target_dataset: bool` — `True` apenas se a transformação **obrigatoriamente** precisa de um segundo dataset no `fit()` (como CORAL)

**Métodos obrigatórios:**
- `fit(source_dataset, target_dataset=None)` — aprende parâmetros; para stateless apenas registra métricas; defina `self._fitted = True` ao final
- `transform(dataset) → EmbeddingDataset` — aplica a transformação; não modifique o dataset original
- `transform_single(embedding_info) → EmbeddingInfo` — aplica a um único item (geralmente cria dataset temporário e chama `transform`)
- `get_execution_metrics() → Dict` — retorna dict com `step_type` e métricas relevantes; calculado no `fit()`
- `save(path)` — persiste estado via pickle com dict explícito (não pickle da instância inteira); para stateless, salva apenas métricas e config
- `load(path)` — restaura estado; reconstrói o DTO via `from_dict()` se houver config salvo
- `get_state() → Dict` — retorna estado serializável para sincronização no modo DUAL; nunca inclua referências ao `math_service`
- `set_state(state)` — restaura o estado retornado por `get_state()`

**Padrões a seguir:**
- Receba `math_service: EmbeddingMathService` e `config: Optional[NomeConfig]` no `__init__`
- Delegue **todo cálculo numérico** ao `math_service` — não implemente matemática diretamente na classe
- Registre métricas forenses no `fit()` dentro de try/except (falha em métricas não deve abortar o treino)
- Logs no formato `[{self.name}] mensagem`
- Suporte legado no `load()`: se o pickle não tiver `config_dict`, reconstrua o DTO com defaults

### 5. Registro na nova arquitetura — `faceaudit/pipelines/embedding_transformation_pipeline.py`
Adicione a entrada no `_STEP_CLASS_MAP`:
```python
TransformationType.NOME_TRANSFORMATION: NomeTransformation,
```
Adicione o import no topo do arquivo, junto dos outros imports de transformadores.

## Passos condicionais

### 6. Registro na arquitetura antiga — `faceaudit/pipelines/transformation_pipeline.py`
Execute **somente se for necessário manter compatibilidade com a arquitetura antiga**. Adicione a entrada no `_TRANSFORMATOR_MAP` seguindo o mesmo padrão do passo 5.

### 7. Política de treino DUAL — apenas se `requires_target_dataset = True` ou se a transformação puder usar o target opcionalmente no modo DUAL

Se a nova transformação pode se beneficiar do `target_dataset` durante o treino no modo DUAL (como Whitening e LDA fazem opcionalmente, ou como CORAL faz obrigatoriamente):

**7a.** Em `faceaudit/models/pipeline/pipeline_training_policy.py`, adicione o campo:
```python
use_target_for_<nome>: bool = field(
    default=True,  # ou False, dependendo da semântica
    metadata={"help": "..."}
)
```
Atualize também os factory methods `default_linear()` e `default_dual()` com o valor apropriado.

**7b.** Em `faceaudit/pipelines/embedding_transformation_pipeline.py`, no método `_choose_dual_fit_datasets()`, adicione o bloco `elif` correspondente:
```python
elif t_value == TransformationType.NOME_TRANSFORMATION.value:
    if training_policy.use_target_for_<nome>:
        # ajuste fit_source ou fit_target conforme a semântica
```

## Verificação final
- [ ] `StepType` tem o novo valor
- [ ] DTO tem `from_dict()` correto com hidratação de enums (se aplicável)
- [ ] `TransformationType` tem o novo membro com `dto_class` correto (ou `None` se stateless)
- [ ] A classe implementa todos os 8 membros do protocolo
- [ ] `get_state()`/`set_state()` são simétricos e não incluem `math_service`
- [ ] `save()`/`load()` são simétricos; `load()` reconstrói o DTO via `from_dict()`
- [ ] A transformação está registrada em `_STEP_CLASS_MAP`
- [ ] Se usar target no modo DUAL: `PipelineTrainingPolicy` e `_choose_dual_fit_datasets()` foram atualizados
