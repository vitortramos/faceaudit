Crie um novo tipo de instrução completo na nova arquitetura do FaceAudit para: $ARGUMENTS

A nova arquitetura usa instruções declarativas. Cada instrução nova exige exatamente 5 componentes criados em ordem. Siga todos os passos abaixo sem pular nenhum.

## Contexto da nova arquitetura

- Uma `LRStrategyDefinition` contém três listas ordenadas de `StrategyInstructionDefinition`, uma por fase (`preparation_instructions`, `validation_instructions`, `casework_instructions`)
- O `StrategyExecutor` itera as fases em ordem e resolve o executor de cada instrução via `InstructionExecutorRegistry`; antes de executar, injeta um `InstructionExecutionFrame` (fase, instrução) no contexto via `context.with_frame(...)`
- Cada executor lê recursos do `StrategyExecutionContext`, executa lógica, e escreve resultados de volta como novos `StrategyResource`
- O `StrategyResource` é a unidade de troca entre instruções (identificado por `resource_id` string)

## Passos obrigatórios

### 1. Enum — `faceaudit/enums/strategy_instruction_type.py`
Adicione o novo membro ao enum `StrategyInstructionType`, em UPPER_SNAKE_CASE. Cada membro é uma tupla `(value, label, label_pt, description)` — `value` igual ao nome do membro (persistido em JSON), `label` em inglês técnico, `label_pt` exibido na UI e `description` longa em PT-BR (tooltip/ajuda). O `from_str()` é genérico (resolve por nome ou valor), então não precisa de caso novo.

### 2. DTO da instrução — `faceaudit/models/instructions/`
Crie `<nome_snake_case>_instruction.py`. O DTO deve:
- Herdar de `StrategyInstructionDefinition`
- Usar `@dataclass`
- Declarar todos os campos necessários com `field()` e `metadata={"help": "..."}`
- Implementar `__post_init__` que force `self.instruction_type = StrategyInstructionType.NOVO_TIPO` e chame `self._sync_declared_io()`
- Implementar `_sync_declared_io()` que popule `self.input_resource_ids` e `self.output_resource_ids` a partir dos campos específicos (sem duplicatas, use `list(dict.fromkeys(...))`)
- Implementar properties booleanas para os modos/tipos (ex: `uses_x`, `uses_y`) em vez de comparações de enum espalhadas
- Implementar `from_dict()` que hidrate enums e sub-objetos antes de chamar `cls(**processed_data)`
- Implementar `get_resource_field_specs()` retornando `List[ResourceFieldSpec]` — cada spec define um campo de recurso para renderização no config panel da UI:
  ```python
  from faceaudit.models.instructions.strategy_instruction_definition import ResourceFieldSpec
  
  def get_resource_field_specs(self) -> list:
      return [
          ResourceFieldSpec(
              field_name="input_dataset_resource_id",  # nome do campo no DTO
              label="Dataset de entrada",               # label exibido na UI
              accepted_types=(StrategyResourceType.DATASET,),
              is_output=False,                          # True para outputs
          ),
          ResourceFieldSpec(
              field_name="output_resource_id",
              label="Recurso de saída",
              accepted_types=(StrategyResourceType.DATASET,),
              is_output=True,   # campos output criam placeholders automaticamente
          ),
      ]
  ```
  **Importante:** Campos com `is_output=True` são usados pelo método herdado `get_output_resource_placeholders()` para criar `StrategyResource` placeholders na definição da estratégia. O `accepted_types[0]` define o `StrategyResourceType` do placeholder. A UI renderiza inputs como dropdowns filtrados por tipo e outputs como campos de texto livre. Datasets produzidos por instrução usam `StrategyResourceType.DATASET` — o antigo `INTERMEDIATE_DATASET` foi extinto (só existe como shim de leitura no `from_str`). `ResourceFieldSpec(..., triggers_rebuild=True)` reconstrói o formulário quando o recurso escolhido muda (útil quando outros campos dependem dele).
- Se a instrução tiver campos baseados em enum (ex: modo, tipo de modelo), implementar `get_choice_field_specs()` retornando `List[ChoiceFieldSpec]`:
  ```python
  from faceaudit.models.instructions.strategy_instruction_definition import ChoiceFieldSpec
  
  def get_choice_field_specs(self) -> list:
      return [
          ChoiceFieldSpec(
              field_name="mode",
              label="Modo",
              choices=tuple(e.value for e in MeuEnum),
              enum_type=MeuEnum,
          ),
      ]
  ```
  `ChoiceFieldSpec` aceita também `descriptions` (ajuda por opção) e `dynamic_choices` (callable que filtra as opções em runtime a partir dos recursos disponíveis).
- Para campos primitivos editáveis (int, float, str, bool, listas), implementar `get_value_field_specs()` retornando `List[ValueFieldSpec]` (`value_type` ∈ `"int"`, `"float"`, `"str"`, `"bool"`, `"list_str"`, `"list_int"`; `triggers_rebuild=True` quando o valor muda outros specs).
- Opcionalmente, implementar `build_validation_rules()` para retornar regras de validação dinâmicas (intrínsecas, não persistidas, não removíveis pelo usuário) instanciadas com os resource_ids atuais dos campos. Exemplo: `MODEL_FITTED`, compatibilidade de dimensões, overlap entre datasets. As regras devem estender `ValidationRuleBase`. O `StrategyInstructionExecutorBase.validate()` chama este método automaticamente.
- Regras com limiares que o usuário pode ajustar ou apagar vão em `validation_rule_configs` (lista de `ValidationRuleConfig`), populadas por um `_ensure_default_validation_rules()` chamado de `__post_init__`/`_sync_declared_io()`. Ele deve fazer early-return quando a lista já não estiver vazia ou quando `validation_rules_user_managed=True` (o usuário apagou as regras de propósito).

### 3. Dispatch — `faceaudit/models/instructions/strategy_instruction_definition.py`
Adicione um bloco `if instruction_type == StrategyInstructionType.NOVO_TIPO:` no método `from_dict()`, com import lazy dentro do bloco (padrão existente no arquivo).

### 4. Executor — `faceaudit/executors/instructions/`
Crie `<nome_snake_case>_instruction_executor.py`. O executor deve:
- Herdar de `StrategyInstructionExecutorBase`
- Sobrescrever `_validate_instruction_specifics(instruction, context) → ValidationResult` que verifique: tipo correto da instrução, campos obrigatórios preenchidos, recursos existentes no contexto, tipos corretos dos payloads
- Sobrescrever `execute(instruction, context) → InstructionExecutionResult` que: chame `self.validate()` primeiro, faça cast para o DTO concreto, leia recursos do contexto (use `context.ensure_payload(id)` antes de `context.get_resource(id).payload` — a hidratação é preguiçosa e o payload pode estar só no disco), execute a lógica, escreva resultados via `context.set_resource(StrategyResource(...))`, retorne `InstructionExecutionResult.ok(produced_resource_ids=[...], metadata={...})`
- Use `self._log(context, "mensagem")` para logging
- Envolva a lógica principal em try/except e retorne `self._failure_result(str(e))` em caso de erro
- Use `ValidationResult.neutral()` como ponto de partida, `result.add_error()` / `result.add_warning()`, `result.mark_ok()` ao final se válido

### 5. Registro — `faceaudit/factories/strategy_executor_factory.py`
Adicione o novo executor no método `build_instruction_executor_registry()`:
```python
registry.register(
    StrategyInstructionType.NOVO_TIPO,
    NovoTipoInstructionExecutor(),
)
```
Adicione o import correspondente no topo do arquivo.

### 6. UI (Strategy Manager V2) — opcional, mas normalmente necessário
- Ícone em `_INSTRUCTION_ICONS` de `interface/tabs/strategy_manager_v2/strategy_editor/instruction_list_widget.py` (sem ele aparece `▪`).
- Se a instrução só faz sentido numa fase, inclua o tipo em `_TRAINING_ONLY_TYPES`, `_VALIDATION_ONLY_TYPES` ou `_CASEWORK_ONLY_TYPES` no mesmo arquivo (filtram o menu "➕ Adicionar" por aba).
- Se o output tiver gráfico no painel de resultados, mapeie o tipo em `_instruction_plot_kind` (`instruction_results_panel.py`).
- Toda cor em QSS novo passa pelos tokens de tema (`apply_themed_stylesheet` / `current_token_fn`) — ver skill `faceaudit-qt-design`.

## Verificação final
Após criar todos os componentes, confirme:
- [ ] O enum tem o novo valor e `from_str()` o reconhece
- [ ] O DTO tem `__post_init__` forçando o `instruction_type` correto
- [ ] `_sync_declared_io()` popula corretamente `input_resource_ids` e `output_resource_ids`
- [ ] O dispatch em `strategy_instruction_definition.py` está presente
- [ ] O executor valida o tipo da instrução antes de fazer cast
- [ ] O executor está registrado na factory
