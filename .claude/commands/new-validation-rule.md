Crie uma nova regra de validação na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, regras de validação vivem em `faceaudit/validation_rules/` e estendem `ValidationRuleBase` (não implementam `ValidationRuleProtocol` diretamente).

`ValidationRuleBase` fornece:
- Campos obrigatórios do protocolo (`rule_id`, `name`, `description`, `default_severity`) via construtor
- Helper `_get_or_compute_stats(resource_id, context) → Optional[DatasetStats]` — busca `"{resource_id}_stats"` no contexto ou computa via `DatasetStatsService` e armazena em cache para a execução. Use este helper para qualquer regra que precise de `identity_set`, `n_ids`, `embedding_dim` de um dataset.

As regras chegam à execução por três caminhos:
- **Plugável (persistida)** — a instrução carrega `validation_rule_configs: List[ValidationRuleConfig]` (`rule_type: ValidationRuleType` + `params: dict`), e a `ValidationRuleFactory.create(config, instruction_id)` instancia a regra concreta em runtime. É o caminho de regras com limiares que o usuário pode ajustar, apagar ou adicionar pela UI.
- **Dinâmica (intrínseca)** — o DTO da instrução a constrói em `build_validation_rules()` a cada validação; não é persistida nem removível.
- **Global da estratégia** — registrada no `StrategyExecutor` via `StrategyExecutorFactory.build(validation_rules={...})` e referenciada por `rule_id` em `LRStrategyDefinition.validation_rule_ids`; roda uma vez antes de qualquer fase. (O antigo `StrategyInstructionDefinition.validation_rule_ids` foi extinto — é descartado ao ler JSON antigo.)

Em todos os casos a regra é executada com `validate(target, context)`, onde `target` é a estratégia ou a instrução.

## Passo 1 — crie o arquivo da regra

Crie `faceaudit/validation_rules/<nome_snake_case>_rule.py`. A classe deve:

- Herdar de `ValidationRuleBase`
- Receber todos os parâmetros configuráveis (resource IDs, limiares) no `__init__` e passar `rule_id`, `name`, `description`, `default_severity` para `super().__init__()`
- Implementar `validate(self, target: Any, context: StrategyExecutionContextProtocol) → ValidationResult`:
  - Iniciar com `result = ValidationResult.neutral()`
  - Usar `self._get_or_compute_stats(resource_id, context)` para acessar stats de datasets (se aplicável)
  - Usar `result.add_error("[{self.rule_id}] mensagem")` para falhas CRITICAL
  - Usar `result.add_warning("[{self.rule_id}] mensagem")` para avisos não bloqueantes
  - Chamar `result.mark_ok()` se `result.is_valid and not result.has_warnings`
  - Retornar `result`

**A regra deve ser stateless** — sem atributos mutáveis entre chamadas.
**Não lançar exceções** — qualquer erro vira `add_error()`.
**Prefixar todas as mensagens com `[{self.rule_id}]`** para rastreabilidade nos logs.

## Dois templates segundo o tipo de regra

### Template A — Regra de dataset (usa `_get_or_compute_stats`)

```python
# faceaudit/validation_rules/<nome>_rule.py
from __future__ import annotations
from typing import Any, Optional
from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import StrategyExecutionContextProtocol
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class NomeRule(ValidationRuleBase):
    """
    Descrição do que a regra verifica.

    Parâmetros configuráveis:
        dataset_resource_id: recurso EmbeddingDataset ou FaceDataset a verificar
        limiar: valor mínimo aceitável (default: X)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        limiar: int = 100,
        name: str = "Nome da Regra",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or f"Verifica que '{dataset_resource_id}' ...",
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._limiar = limiar

    def validate(self, target: Any, context: StrategyExecutionContextProtocol) -> ValidationResult:
        result = ValidationResult.neutral()

        stats = self._get_or_compute_stats(self._dataset_id, context)
        if stats is None:
            result.add_error(f"[{self.rule_id}] Recurso '{self._dataset_id}' não encontrado no contexto.")
            return result

        if stats.n_ids < self._limiar:
            result.add_error(
                f"[{self.rule_id}] Dataset '{self._dataset_id}' possui {stats.n_ids} identidade(s), "
                f"mínimo exigido: {self._limiar}."
            )
        else:
            result.mark_ok()

        return result
```

### Template B — Regra de outro recurso (acessa contexto diretamente)

```python
# faceaudit/validation_rules/<nome>_rule.py
from __future__ import annotations
from typing import Any, Optional
from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import StrategyExecutionContextProtocol
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class NomeRule(ValidationRuleBase):
    """
    Descrição do que a regra verifica.
    Target esperado: StrategyInstructionDefinition (ou subclasse específica)
    """

    def __init__(
        self,
        rule_id: str,
        resource_id: str,
        name: str = "Nome da Regra",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or "Verifica que ...",
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._resource_id = resource_id

    def validate(self, target: Any, context: StrategyExecutionContextProtocol) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._resource_id):
            result.add_error(f"[{self.rule_id}] Recurso '{self._resource_id}' não encontrado.")
            return result

        payload = context.get_resource(self._resource_id).payload

        # lógica de verificação aqui

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result
```

## Passo 2 — torne a regra plugável (se for configurável por instrução)

1. **Enum** — `faceaudit/enums/validation_rule_type.py`: adicione o membro ao `ValidationRuleType`. Cada membro é uma tupla `(value, label, description)` — `value` igual ao nome do membro, `label` já em PT-BR (exibido na UI), `description` longa. `from_str()` é genérico.
2. **Factory** — `faceaudit/factories/validation_rule_factory.py`: em `ValidationRuleFactory.create()`, adicione um bloco `if rule_type == ValidationRuleType.NOVA:` com import lazy da classe, passando `rule_id=rule_id` (gerado por `_make_rule_id`) e lendo os parâmetros de `params` — obrigatórios via `_req_str(params, "chave", rule_type)`, opcionais via `params.get("chave", default)`.
3. **Defaults no DTO** (opcional) — se a regra deve vir por padrão numa instrução, acrescente o `ValidationRuleConfig` no `_ensure_default_validation_rules()` do DTO (respeitando o early-return quando a lista não está vazia ou `validation_rules_user_managed=True`).
4. **UI** (opcional, para o usuário poder adicioná-la e ver um resumo):
   - `_RULE_PARAM_SCHEMAS` em `interface/tabs/strategy_manager_v2/strategy_editor/add_validation_rule_dialog.py` — declara os campos do formulário (`_RuleParamSpec` com `kind` ∈ `resource`/`resource_multi`/`int`/`float`/`enum_choice`). Só os tipos presentes ali aparecem no diálogo "➕ Adicionar Regra".
   - `_summarize_rule_params` em `instruction_config_panel.py` — sumário HTML dos parâmetros na seção "Regras de Validação".

Regras intrínsecas (que não fazem sentido desligadas) não passam pelos itens 3–4: o DTO as instancia diretamente em `build_validation_rules()`.

## Exemplos de uso

```python
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig
from faceaudit.enums.validation_rule_type import ValidationRuleType

# Plugável numa instrução (persistida no JSON da estratégia):
instruction.validation_rule_configs.append(ValidationRuleConfig(
    rule_type=ValidationRuleType.NOVA,
    params={"dataset_resource_id": "ds_treino", "limiar": 100},
))

# Global da estratégia:
executor = StrategyExecutorFactory.build(validation_rules={
    "minha_regra_id": MinhaRule(rule_id="minha_regra_id", dataset_resource_id="ds_treino"),
})
definition.validation_rule_ids.append("minha_regra_id")
```

## Regras já implementadas (para referência)

Há 24 regras em `faceaudit/validation_rules/` (uma por arquivo `*_rule.py`): 23 com tipo em `ValidationRuleType` e a `StrategyEmbeddingConsistencyRule`, global e sempre ligada por `StrategyExecutorFactory.build_default_validation_rules()`. Alguns exemplos:

| Arquivo | Regra | Verifica |
|---|---|---|
| `dataset_isolation_rule.py` | `DatasetIsolationRule` | Zero overlap entre dois datasets (anti-leakage) |
| `dataset_overlap_required_rule.py` | `DatasetOverlapRequiredRule` | Overlap mínimo obrigatório entre dois datasets |
| `fisher_rule.py` | `FisherRule` | n_identidades > embedding_dim para PLDA |
| `min_calibration_pairs_rule.py` | `MinCalibrationPairsRule` | Mínimo de pares genuínos/impostores para calibração |
| `holdout_leakage_rule.py` | `HoldoutLeakageRule` | Hold-out disjunto de todos os datasets de treino |
| `min_identities_rule.py` | `MinIdentitiesRule` | Mínimo de identidades no dataset |
| `model_fitted_rule.py` | `ModelFittedRule` | Modelo/pipeline treinado (`is_fitted`) — regra intrínseca |
| `conditional_monotonicity_rule.py` | `ConditionalMonotonicityRule` | ∂log_LR/∂s > 0 no calibrador treinado |

## Verificação final
- [ ] Arquivo criado em `faceaudit/validation_rules/`
- [ ] Se plugável: membro em `ValidationRuleType` + bloco em `ValidationRuleFactory.create()` (e, se quiser na UI, entrada em `_RULE_PARAM_SCHEMAS` e `_summarize_rule_params`)
- [ ] Herda de `ValidationRuleBase`, não de `ValidationRuleProtocol`
- [ ] Todos os parâmetros configuráveis são do construtor (sem hard-code de limiares)
- [ ] Regra é stateless
- [ ] Não lança exceções — apenas `add_error()` / `add_warning()`
- [ ] Todas as mensagens prefixadas com `[{self.rule_id}]`
- [ ] `mark_ok()` chamado quando não há erros nem avisos
- [ ] Tipo esperado de `target` documentado no docstring
