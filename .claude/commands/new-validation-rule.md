Crie uma nova regra de validação na nova arquitetura do FaceAudit para: $ARGUMENTS

## Contexto

Na nova arquitetura, regras de validação vivem em `faceaudit/validation_rules/` e estendem `ValidationRuleBase` (não implementam `ValidationRuleProtocol` diretamente).

`ValidationRuleBase` fornece:
- Campos obrigatórios do protocolo (`rule_id`, `name`, `description`, `default_severity`) via construtor
- Helper `_get_or_compute_stats(resource_id, context) → Optional[DatasetStats]` — busca `"{resource_id}_stats"` no contexto ou computa via `DatasetStatsService` e armazena em cache para a execução. Use este helper para qualquer regra que precise de `identity_set`, `n_ids`, `embedding_dim` de um dataset.

As regras são:
- Registradas no `StrategyExecutor` via `StrategyExecutorFactory.build(validation_rules={...})`
- Referenciadas por `rule_id` na `LRStrategyDefinition.validation_rule_ids` (validação global, antes de qualquer instrução) ou em `StrategyInstructionDefinition.validation_rule_ids` (validação antes de uma instrução específica)
- Executadas com `validate(target, context)` onde `target` é a estratégia ou instrução que declarou a regra

## Passo único — crie o arquivo da regra

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

## Como registrar e usar a nova regra

```python
# Na montagem do executor (ex: em um serviço ou script de uso):
executor = StrategyExecutorFactory.build(validation_rules={
    "minha_regra_id": MinhaRule(rule_id="minha_regra_id", dataset_resource_id="ds_treino"),
})

# Na LRStrategyDefinition (validação global):
definition.validation_rule_ids.append("minha_regra_id")

# Ou em uma instrução específica:
instruction.validation_rule_ids.append("minha_regra_id")
```

## Regras já implementadas (para referência)

| Arquivo | Regra | Verifica |
|---|---|---|
| `dataset_isolation_rule.py` | `DatasetIsolationRule` | Zero overlap entre dois datasets (anti-leakage) |
| `dataset_overlap_required_rule.py` | `DatasetOverlapRequiredRule` | Overlap mínimo obrigatório entre dois datasets |
| `fisher_rule.py` | `FisherRule` | n_identidades > embedding_dim para PLDA |
| `min_calibration_pairs_rule.py` | `MinCalibrationPairsRule` | Mínimo de pares genuínos/impostores para calibração |
| `holdout_leakage_rule.py` | `HoldoutLeakageRule` | Hold-out disjunto de todos os datasets de treino |

## Verificação final
- [ ] Arquivo criado em `faceaudit/validation_rules/`
- [ ] Herda de `ValidationRuleBase`, não de `ValidationRuleProtocol`
- [ ] Todos os parâmetros configuráveis são do construtor (sem hard-code de limiares)
- [ ] Regra é stateless
- [ ] Não lança exceções — apenas `add_error()` / `add_warning()`
- [ ] Todas as mensagens prefixadas com `[{self.rule_id}]`
- [ ] `mark_ok()` chamado quando não há erros nem avisos
- [ ] Tipo esperado de `target` documentado no docstring
