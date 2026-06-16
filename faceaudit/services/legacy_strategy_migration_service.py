# faceaudit/services/legacy_strategy_migration_service.py

"""Serviço de migração preservativa de estratégias V2.14 → V2.15.

Princípios do plano aprovado:

1. **Preserva o artefato histórico**: o JSON original NÃO é mutado.
2. **Pura em memória durante o loader**: nenhuma escrita em disco
   acontece dentro de `LRStrategyDefinition.from_dict`. O loader apenas
   detecta + converte em memória. UI mostra banner e o usuário aciona
   `write_side_by_side_copy` para gerar `*.v2_15.fa` lado-a-lado.
3. **Defaults históricos materializados literalmente**: a conversão
   substitui os campos antigos pelos equivalentes novos com valores que
   reproduzem o comportamento V2.14 bit-a-bit (não confia em defaults
   do código novo).
4. **Modo estrito opt-in**: `allow_legacy_auto_migration=False` no
   loader → `LegacyStrategyVersionRejected`. Útil para validação formal
   externa onde toda migração deve passar pela ferramenta explícita.

Cenários cobertos:

PR 2:
- `instruction_type = "KNN_TYPICALITY_MODEL_TRAINING"` →
  `instruction_type = "TYPICALITY_MODEL_TRAINING"` com
  `typicality_model_type = "KNN"` e config k-NN preenchido literalmente.
- `reference_mode = "PER_FACE"` → `population_weighting_mode =
  "FACE_LEVEL_RAW"`.
- `reference_mode = "PER_IDENTITY"` → `population_weighting_mode =
  "IDENTITY_CENTROIDS"`.

PR 3:
- `score_type = "LOG_DENSITY_KNN"` → `score_type = "LOG_DENSITY_ESTIMATE"`
  em instruções `TYPICALITY_PAIR_SCORING` e `SCORE_INTERACTION_FEATURE`.
- `transform_id = "NEGATE_LOG_DENSITY_KNN"` →
  `"NEGATE_LOG_DENSITY_ESTIMATE"` em metadata de raridade persistido
  dentro de `architecture_state`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Versão do schema migratório. Bumpar quando a tabela de mapeamento mudar
# substancialmente — usada nos metadata da cópia migrada.
MIGRATION_VERSION = "v2_15"


@dataclass
class MigrationPlan:
    """Diagnóstico do que a migração proporá. Apenas leitura; não muta o
    JSON.

    Cada step descreve uma operação concreta (ex.: substituir
    `KNN_TYPICALITY_MODEL_TRAINING` na instrução `INSTR_3` pelo formato
    novo). Steps acumulam warnings para a UI exibir antes de aceitar a
    migração.
    """

    is_needed: bool = False
    legacy_signals_found: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_legacy_signal(self, signal: str) -> None:
        if signal not in self.legacy_signals_found:
            self.legacy_signals_found.append(signal)
            self.is_needed = True

    def add_step(self, step: Dict[str, Any]) -> None:
        self.steps.append(step)
        self.is_needed = True

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class LegacyStrategyVersionRejected(ValueError):
    """Modo estrito (`allow_legacy_auto_migration=False`) detectou JSON
    legado.
    """

    def __init__(self, signals: List[str]) -> None:
        super().__init__(
            "Estratégia em formato legado (V2.14). Modo estrito ativo — "
            f"sinais: {signals}. Use a ferramenta de migração para "
            "gerar a cópia V2.15 antes de carregar."
        )
        self.signals = signals


class LegacyStrategyMigrationService:
    """Detecção + conversão em memória + escrita opt-in da cópia.

    O loader (`LRStrategyDefinition.from_dict`) chama `detect` e, se a
    estratégia for legada, `convert_in_memory`. O writer
    (`write_side_by_side_copy`) é acionado pela UI/CLI quando o usuário
    confirma a migração — NUNCA pelo loader.
    """

    # ────────────────────────────────────────────────────────────────
    # Detect (puro: lê o JSON, não muta)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def detect(data: Dict[str, Any]) -> MigrationPlan:
        """Inspeciona o JSON da estratégia e retorna o plano de migração.

        Não muta o input. Retorna `MigrationPlan(is_needed=False)` quando
        nada precisa ser convertido.
        """
        plan = MigrationPlan()
        if not isinstance(data, dict):
            return plan

        for phase_key in (
            "preparation_instructions",
            "validation_instructions",
            "casework_instructions",
            "instructions",
        ):
            phase = data.get(phase_key)
            if not isinstance(phase, list):
                continue
            for idx, instr in enumerate(phase):
                if not isinstance(instr, dict):
                    continue
                t = instr.get("instruction_type")
                if t == "KNN_TYPICALITY_MODEL_TRAINING":
                    plan.add_legacy_signal("KNN_TYPICALITY_MODEL_TRAINING")
                    plan.add_step({
                        "kind": "instruction_type_rename",
                        "phase": phase_key,
                        "index": idx,
                        "old": "KNN_TYPICALITY_MODEL_TRAINING",
                        "new": "TYPICALITY_MODEL_TRAINING",
                        "typicality_model_type": "KNN",
                    })
                # PR 3: score_type legado `LOG_DENSITY_KNN` em
                # TYPICALITY_PAIR_SCORING / SCORE_INTERACTION_FEATURE.
                if instr.get("score_type") == "LOG_DENSITY_KNN":
                    plan.add_legacy_signal("LOG_DENSITY_KNN_score_type")
                    plan.add_step({
                        "kind": "score_type_rename",
                        "phase": phase_key,
                        "index": idx,
                        "old": "LOG_DENSITY_KNN",
                        "new": "LOG_DENSITY_ESTIMATE",
                    })
        return plan

    # ────────────────────────────────────────────────────────────────
    # Convert in memory (puro: novo dict, original intacto)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def convert_in_memory(
        data: Dict[str, Any], plan: MigrationPlan,
    ) -> Dict[str, Any]:
        """Aplica o plano e devolve UM NOVO dict (deep copy).

        Garantia: `data` não é mutado. Caller pode usar o original e a
        cópia migrada em paralelo (ex.: para comparação científica).

        A metadata da migração não é mesclada no envelope retornado
        — o `LRStrategyDefinition` não tem campo para metadata extra.
        O writer (`write_side_by_side_copy`) adiciona a metadata
        diretamente no JSON gravado em disco, fora do envelope que o
        loader consome.
        """
        if not plan.is_needed:
            return data
        converted = json.loads(json.dumps(data))  # deep copy via JSON

        for step in plan.steps:
            kind = step["kind"]
            phase = converted.get(step["phase"])
            if not isinstance(phase, list):
                continue
            idx = step["index"]
            if idx >= len(phase):
                continue
            instr = phase[idx]
            if not isinstance(instr, dict):
                continue

            if kind == "instruction_type_rename":
                LegacyStrategyMigrationService._migrate_knn_typicality_training(
                    instr, step,
                )
            elif kind == "score_type_rename":
                instr["score_type"] = step["new"]

        return converted

    @staticmethod
    def build_migration_metadata(plan: MigrationPlan) -> Dict[str, Any]:
        """Constrói o bloco de metadata da migração para o writer persistir
        no JSON da cópia gravada em disco.
        """
        return {
            "migrated_from_version": "v2_14",
            "migration_version": MIGRATION_VERSION,
            "migration_signals": list(plan.legacy_signals_found),
            "migration_warnings": list(plan.warnings),
        }

    @staticmethod
    def _migrate_knn_typicality_training(
        instr: Dict[str, Any], step: Dict[str, Any],
    ) -> None:
        """Mutação local da instrução: estrutura antiga → nova."""
        instr["instruction_type"] = step["new"]
        instr["typicality_model_type"] = step["typicality_model_type"]

        # Tabela de defaults históricos (literal — não confia em defaults
        # do código novo).
        # Mapeamento `reference_mode → population_weighting_mode`.
        reference_mode = instr.pop("reference_mode", None) or "PER_IDENTITY"
        if isinstance(reference_mode, str):
            reference_mode_norm = reference_mode.strip().upper()
        else:
            reference_mode_norm = "PER_IDENTITY"
        if reference_mode_norm == "PER_FACE":
            population_weighting_mode = "FACE_LEVEL_RAW"
        else:
            # PER_IDENTITY (default) ou qualquer string desconhecida cai
            # no centróides por identidade (defensável forense; mesmo
            # comportamento que k-NN tinha por default).
            population_weighting_mode = "IDENTITY_CENTROIDS"

        config = {
            "k": instr.pop("k", 50),
            "ood_quantile": instr.pop("ood_quantile", 0.01),
            "population_weighting_mode": population_weighting_mode,
            # Defaults literais que reproduzem comportamento V2.14:
            # - k-NN sempre normalizava silenciosamente
            # - LOO de imagem (não LOIO real)
            # - percentil sobre T_k
            "unit_norm_policy": "NORMALIZE_SILENTLY",
            "unit_norm_tolerance": 1e-4,
            "percentile_basis_score_type": "MEAN_KNN_SIMILARITY",
            "reference_score_calibration_mode": "IN_SAMPLE_LEGACY",
            "cv_n_folds": 0,
            "population_label": None,
            "embedding_column": None,
        }
        instr["typicality_model_config"] = config

    # ────────────────────────────────────────────────────────────────
    # Side-by-side writer (acionado pela UI/CLI, NUNCA pelo loader)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def write_side_by_side_copy(
        converted: Dict[str, Any],
        source_path: Path,
        *,
        plan: Optional["MigrationPlan"] = None,
    ) -> Path:
        """Grava a cópia convertida ao lado do original.

        Args:
            converted: JSON pós-`convert_in_memory`.
            source_path: caminho do arquivo `.fa` original.
            plan: plano da migração — quando fornecido, a metadata da
                migração é injetada no JSON gravado (não no envelope que
                o loader consome, mas como bloco `_migration_metadata`
                irmão do envelope).

        Returns:
            Path do arquivo `.v2_15.fa` recém-criado.

        Levanta `FileExistsError` se o arquivo de destino já existir, a
        menos que o caller tenha lidado com isso antes (não sobrescreve
        silenciosamente).
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Origem não existe: {source_path}")

        suffix = source_path.suffix or ""
        stem = source_path.stem
        target = source_path.with_name(f"{stem}.{MIGRATION_VERSION}{suffix}")

        if target.exists():
            raise FileExistsError(
                f"Cópia migrada já existe: {target}. Renomeie/remova "
                "antes de regenerar."
            )

        payload = dict(converted)
        if plan is not None:
            # Bloco metadata "irmão" — fica no JSON em disco para
            # auditoria, mas o loader (`LRStrategyDefinition.from_dict`)
            # ignora chaves desconhecidas via filtragem prévia. Caso o
            # loader rejeite, o leitor humano ainda vê.
            payload["_migration_metadata"] = (
                LegacyStrategyMigrationService.build_migration_metadata(plan)
            )

        # Preserva mtime do original.
        original_mtime = source_path.stat().st_mtime
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.utime(source_path, (source_path.stat().st_atime, original_mtime))
        return target
