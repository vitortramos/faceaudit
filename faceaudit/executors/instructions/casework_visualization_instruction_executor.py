# faceaudit/executors/instructions/casework_visualization_instruction_executor.py

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.casework_visualization_result import CaseworkVisualizationResult
from faceaudit.models.instructions.casework_visualization_instruction import (
    CaseworkVisualizationInstruction,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class CaseworkVisualizationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução CASEWORK_VISUALIZATION.

    Recebe um dataset de scores de referência (validação, com gen+imp) e
    um dataset de scores do caso (casework, sem ground truth). Extrai os
    arrays para visualização posterior, computa o threshold de EER quando
    em modo RAW, e empacota tudo em um CaseworkVisualizationResult
    persistido como pickle.

    A visualização gráfica em si fica a cargo do
    CaseworkVisualizationViewerDialog na UI — este executor apenas
    prepara os dados.
    """

    # Threshold abaixo do qual o dialog usa linhas verticais tracejadas em
    # vez de KDE para a distribuição do caso. Documentado aqui para que o
    # validador possa logar uma nota informativa sobre qual modo o usuário
    # vai ver.
    KDE_MIN_CASE_SAMPLES = 30

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, CaseworkVisualizationInstruction):
            return self._failure_result(
                "O executor CaseworkVisualizationInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando visualização forense de caso: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            ref_resource = context.get_resource(instruction.reference_scores_resource_id)
            case_resource = context.get_resource(instruction.case_scores_resource_id)

            ref_dataset: SimilarityScoreDataset = ref_resource.payload
            case_dataset: SimilarityScoreDataset = case_resource.payload

            # Extrai arrays conforme o modo.
            gen_arr, imp_arr = self._extract_reference_arrays(
                ref_dataset, instruction.score_type
            )
            case_arr = self._extract_case_array(
                case_dataset, instruction.score_type
            )

            self._log(
                context,
                f"📐 Modo {instruction.score_type} | "
                f"referência: {len(gen_arr)} gen / {len(imp_arr)} imp | "
                f"caso: {len(case_arr)} pares"
            )

            # EER threshold só faz sentido em modo RAW (em CALIBRATED a
            # referência é log-LR=0).
            eer_threshold: Optional[float] = None
            if instruction.score_type == "RAW" and gen_arr.size > 0 and imp_arr.size > 0:
                eer_threshold = self._compute_eer_threshold(gen_arr, imp_arr)
                self._log(
                    context,
                    f"  ↳ EER threshold computado: {eer_threshold:.6f}"
                )

            result = CaseworkVisualizationResult(
                score_type=instruction.score_type,
                genuine_scores=gen_arr,
                impostor_scores=imp_arr,
                case_scores=case_arr,
                reference_dataset_name=str(ref_dataset.name or ""),
                case_dataset_name=str(case_dataset.name or ""),
                eer_threshold=eer_threshold,
            )

            output_resource_id = instruction.output_visualization_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.CASEWORK_VISUALIZATION,
                payload=result,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            # Nota informativa sobre o modo de renderização que o dialog
            # vai usar para a distribuição do caso.
            if result.n_case >= self.KDE_MIN_CASE_SAMPLES:
                render_mode = f"KDE (≥{self.KDE_MIN_CASE_SAMPLES} pares)"
            else:
                render_mode = "linhas verticais tracejadas"

            self._log(
                context,
                f"✅ Visualização registrada como recurso '{output_resource_id}'. "
                f"Caso será renderizado como {render_mode}."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "score_type": instruction.score_type,
                    "reference_scores_resource_id":
                        instruction.reference_scores_resource_id,
                    "case_scores_resource_id":
                        instruction.case_scores_resource_id,
                    "output_visualization_resource_id": output_resource_id,
                    "n_genuine": result.n_genuine,
                    "n_impostor": result.n_impostor,
                    "n_case": result.n_case,
                    "eer_threshold": eer_threshold,
                    "case_min": result.case_summary.get("min"),
                    "case_max": result.case_summary.get("max"),
                    "case_mean": result.case_summary.get("mean"),
                },
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro ao executar visualização de caso "
                f"{instruction.instruction_id}: {exc}"
            )
            return self._failure_result(str(exc))

    # ─── Helpers de extração ─────────────────────────────────────────────

    def _extract_reference_arrays(
        self, ds: SimilarityScoreDataset, score_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrai (gen_arr, imp_arr) do dataset de referência.

        - CALIBRATED: lê `sample.metadata['log_lr']`.
        - RAW: lê `sample.raw_score`.
        Separa por `same_person` (True=gen, False=imp). Samples sem ground
        truth são ignorados (não devem existir em referência por validação).
        """
        gen_vals: List[float] = []
        imp_vals: List[float] = []

        if score_type == "CALIBRATED":
            for s in ds.items:
                md = getattr(s, "metadata", None)
                v = md.get("log_lr") if isinstance(md, dict) else None
                if v is None:
                    continue
                if s.same_person is True:
                    gen_vals.append(float(v))
                elif s.same_person is False:
                    imp_vals.append(float(v))
        else:  # RAW
            for s in ds.items:
                if s.same_person is True:
                    gen_vals.append(float(s.raw_score))
                elif s.same_person is False:
                    imp_vals.append(float(s.raw_score))

        return (
            np.asarray(gen_vals, dtype=np.float64),
            np.asarray(imp_vals, dtype=np.float64),
        )

    def _extract_case_array(
        self, ds: SimilarityScoreDataset, score_type: str,
    ) -> np.ndarray:
        """
        Extrai array de scores do dataset de caso (todos os samples,
        ignorando ground truth — em casework é None mesmo).
        """
        vals: List[float] = []

        if score_type == "CALIBRATED":
            for s in ds.items:
                md = getattr(s, "metadata", None)
                v = md.get("log_lr") if isinstance(md, dict) else None
                if v is not None:
                    vals.append(float(v))
        else:  # RAW
            for s in ds.items:
                vals.append(float(s.raw_score))

        return np.asarray(vals, dtype=np.float64)

    @staticmethod
    def _compute_eer_threshold(gen: np.ndarray, imp: np.ndarray) -> float:
        """
        Computa o threshold de EER (FNR=FPR crossover) via varredura linear
        de 1000 thresholds entre min e max combinados. Suficiente para
        visualização (não precisa de PAVA ou bisseção fina).
        """
        combined = np.concatenate([gen, imp])
        if combined.size == 0:
            return 0.0
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if lo == hi:
            return lo

        thresholds = np.linspace(lo, hi, 1000)
        # FNR(t) = P(gen < t); FPR(t) = P(imp >= t)
        gen_col = gen.reshape(-1, 1)
        imp_col = imp.reshape(-1, 1)
        fnr = (gen_col < thresholds).mean(axis=0)
        fpr = (imp_col >= thresholds).mean(axis=0)
        idx = int(np.argmin(np.abs(fnr - fpr)))
        return float(thresholds[idx])

    # ─── Validador ────────────────────────────────────────────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, CaseworkVisualizationInstruction):
            result.add_error(
                "A instrução informada não é do tipo CaseworkVisualizationInstruction."
            )
            return result

        # ─── score_type ────────────────────────────────────────────────────
        if instruction.score_type not in ("CALIBRATED", "RAW"):
            result.add_error(
                f"score_type inválido: '{instruction.score_type}'. "
                "Esperado 'CALIBRATED' ou 'RAW'."
            )

        # ─── Inputs obrigatórios ──────────────────────────────────────────
        if not instruction.reference_scores_resource_id:
            result.add_error(
                "A instrução deve definir 'reference_scores_resource_id'."
            )
        if not instruction.case_scores_resource_id:
            result.add_error(
                "A instrução deve definir 'case_scores_resource_id'."
            )
        if not instruction.output_visualization_resource_id:
            result.add_error(
                "A instrução deve definir 'output_visualization_resource_id'."
            )

        # ─── Validação do recurso de referência ───────────────────────────
        ref_id = instruction.reference_scores_resource_id
        if ref_id and context.has_resource(ref_id):
            ref_resource = context.get_resource(ref_id)
            self._validate_payload_present(
                ref_resource, ref_id, role="referência",
                expected_producer="SIMILARITY_SCORING (modo validação)",
                result=result,
            )
            if (
                ref_resource is not None
                and isinstance(ref_resource.payload, SimilarityScoreDataset)
            ):
                ds: SimilarityScoreDataset = ref_resource.payload
                # Referência NÃO pode ser casework.
                if ds.metadata.get("is_casework"):
                    result.add_error(
                        f"O recurso de referência '{ref_id}' está marcado "
                        "como is_casework=True (sem ground truth). "
                        "Visualização de caso exige um dataset de referência "
                        "com gen+imp para desenhar as distribuições — não "
                        "pode usar outro dataset de caso como referência."
                    )
                else:
                    # Validação extra do modo CALIBRATED: log_lr precisa estar
                    # populado em ao menos um sample.
                    if instruction.score_type == "CALIBRATED":
                        has_log_lr = any(
                            isinstance(getattr(s, "metadata", None), dict)
                            and s.metadata.get("log_lr") is not None
                            for s in ds.items
                        )
                        if not has_log_lr:
                            result.add_error(
                                f"O recurso de referência '{ref_id}' não "
                                "contém scores calibrados (sem 'log_lr' em "
                                "metadata dos samples). Em modo CALIBRATED, "
                                "use a saída de SCORE_CALIBRATION."
                            )
        elif ref_id and not context.has_resource(ref_id):
            result.add_error(
                f"O recurso de referência '{ref_id}' não foi encontrado "
                "no contexto."
            )

        # ─── Validação do recurso de caso ─────────────────────────────────
        case_id = instruction.case_scores_resource_id
        if case_id and context.has_resource(case_id):
            case_resource = context.get_resource(case_id)
            self._validate_payload_present(
                case_resource, case_id, role="caso",
                expected_producer="SIMILARITY_SCORING (is_casework=True)",
                result=result,
            )
            if (
                case_resource is not None
                and isinstance(case_resource.payload, SimilarityScoreDataset)
            ):
                ds_case: SimilarityScoreDataset = case_resource.payload
                # Caso DEVE ser casework.
                if not ds_case.metadata.get("is_casework"):
                    result.add_error(
                        f"O recurso de caso '{case_id}' não está marcado "
                        "como is_casework=True. O dataset do caso precisa "
                        "ter sido produzido por SIMILARITY_SCORING com "
                        "is_casework=True para distinguí-lo de um conjunto "
                        "de validação."
                    )
                else:
                    if instruction.score_type == "CALIBRATED":
                        has_log_lr = any(
                            isinstance(getattr(s, "metadata", None), dict)
                            and s.metadata.get("log_lr") is not None
                            for s in ds_case.items
                        )
                        if not has_log_lr:
                            result.add_error(
                                f"O recurso de caso '{case_id}' não contém "
                                "scores calibrados. Em modo CALIBRATED, "
                                "passe pelo SCORE_CALIBRATION antes."
                            )
        elif case_id and not context.has_resource(case_id):
            result.add_error(
                f"O recurso de caso '{case_id}' não foi encontrado no contexto."
            )

        # ─── Sanity output != inputs ──────────────────────────────────────
        if instruction.output_visualization_resource_id:
            if instruction.output_visualization_resource_id in (
                instruction.reference_scores_resource_id,
                instruction.case_scores_resource_id,
            ):
                result.add_warning(
                    "output_visualization_resource_id coincide com um "
                    "recurso de entrada. O contexto irá sobrescrever esse "
                    "recurso."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    @staticmethod
    def _validate_payload_present(
        resource,
        rid: str,
        role: str,
        expected_producer: str,
        result: ValidationResult,
    ) -> None:
        """
        Padrão de mensagens-padrão para payload ausente
        (não produzido / falha de load do disco / tipo errado).
        """
        if resource is None:
            result.add_error(
                f"O recurso de {role} '{rid}' está registrado mas é nulo "
                "no contexto."
            )
            return

        if resource.payload is None:
            ref_path = (
                resource.artifact_ref.path
                if resource.artifact_ref is not None else None
            )
            if ref_path:
                result.add_error(
                    f"O recurso de {role} '{rid}' tem artifact_ref "
                    f"('{ref_path}') mas o payload está vazio — falha ao "
                    f"carregar do disco. Re-execute {expected_producer}."
                )
            else:
                result.add_error(
                    f"O recurso de {role} '{rid}' ainda não foi produzido. "
                    f"Execute {expected_producer} antes."
                )
            return

        if not isinstance(resource.payload, SimilarityScoreDataset):
            result.add_error(
                f"O recurso de {role} '{rid}' contém um payload do tipo "
                f"{type(resource.payload).__name__}, esperado "
                "SimilarityScoreDataset."
            )
