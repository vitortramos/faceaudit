import copy
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

from faceaudit.models import EmbeddingDataset
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.pipelines.transformation_pipeline import TransformationPipeline
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.services.embedding_math_service import EmbeddingMathService

# Protocolo (para type hints / contrato)
from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol


class PLDATransformationPipelinesController:
    """
    Controlador responsável por treinar, sincronizar e persistir os pipelines PLDA.

    Correções aplicadas:
    - C1.1: _find_equivalent_step agora usa (TransformationType + ocorrência) e não "primeiro por type".
    - C1.2: respeita requires_target_dataset do protocolo ao decidir target_dataset no fit.
    - C1.3: valida compatibilidade estrutural (Target deve ser subsequência compatível do Source).
    - C1.4: _copy_weights usa get_state()/set_state() dos steps (estado explícito), sem copiar __dict__ inteiro.
    """

    def __init__(self):
        self.math_service = EmbeddingMathService()

    # ---------------------------------------------------------------------
    # API principal
    # ---------------------------------------------------------------------
    def train_and_persist(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset],
        source_config: TransformationPipelineConfig,
        target_config: TransformationPipelineConfig,
        strategy: PLDATrainingStrategy,
        source_save_path: Union[str, Path],
        target_save_path: Union[str, Path],
        logger_callback=print,
    ) -> None:
        """
        Fluxo:
        1) Instancia pipelines (ocosos) com configs.
        2) Valida compatibilidade estrutural entre pipelines (C1.3).
        3) Treina o pipeline Source passo-a-passo, com data-flow guiado por strategy.
        4) Shadow-copy do estado treinado para o passo equivalente no Target (C1.1/C1.4).
        5) Avança (transform) os datasets correntes.
        6) Salva os artefatos finais.
        """

        # 1) Instanciação dos Pipelines (ocosos)
        pipe_src = TransformationPipeline(source_config, self.math_service)
        pipe_tgt = TransformationPipeline(target_config, self.math_service)

        # 2) Validação estrutural (C1.3) + preparação de mapeamento (C1.1)
        mapping = self._build_and_validate_mapping(
            pipe_src=pipe_src,
            pipe_tgt=pipe_tgt,
            source_config=source_config,
            target_config=target_config,
        )

        logger_callback(f"[Controller] Iniciando treino Dual. Estratégia: {getattr(strategy, 'label', str(strategy))}")

        # Dados correntes que fluem pelo pipeline
        curr_src_data = source_dataset
        curr_tgt_data = target_dataset

        # 3) Loop de Treinamento (guiado pelo Source)
        for i, src_step in enumerate(pipe_src.steps):
            step_type = self._get_step_type_from_source_config(source_config, i, src_step)

            step_name = getattr(src_step, "name", None) or type(src_step).__name__
            
            # SANEAMENTO PARA O LOG: Evita AttributeError se step_type for string
            step_type_name = step_type.name if hasattr(step_type, 'name') else str(step_type)
            logger_callback(f"    ► Treinando passo {i+1}: {step_name} ({step_type_name})")

            # --- A) Define dados de fit para o Source (data-flow strategy) ---
            data_for_fit_src, data_for_fit_tgt = self._choose_fit_datasets(
                step_type=step_type,
                src_step=src_step,
                strategy=strategy,
                curr_src_data=curr_src_data,
                curr_tgt_data=curr_tgt_data,
            )

            # --- B) Fit no Source ---
            src_step.fit(source_dataset=data_for_fit_src, target_dataset=data_for_fit_tgt)

            # --- C) Shadow-copy para o Target, se existir equivalente ---
            tgt_step = mapping.get(i)  # índice do src -> step alvo ou None
            if tgt_step is not None:
                tgt_name = getattr(tgt_step, "name", None) or type(tgt_step).__name__
                logger_callback(f"      ↳ Sincronizando estado com Target: {tgt_name}")
                self._copy_weights(src=src_step, dst=tgt_step)

            # --- D) Avança dados (transform) ---
            curr_src_data = src_step.transform(curr_src_data)

            if tgt_step is not None and curr_tgt_data is not None:
                curr_tgt_data = tgt_step.transform(curr_tgt_data)

        # 4) Persistência
        logger_callback("[Controller] Salvando pipelines treinados...")
        pipe_src.save(source_save_path)
        pipe_tgt.save(target_save_path)
        logger_callback("[Controller] Concluído.")

    # ---------------------------------------------------------------------
    # C1.2 - Data-flow / requires_target_dataset
    # ---------------------------------------------------------------------
    def _choose_fit_datasets(
        self,
        step_type: Union[TransformationType, str],
        src_step: EmbeddingTransformationProtocol,
        strategy: PLDATrainingStrategy,
        curr_src_data: EmbeddingDataset,
        curr_tgt_data: Optional[EmbeddingDataset],
    ) -> Tuple[EmbeddingDataset, Optional[EmbeddingDataset]]:
        """
        Decide quais datasets serão usados no fit do passo Source.

        Regras:
        - Mantém seus "strategy overrides" (CORAL / Whitening / LDA usando target opcionalmente).
        - Mas respeita requires_target_dataset (C1.2): se o step exige target, valida e passa.
        """
        data_for_fit_src: EmbeddingDataset = curr_src_data
        data_for_fit_tgt: Optional[EmbeddingDataset] = None

        # 1) Primeiro: contrato do protocolo
        if getattr(src_step, "requires_target_dataset", False):
            if curr_tgt_data is None:
                raise ValueError(
                    f"[{getattr(src_step, 'name', type(src_step).__name__)}] "
                    f"requires_target_dataset=True, mas curr_tgt_data=None."
                )
            data_for_fit_tgt = curr_tgt_data
            # Mesmo exigindo target, ainda podemos manter data_for_fit_src como curr_src_data
            # (normalmente CORAL usa ambos).

        # 2) Depois: seus overrides por strategy (sem violar o contrato)
        # SANEAMENTO: Usa .value para comparação segura entre Enum e String
        st_val = step_type.value if hasattr(step_type, 'value') else str(step_type)

        if st_val == TransformationType.CORAL_TRANSFORMATION.value:
            # CORAL normalmente exige target. Se strategy tiver flag, respeita; se exige, já garantimos acima.
            if getattr(strategy, "use_target_for_coral", False) and curr_tgt_data is not None:
                data_for_fit_tgt = curr_tgt_data

        elif st_val == TransformationType.WHITENING_TRANSFORMATION.value:
            if getattr(strategy, "use_target_for_whitening", False) and curr_tgt_data is not None:
                # PCA/Whitening global no espaço comum
                data_for_fit_src = curr_src_data + curr_tgt_data

        elif st_val == TransformationType.LDA_TRANSFORMATION.value:
            if getattr(strategy, "use_target_for_lda", False) and curr_tgt_data is not None:
                # LDA supervisionado com labels (se houver)
                data_for_fit_src = curr_src_data + curr_tgt_data

        return data_for_fit_src, data_for_fit_tgt

    # ---------------------------------------------------------------------
    # C1.3 - Validação de compatibilidade / C1.1 - mapeamento por ocorrência
    # ---------------------------------------------------------------------
    def _build_and_validate_mapping(
        self,
        pipe_src: TransformationPipeline,
        pipe_tgt: TransformationPipeline,
        source_config: TransformationPipelineConfig,
        target_config: TransformationPipelineConfig,
    ) -> Dict[int, Optional[EmbeddingTransformationProtocol]]:
        """
        Retorna um dict: {src_index -> tgt_step_or_None}

        Regra forte (C1.3):
        - A sequência de transformation_types do Target deve ser uma SUBSEQUÊNCIA
          da sequência do Source, preservando ordem.
        - Isso evita "extra steps" no target que quebram sincronismo da trajetória.

        Mapeamento (C1.1):
        - O match é feito por TransformationType e ocorrência (não por "primeiro type").
        - Implementação prática: usamos um ponteiro no Target e consumimos em ordem
          quando encontramos um passo do mesmo TransformationType no Source.
        """
        src_types = self._pipeline_types_from_config_and_steps(source_config, pipe_src.steps)
        tgt_types = self._pipeline_types_from_config_and_steps(target_config, pipe_tgt.steps)

        # valida subsequência: target deve caber no source preservando ordem
        mapping: Dict[int, Optional[EmbeddingTransformationProtocol]] = {}
        j = 0  # ponteiro no target

        for i, src_step in enumerate(pipe_src.steps):
            if j < len(pipe_tgt.steps):
                # SANEAMENTO: Comparação segura por valor
                s_val = src_types[i].value if hasattr(src_types[i], 'value') else str(src_types[i])
                t_val = tgt_types[j].value if hasattr(tgt_types[j], 'value') else str(tgt_types[j])
                
                if s_val == t_val:
                    mapping[i] = pipe_tgt.steps[j]
                    j += 1
                else:
                    mapping[i] = None
            else:
                mapping[i] = None

        # Se sobrou coisa no target, então target NÃO é subsequência do source -> erro claro (C1.3)
        if j < len(pipe_tgt.steps):
            remaining = tgt_types[j:]
            # SANEAMENTO: Acesso seguro ao name para a mensagem de erro
            rem_names = [t.name if hasattr(t, 'name') else str(t) for t in remaining]
            raise ValueError(
                "Incompatibilidade entre pipelines (C1.3): "
                "o Target possui steps extras ou fora de ordem em relação ao Source. "
                f"Tipos restantes no Target (a partir do índice {j}): {rem_names}"
            )

        return mapping

    def _pipeline_types_from_config_and_steps(
        self,
        self_config: TransformationPipelineConfig,
        steps: List[EmbeddingTransformationProtocol],
    ) -> List[Union[TransformationType, str]]:
        """
        Extrai a lista de TransformationType na ordem do pipeline.

        Preferência:
        1) config.transformators[i].type (fonte de verdade do "pipeline lógico")
        2) fallback: step.transformation_type (se existir)
        """
        out: List[Union[TransformationType, str]] = []

        # Se a config estiver alinhada em tamanho, use-a como fonte de verdade.
        if hasattr(self_config, "transformators") and len(getattr(self_config, "transformators")) == len(steps):
            for i in range(len(steps)):
                out.append(self_config.transformators[i].type)
            return out

        # Fallback robusto
        for step in steps:
            if hasattr(step, "transformation_type"):
                out.append(step.transformation_type)
            else:
                # último recurso: tentar inferir por class name (não recomendado)
                out.append("UNKNOWN")
        return out

    def _get_step_type_from_source_config(
        self,
        source_config: TransformationPipelineConfig,
        index: int,
        src_step: EmbeddingTransformationProtocol,
    ) -> Union[TransformationType, str]:
        """
        Obtém o tipo do passo para logs/decisões.
        """
        if hasattr(source_config, "transformators") and index < len(source_config.transformators):
            return source_config.transformators[index].type
        if hasattr(src_step, "transformation_type"):
            return src_step.transformation_type
        return "UNKNOWN"

    # ---------------------------------------------------------------------
    # C1.4 - Shadow-copy seguro
    # ---------------------------------------------------------------------
    def _copy_weights(self, src: EmbeddingTransformationProtocol, dst: EmbeddingTransformationProtocol) -> None:
        """
        Transfere o estado treinado de um step para outro.

        Preferência:
        - Usa get_state()/set_state() (C1.4).
        - Fallback:
        - Deepcopy de __dict__ com exclusões (não ideal, mas evita quebrar versões antigas).
        """
        if hasattr(src, "get_state") and hasattr(dst, "set_state"):
            state = copy.deepcopy(src.get_state())
            dst.set_state(state)
            return

        # Fallback legado (idealmente nunca usado após você implementar get_state/set_state)
        state = copy.deepcopy(getattr(src, "__dict__", {}))
        for k in ("math_service", "logger", "name", "_name"):
            state.pop(k, None)
        for key, value in state.items():
            setattr(dst, key, value)