# faceaudit/services/pipeline_trajectory_service.py

from __future__ import annotations

import matplotlib
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData
from faceaudit.pipelines.embedding_transformation_pipeline import (
    EmbeddingTransformationPipeline,
)
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.services.trajectory_plotter_service import TrajectoryPlotterService


class PipelineTrajectoryService:
    """
    Serviço de trajetória 3D para a nova arquitetura.

    Conecta um EmbeddingTransformationPipeline já treinado à infraestrutura de
    visualização existente (EmbeddingMathService + TrajectoryPlotterService),
    sem modificar nenhum desses componentes.

    Suporta os dois modos do pipeline:

    - LINEAR: itera pipeline.steps, visualiza a evolução do dataset único
      passo a passo. target_dataset é ignorado.
    - DUAL: itera pipeline.source_steps como mestre, sincroniza
      pipeline.target_steps por tipo (subsequência compatível), visualiza
      o alinhamento Source→Target em cada etapa.

    Uso típico (após PIPELINE_TRAINING):
        svc = PipelineTrajectoryService()
        traj = svc.compute_trajectory(pipeline, source_dataset, target_dataset)
        # UI: renderiza traj.steps interativamente com TrajectoryPlotterService
        # ou exporta para disco:
        paths = svc.export_plots(traj, "/projeto/traj/step_{step}.png")
    """

    def __init__(self, math_service: Optional[EmbeddingMathService] = None) -> None:
        self._math = math_service or EmbeddingMathService()
        self._plotter = TrajectoryPlotterService(self._math)

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def compute_trajectory(
        self,
        pipeline: EmbeddingTransformationPipeline,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset] = None,
    ) -> TransformationTrajectoryData:
        """
        Executa o pipeline passo a passo e retorna a trajetória 3D projetada.

        Os frames brutos (matrizes em alta dimensão) são construídos aqui e
        repassados para EmbeddingMathService.compute_trajectory_projection(),
        que aplica Joint PCA, estabilização de Procrustes e a máquina de
        estados de orientação dos elipsoides.

        Args:
            pipeline: Pipeline treinado (fit() ou fit_dual() já executado).
            source_dataset: Dataset de origem. Usado em LINEAR e DUAL.
            target_dataset: Dataset de destino. Obrigatório em DUAL, ignorado
                            em LINEAR.

        Returns:
            TransformationTrajectoryData com um TransformationStep3D por passo
            (incluindo o passo 0 "Raw").

        Raises:
            ValueError: Se pipeline não estiver treinado, ou se DUAL for
                        solicitado sem target_dataset.
        """
        if not pipeline._fitted:
            raise ValueError(
                "O pipeline ainda não foi treinado. Execute fit() ou fit_dual() antes."
            )

        if pipeline.mode == PipelineMode.DUAL and target_dataset is None:
            raise ValueError(
                "target_dataset é obrigatório para pipelines em modo DUAL."
            )

        if pipeline.mode == PipelineMode.LINEAR:
            raw_frames = self._build_linear_frames(pipeline, source_dataset)
        else:
            raw_frames = self._build_dual_frames(pipeline, source_dataset, target_dataset)

        return self._math.compute_trajectory_projection(raw_frames)

    def export_plots(
        self,
        trajectory_data: TransformationTrajectoryData,
        base_path_template: str,
        logger: Callable[[str], None] = print,
    ) -> List[Path]:
        """
        Exporta cada passo da trajetória como PNG no disco (modo headless).

        Usa o mesmo TrajectoryPlotterService da UI interativa, garantindo
        aparência idêntica entre interface e arquivos exportados.

        Args:
            trajectory_data: Resultado de compute_trajectory().
            base_path_template: Template de caminho com placeholder {step}
                                 (índice inteiro, base 0). Ex:
                                 "/projeto/traj/step_{step}.png"
            logger: Função de log (default: print).

        Returns:
            Lista de Path dos arquivos PNG gerados.
        """
        import matplotlib.pyplot as plt

        current_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        saved_paths: List[Path] = []
        n_steps = len(trajectory_data.steps)
        logger(f"🖼️ [Trajectory] Renderizando {n_steps} gráfico(s)...")

        try:
            for idx, step in enumerate(trajectory_data.steps):
                fig = plt.figure(figsize=(10, 6), dpi=150)
                ax1 = fig.add_subplot(121, projection="3d")
                ax2 = fig.add_subplot(122, projection="3d")

                fig.subplots_adjust(
                    left=0.05, right=0.95, top=0.88, bottom=0.10,
                    wspace=0.10, hspace=0,
                )

                self._plotter.plot_step(ax_source=ax1, ax_target=ax2, step=step)

                fig.suptitle(
                    f"Passo {idx + 1}/{n_steps}: {step.step_name}",
                    fontsize=14,
                    fontweight="bold",
                )

                output_path = Path(base_path_template.format(step=idx))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(output_path))
                plt.close(fig)

                saved_paths.append(output_path)
                logger(f"   ✓ {output_path.name}")

        except Exception as e:
            logger(f"⚠️ [Trajectory] Erro ao renderizar passo {idx}: {e}")
        finally:
            matplotlib.use(current_backend)

        logger(f"✅ [Trajectory] {len(saved_paths)} gráfico(s) exportado(s).")
        return saved_paths

    # =========================================================================
    # CONSTRUÇÃO DE FRAMES
    # =========================================================================

    def _build_linear_frames(
        self,
        pipeline: EmbeddingTransformationPipeline,
        source_dataset: EmbeddingDataset,
    ) -> List[SimpleNamespace]:
        """
        Constrói os frames brutos para um pipeline LINEAR.

        Frame 0: matrizes originais sem transformação.
        Frames 1..N: matriz após cada step, target_matrix=None.
        """
        frames: List[SimpleNamespace] = []

        frames.append(SimpleNamespace(
            step_name="0. Original (Raw)",
            source_matrix=source_dataset.matrix.copy(),
            target_matrix=None,
            metrics={},
        ))

        current = source_dataset
        for i, step in enumerate(pipeline.steps):
            current = step.transform(current)
            frames.append(SimpleNamespace(
                step_name=f"{i + 1}. {step.name}",
                source_matrix=current.matrix,
                target_matrix=None,
                metrics=step.get_execution_metrics(),
            ))

        return frames

    def _build_dual_frames(
        self,
        pipeline: EmbeddingTransformationPipeline,
        source_dataset: EmbeddingDataset,
        target_dataset: EmbeddingDataset,
    ) -> List[SimpleNamespace]:
        """
        Constrói os frames brutos para um pipeline DUAL.

        Frame 0: matrizes originais de Source e Target sem transformação.
        Frames 1..N: itera source_steps como mestre; sincroniza target_steps
        por tipo (subsequência compatível, mesma lógica de _build_dual_mapping).
        """
        frames: List[SimpleNamespace] = []

        frames.append(SimpleNamespace(
            step_name="0. Original (Raw)",
            source_matrix=source_dataset.matrix.copy(),
            target_matrix=target_dataset.matrix.copy(),
            metrics={},
        ))

        curr_src = source_dataset
        curr_tgt = target_dataset
        tgt_ptr = 0
        tgt_steps = pipeline.target_steps

        for i, src_step in enumerate(pipeline.source_steps):
            curr_src = src_step.transform(curr_src)

            # Sincroniza com o Target pelo tipo (subsequência)
            if tgt_ptr < len(tgt_steps):
                tgt_step = tgt_steps[tgt_ptr]
                src_val = (
                    src_step.transformation_type.value
                    if hasattr(src_step.transformation_type, "value")
                    else str(src_step.transformation_type)
                )
                tgt_val = (
                    tgt_step.transformation_type.value
                    if hasattr(tgt_step.transformation_type, "value")
                    else str(tgt_step.transformation_type)
                )
                if src_val == tgt_val:
                    curr_tgt = tgt_step.transform(curr_tgt)
                    tgt_ptr += 1

            frames.append(SimpleNamespace(
                step_name=f"{i + 1}. {src_step.name}",
                source_matrix=curr_src.matrix,
                target_matrix=curr_tgt.matrix,
                metrics=src_step.get_execution_metrics(),
            ))

        return frames
