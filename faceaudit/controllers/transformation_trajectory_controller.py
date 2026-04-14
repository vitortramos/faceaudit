from typing import List, Optional, Any, Callable
from types import SimpleNamespace 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Models
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData

# Pipeline
from faceaudit.pipelines.transformation_pipeline import TransformationPipeline

# Serviço
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.services.trajectory_plotter_service import TrajectoryPlotterService


class TransformationTrajectoryController:
    """
    Controlador responsável por:
    1. Calcular a trajetória matemática passo-a-passo (Frames).
    2. Converter a trajetória para visualização 3D (Data) usando o MathService.
    3. Exportar os gráficos 3D da trajetória no modo invisível (Headless).
    """

    def __init__(self, math_service: Optional[EmbeddingMathService] = None):
        self.math_service = math_service or EmbeddingMathService()
        self.plotter_service = TrajectoryPlotterService(self.math_service)

    def calculate_trajectory(self, 
                             source_dataset: EmbeddingDataset, 
                             target_dataset: EmbeddingDataset, 
                             source_pipeline: TransformationPipeline,
                             target_pipeline: TransformationPipeline,
                             logger_callback=None) -> List[Any]:
        """
        Executa a inferência passo-a-passo e retorna frames brutos (High-Dim).
        Retorna uma lista de objetos simples (SimpleNamespace) compatíveis com o MathService.
        """
        frames = []
        
        # 0. Frame Inicial (Raw)
        frames.append(SimpleNamespace(
            step_name="0. Original (Raw)",
            source_matrix=source_dataset.matrix.copy(),
            target_matrix=target_dataset.matrix.copy()
        ))

        # Ponteiros de estado
        curr_src = source_dataset
        curr_tgt = target_dataset

        # Controle de fluxo do Target
        tgt_step_index = 0
        tgt_steps = target_pipeline.steps

        # Loop sobre os passos do SOURCE (Mestre)
        for i, src_step in enumerate(source_pipeline.steps):
            step_name = src_step.name if src_step.name else type(src_step).__name__
            
            if logger_callback:
                logger_callback(f"   Aplicando passo {i+1}: {step_name}...")

            # 1. Transforma o Source
            curr_src = src_step.transform(curr_src)

            # 2. Verifica simetria com Target
            if tgt_step_index < len(tgt_steps):
                tgt_step = tgt_steps[tgt_step_index]
                
                # SANEAMENTO: Comparação robusta de tipos (Enum vs Enum ou String vs Enum)
                src_t = src_step.transformation_type
                tgt_t = tgt_step.transformation_type
                
                src_val = src_t.value if hasattr(src_t, 'value') else str(src_t)
                tgt_val = tgt_t.value if hasattr(tgt_t, 'value') else str(tgt_t)

                # Se o tipo do passo for igual (Sincronia)
                if src_val == tgt_val:
                    curr_tgt = tgt_step.transform(curr_tgt)
                    tgt_step_index += 1
            
            # 3. Snapshot com Métricas
            metrics = getattr(src_step, 'get_execution_metrics', lambda: {})()

            frames.append(SimpleNamespace(
                step_name=f"{i+1}. {step_name}",
                source_matrix=curr_src.matrix,
                target_matrix=curr_tgt.matrix,  
                metrics=metrics
            ))

        return frames

    def create_3d_projection(self, frames: List[Any]) -> TransformationTrajectoryData:
        """
        Delega para o MathService a criação do objeto de visualização (Joint PCA 3D).
        """
        if not frames:
            raise ValueError("Não há frames para projetar.")
            
        # O MathService agora usa Joint PCA para travar os eixos
        viz_data = self.math_service.compute_trajectory_projection(frames)
        
        return viz_data
    
    def export_headless_plots(
        self, 
        trajectory_data: 'TransformationTrajectoryData', 
        base_path_template: str, 
        logger_callback: Callable[[str], None] = print
    ) -> list:
        """
        Gera os gráficos invisíveis (Headless) e os salva, delegando 
        a pintura pesada ao TrajectoryPlotterService.
        """
        import matplotlib
        current_backend = matplotlib.get_backend()
        # Força o backend sem interface gráfica para threads
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pathlib import Path

        saved_paths = []
        n_steps = len(trajectory_data.steps)
        logger_callback(f"🖼️ [Export] Renderizando {n_steps} gráficos. Ajustando enquadramento para evitar cortes...")

        try:
            for idx, step in enumerate(trajectory_data.steps):
                # Mantemos a "Tela" (Canvas) em branco
                fig = plt.figure(figsize=(10, 6), dpi=150)
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                
                fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.10, wspace=0.10, hspace=0)
                
                # Delega a pintura para o Serviço Centralizado!
                self.plotter_service.plot_step(ax_source=ax1, ax_target=ax2, step=step)

                # Título Global Headless (Opcional, pois a UI não tem "Título da Janela" no print)
                fig.suptitle(f"Passo {idx+1}/{n_steps}: {step.step_name}", fontsize=14, fontweight='bold')

                # Salva no Disco
                output_path_str = base_path_template.format(step=idx)
                output_path = Path(output_path_str)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(str(output_path))
                plt.close(fig) # Libera memória RAM!
                
                saved_paths.append(output_path)
                
        except Exception as e:
            logger_callback(f"⚠️ [Export] Erro ao renderizar gráfico {idx}: {e}")
        finally:
            matplotlib.use(current_backend)

        return saved_paths