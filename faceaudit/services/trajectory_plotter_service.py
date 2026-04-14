import numpy as np
from typing import Tuple
from faceaudit.enums.step_type import StepType
from faceaudit.services.embedding_math_service import EmbeddingMathService

class TrajectoryPlotterService:
    """
    Serviço centralizado para renderização 3D da Trajetória de Transformação.
    Garante que a UI interativa e os relatórios PNG no disco tenham exatamente a mesma aparência.
    """
    
    def __init__(self, math_service: EmbeddingMathService):
        self.math_service = math_service

    def plot_step(self, ax_source, ax_target, step) -> str:
        """
        Pinta os dados do passo atual nos eixos fornecidos (Matplotlib Axes3D).
        
        Retorna:
            str: O texto de auditoria gerado para que o caller (UI ou Exporter) possa usá-lo.
        """
        # 0. Limpa os eixos para o novo frame
        ax_source.clear()
        ax_target.clear()

        src_3d = step.source_points
        tgt_3d = step.target_points
        
        # 1. Calcula limites cúbicos robustos
        xlim, ylim, zlim = self._calculate_step_bounds(src_3d, tgt_3d)

        # 2. Recupera Orientações Isoladas
        src_orientation = getattr(step, 'source_orientation_matrix', None)
        tgt_orientation = getattr(step, 'target_orientation_matrix', None)

        # 3. Elipses (Wireframes) Independentes e Seguras
        src_wire = None; tgt_wire = None
        
        if src_3d is not None and len(src_3d) > 5:
            try: 
                src_wire = self.math_service.get_cov_ellipsoid(
                    src_3d, n_std=2.0, force_eigvecs=src_orientation
                )
            except: pass
            
        if tgt_3d is not None and len(tgt_3d) > 5:
            try: 
                tgt_wire = self.math_service.get_cov_ellipsoid(
                    tgt_3d, n_std=2.0, force_eigvecs=tgt_orientation
                )
            except: pass

        # 4. Plot Source (Esquerda)
        if src_3d is not None:
            ax_source.scatter(src_3d[:,0], src_3d[:,1], src_3d[:,2], c='blue', alpha=0.05, s=10, label='Source')
            if src_wire is not None:
                ax_source.plot_wireframe(*src_wire, color='navy', alpha=0.5, linewidth=1.0)

        # Plot "Fantasma" do Target na Esquerda
        if tgt_wire is not None: 
            ax_source.plot_wireframe(*tgt_wire, color='red', alpha=0.5, linewidth=1.0, label='Target Ref')

        ax_source.set_title("Alinhamento (Source -> Target)")
        
        # Caixa de Auditoria Interna
        audit_text = self._get_audit_text(step)
        ax_source.text2D(0.05, 0.95, audit_text, transform=ax_source.transAxes, 
                         verticalalignment='top', family='monospace', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 5. Plot Target (Direita)
        if tgt_3d is not None:
            ax_target.scatter(tgt_3d[:,0], tgt_3d[:,1], tgt_3d[:,2], c='red', alpha=0.1, s=10)
            if tgt_wire is not None:
                ax_target.plot_wireframe(*tgt_wire, color='darkred', alpha=0.5, linewidth=1.0)

        ax_target.set_title("Referência (Target)")

        # 6. Ajustes Finais Visuais
        for ax in [ax_source, ax_target]:
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_box_aspect([1, 1, 1]) 
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
            ax.grid(False) 
            
        return audit_text

    def _calculate_step_bounds(self, src_3d, tgt_3d) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Calcula os limites CÚBICOS usando Percentis (Robustez a Outliers)."""
        data_list = []
        if src_3d is not None: data_list.append(src_3d)
        if tgt_3d is not None: data_list.append(tgt_3d)
        
        if not data_list:
            return (-1, 1), (-1, 1), (-1, 1)
            
        combined = np.vstack(data_list)
        min_vals = np.percentile(combined, 2, axis=0)
        max_vals = np.percentile(combined, 98, axis=0)
        center = (max_vals + min_vals) / 2
        ranges = max_vals - min_vals
        half_range = (np.max(ranges) * 1.05) / 2 
        
        return (
            (float(center[0] - half_range), float(center[0] + half_range)),
            (float(center[1] - half_range), float(center[1] + half_range)),
            (float(center[2] - half_range), float(center[2] + half_range))
        )

    def _get_audit_text(self, step) -> str:
        """Gera o texto técnico para a legenda interna do gráfico."""
        m = step.metrics
        st = step.step_type
        lines = []

        lines.append(f"STEP: {step.step_name}")
        lines.append(f"Shape: {step.original_dimension}D")
        lines.append("-" * 20)

        if st == StepType.RAW:
            lines.append(f"N Samples: {m.get('n_samples_source', '?')}") 
            lines.append("Status: Original")
        elif st == StepType.WHITENING:
            lines.append(f"Redução: {m.get('original_dimension', '?')} -> {m.get('output_dimension', '?')}")
            lines.append(f"Var Preserved: {m.get('explained_variance_percent', '?')}")
        elif st == StepType.CORAL:
            est = m.get('estimator_used', 'Standard') 
            if hasattr(est, 'split'): est = est.split('.')[-1]
            lines.append(f"Method: {est}")
            lines.append(f"Frob Dist: {m.get('frobenius_distance', 0.0):.4f}")
            lines.append(f"Reg: {m.get('regularization_used', 0.0):.1e}")
        elif st == StepType.LDA:
            lines.append(f"Classes: {m.get('n_classes_found', '?')}")
            lines.append(f"Projection: {m.get('output_dimension', '?')}D")
        elif st == StepType.L2_NORM:
            lines.append("Geometry: Hypersphere")
            lines.append("Norm: Fixed (1.0)")
            
        return "\n".join(lines)