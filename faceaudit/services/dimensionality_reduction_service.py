import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Optional

# Imports atualizados com os nossos novos DTOs e Enums
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData
from faceaudit.models.transformation_step_3d import TransformationStep3D
from faceaudit.enums.step_type import StepType

class DimensionalityReductionService:
    """
    Serviço responsável por projetar trajetórias de embeddings de alta dimensão
    em um espaço 3D, mantendo a consistência por etapa.
    """

    def project_trajectory(self, 
                           trajectory_data: Dict[str, np.ndarray], 
                           target_data: Optional[np.ndarray] = None) -> TransformationTrajectoryData:
        """
        Calcula o PCA por etapa e projeta a transformação para visualização.

        Args:
            trajectory_data (Dict[str, np.ndarray]): Snapshots dos dados (Source) em cada etapa.
            target_data (np.ndarray, opcional): Dados do Target. Se a dimensão não bater com 
                                                a do source na etapa atual, ele será ignorado.

        Returns:
            TransformationTrajectoryData: Objeto encapsulando a lista de passos 3D.
        """
        steps = []
        
        for step_name, source_matrix in trajectory_data.items():
            current_dim = source_matrix.shape[1]
            
            # 1. Validação de Target e Empilhamento Seguro
            # Se a dimensão do Target bater com o Source atual, empilhamos para o PCA.
            # Se não bater, projetamos apenas o Source (evita o crash do vstack).
            valid_target = None
            combined_matrix = source_matrix
            
            if target_data is not None and target_data.shape[1] == current_dim:
                combined_matrix = np.vstack([source_matrix, target_data])
                valid_target = target_data
            
            # 2. Treina o Projetor 3D específico para esta etapa (dimensão)
            # Proteção caso haja menos de 3 amostras no dataset
            n_comp = min(3, combined_matrix.shape[0], combined_matrix.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(combined_matrix)
            
            variance_explained = float(np.sum(pca.explained_variance_ratio_))
            
            # 3. Projeta os dados
            source_3d = pca.transform(source_matrix)
            target_3d = pca.transform(valid_target) if valid_target is not None else None
            
            # Preenchemos com zeros caso n_comp < 3 (raríssimo, mas seguro)
            if source_3d.shape[1] < 3:
                source_3d = np.pad(source_3d, ((0,0), (0, 3 - source_3d.shape[1])))
                if target_3d is not None:
                    target_3d = np.pad(target_3d, ((0,0), (0, 3 - target_3d.shape[1])))

            # 4. Constrói o DTO exato que refatoramos anteriormente
            step_3d = TransformationStep3D(
                step_name=step_name,
                source_points=source_3d,
                target_points=target_3d,
                variance=variance_explained,
                original_dimension=current_dim,
                step_type=StepType.UNKNOWN, # Pode ser preenchido pelo orquestrador depois
                visual_orientation_matrix=pca.components_
            )
            
            steps.append(step_3d)
            
        # Retorna o contêiner tipado correto
        return TransformationTrajectoryData(steps=steps)