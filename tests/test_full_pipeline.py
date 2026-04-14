import numpy as np
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.enums.transformation_type import TransformationType

# Importar as classes que acabamos de criar
from faceaudit.transformators.coral_transformation import CoralTransformation
from faceaudit.transformators.whitening_transformation import WhiteningTransformation
from faceaudit.transformators.lda_transformation import LdaTransformation
from faceaudit.transformators.l2_normalization import L2Normalization

def mock_dataset(name, n_samples=100, n_classes=5):
    """Cria dados fake para teste"""
    dim = 512
    # Gera dados aleatórios
    matrix = np.random.rand(n_samples, dim).astype(np.float32)
    items = []
    for i in range(n_samples):
        # Cria identidades fictícias (0, 1, 2, 3, 4...)
        identity = f"person_{i % n_classes}"
        items.append(EmbeddingInfo(
            embedding=matrix[i],
            face_hash=f"hash_{i}",
            image_path=f"/tmp/path/img_{i}.jpg",
            identity=identity,
            file_name=f"file_{i}.jpg",
            # Preenchendo os novos campos para teste
            source="test_generator",
            origin="synthetic_data"
        ))
    return EmbeddingDataset(name=name, items=items)

def pipeline_factory(config: TransformationPipelineConfig, math_service: EmbeddingMathService):
    """
    ESSA É A PEÇA QUE FALTA: O construtor que transforma Config em Objetos.
    Isso geralmente ficaria dentro de um TransformationPipelineService.
    """
    steps = []
    print(f"\n🏭 Construindo Pipeline: {config.name}")
    
    for step_config in config.transformators:
        t_type = step_config.type
        print(f"   -> Instanciando: {t_type}")
        
        # Aqui acontece a mágica de ligar o Enum à Classe
        if t_type == TransformationType.CORAL_TRANSFORMATION:
            step = CoralTransformation(math_service, name=step_config.name)
        elif t_type == TransformationType.WHITENING_TRANSFORMATION:
            # Exemplo de como passar params do config (se houvesse)
            step = WhiteningTransformation(math_service, name=step_config.name)
        elif t_type == TransformationType.LDA_TRANSFORMATION:
            step = LdaTransformation(math_service, name=step_config.name)
        elif t_type == TransformationType.L2_NORMALIZATION:
            step = L2Normalization(math_service, name=step_config.name)
        else:
            print(f"   ⚠️ Tipo desconhecido: {t_type}")
            continue
            
        steps.append(step)
    
    return steps

def run_test():
    # 1. Setup Básico
    math_service = EmbeddingMathService()
    train_ds = mock_dataset("Treino", n_samples=200, n_classes=10)
    calib_ds = mock_dataset("Calibração (Target)", n_samples=50, n_classes=5)

    # 2. Simular uma Configuração vinda da UI (Como se tivéssemos carregado o Pickle)
    # Vamos simular um pipeline complexo: CORAL -> LDA -> Whitening -> L2
    fake_config = TransformationPipelineConfig(
        name="Teste_Integration_Complexo",
        description="Pipeline completa gerada manualmente",
        transformators=[
            TransformatorConfig(type=TransformationType.CORAL_TRANSFORMATION, params={}, name="1_Coral"),
            TransformatorConfig(type=TransformationType.LDA_TRANSFORMATION, params={"n_components": 9}, name="2_LDA"),
            TransformatorConfig(type=TransformationType.WHITENING_TRANSFORMATION, params={}, name="3_White"),
            TransformatorConfig(type=TransformationType.L2_NORMALIZATION, params={}, name="4_L2")
        ]
    )

    # 3. Construir os Objetos (Factory)
    pipeline_steps = pipeline_factory(fake_config, math_service)

    # 4. Executar Treino (FIT)
    print("\n🏋️ Iniciando Treinamento Sequencial...")
    current_ds = train_ds
    
    # O CORAL precisa do calib_ds como target. Os outros não.
    target_ds = calib_ds 

    for step in pipeline_steps:
        print(f"   Running FIT no step: {step.name}")
        
        if step.requires_target_dataset:
            step.fit(current_ds, target_dataset=target_ds)
            # Dica: Normalmente, após o fit do CORAL, o output dele vira o input do próximo fit?
            # Em pipelines de adaptação, sim! Transformamos o treino antes de passar pro próximo fit.
            current_ds = step.transform(current_ds)
        else:
            step.fit(current_ds)
            # Transforma os dados de treino para que o próximo passo aprenda na distribuição já alterada
            current_ds = step.transform(current_ds)

    print("\n✅ Pipeline Treinado com Sucesso!")

    # 5. Executar Inferência (TRANSFORM) em novos dados
    print("\n🧪 Testando Transformação (Inferência)...")
    test_ds = mock_dataset("Teste_Final", n_samples=10)
    
    final_ds = test_ds
    for step in pipeline_steps:
        final_ds = step.transform(final_ds)
    
    print(f"   Output Shape: {final_ds.matrix.shape}")
    print(f"   Exemplo de Norma do vetor final: {np.linalg.norm(final_ds.matrix[0]):.4f}")
    
    if abs(np.linalg.norm(final_ds.matrix[0]) - 1.0) < 1e-5:
        print("   🌟 SUCESSO: O vetor final está normalizado (L2)!")
    else:
        print("   ❌ ERRO: O vetor final não está normalizado.")

if __name__ == "__main__":
    run_test()