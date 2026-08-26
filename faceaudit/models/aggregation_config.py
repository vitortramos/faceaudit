from dataclasses import dataclass, field
from typing import ClassVar, Optional, Set, Tuple, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Importação dos Enums "Ricos"
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.combination_sampling_mode import CombinationSamplingMode
from faceaudit.enums.quality_score_type import QualityScoreType

T = TypeVar("T", bound="AggregationConfig")

@dataclass
class AggregationConfig(JSONSerializableMixin):
    """
    Configuração independente para a etapa de Agregação de Embeddings (Smart Enrollment).
    Define como múltiplos vetores de uma mesma identidade serão combinados
    para formar um template sintético de maior qualidade.

    Consumido por dois caminhos com a MESMA implementação: o step
    `IDENTITY_AGGREGATION` do pipeline de transformação e a instrução
    autônoma `EMBEDDING_AGGREGATION`.
    """

    # Método matemático de fusão (Ex: Média, Média Ponderada, COMBINATORIAL)
    method: AggregationMethod = field(
        default=AggregationMethod.CENTROID,
        metadata={"help": (
            "<html><b>Centróide</b>: funde todas as amostras da identidade em "
            "UM template (N→1).<br><b>Combinatória</b>: gera vários templates "
            "por subgrupos de k faces (N→M), aumentando o número de amostras "
            "sintéticas por identidade.</html>"
        )},
    )

    # Se o método for COMBINATORIAL, quantos vetores formarão cada subgrupo?
    combination_group_size: int = field(
        default=3,
        metadata={"help": (
            "<html>Quantas faces entram em cada template sintético. Só se "
            "aplica à Combinatória.<br>Identidades com menos faces que este "
            "valor caem em centróide.<br><b>Mínimo 2</b> — subgrupo de uma "
            "face só não agrega nada.</html>"
        )},
    )

    # Limite máximo de templates sintéticos gerados por identidade (evita explosão combinatória)
    max_synthetic_templates_per_id: int = field(
        default=100,
        metadata={"help": (
            "<html>Teto de templates <b>por identidade</b>. O custo bruto é "
            "C(N,k), que explode rápido: 20 faces em subgrupos de 3 dá 1140 "
            "combinações.<br><br>"
            "Quando o total de combinações <b>cabe</b> no teto, todas são "
            "geradas.<br>"
            "Quando <b>excede</b>, são escolhidas exatamente essa quantidade de "
            "combinações distintas — o critério de escolha está no campo "
            "abaixo. Em qualquer caso a seleção é reprodutível: a mesma "
            "configuração produz sempre os mesmos templates.</html>"
        )},
    )

    # Como escolher QUAIS combinações gerar quando C(N,k) passa do teto.
    combination_sampling: CombinationSamplingMode = field(
        default=CombinationSamplingMode.BALANCED_COVERAGE,
        metadata={"help": (
            "<html>Só tem efeito quando o total de combinações <b>excede</b> o "
            "teto — abaixo dele todas são geradas.<br><br>"
            "<b>Cobertura balanceada</b>: cada face participa de "
            "aproximadamente o mesmo número de templates, e nenhuma fica de "
            "fora.<br>"
            "<b>Sorteio uniforme</b>: amostra não-enviesada do espaço de "
            "combinações, mas o sorteio é uniforme sobre combinações e não "
            "sobre faces — a cobertura sai desigual e uma foto pode não entrar "
            "em template nenhum.</html>"
        )},
    )

    # Habilita a ponderação matemática usando scores de qualidade da face
    use_quality_weighting: bool = field(
        default=False,
        metadata={"help": (
            "<html>Pondera a média pela qualidade de cada face, dando mais "
            "peso às melhores.<br>Faces sem a métrica escolhida entram com "
            "peso 1.0 — se a métrica faltar em todas, a média ponderada vira "
            "média simples silenciosamente (a regra "
            "QUALITY_METRIC_AVAILABLE existe para avisar).</html>"
        )},
    )

    # Se a ponderação estiver habilitada, qual score usar (FIQA, OFIQ, etc)?
    quality_weight_type: Optional[QualityScoreType] = field(
        default=QualityScoreType.FIQA,
        metadata={"help": "Métrica de qualidade usada como peso na média."},
    )

    # Campos reativos: alterar qualquer um deles muda quais outros fazem
    # sentido, então o formulário é reconstruído.
    # ClassVar é obrigatório — anotação simples viraria um FIELD do dataclass,
    # entrando no construtor, no to_dict e na própria UI.
    REACTIVE_FIELDS: ClassVar[Tuple[str, ...]] = ("method", "use_quality_weighting")

    def __post_init__(self):
        # Beco sem saída: com a ponderação LIGADA e a métrica em None, o editor
        # inline não renderiza o campo (ele despacha pelo TIPO DO VALOR, e None
        # não casa com nenhum ramo — some em silêncio), enquanto o executor
        # rejeita a execução por "ponderação ativada sem métrica escolhida".
        # O usuário ficaria travado sem nada para clicar. Estado alcançável por
        # JSON editado à mão ou por código que passa None explicitamente.
        if self.use_quality_weighting and self.quality_weight_type is None:
            self.quality_weight_type = QualityScoreType.FIQA

    def get_hidden_fields(self) -> Set[str]:
        """Esconde o que não se aplica à combinação atual de opções.

        Espelha a semântica que o inspector do Pipeline Creator já implementa
        à mão (`_build_aggregation_form`): k e teto só existem na Combinatória,
        e a métrica de qualidade só existe com a ponderação ligada.
        """
        ocultos: Set[str] = set()
        if self.method != AggregationMethod.COMBINATORIAL:
            ocultos.update({
                "combination_group_size", "max_synthetic_templates_per_id",
                "combination_sampling",
            })
        if not self.use_quality_weighting:
            ocultos.add("quality_weight_type")
        return ocultos

    @property
    def quality_key_str(self) -> Optional[str]:
        """Helper rápido para extrair a string da chave ('fiqa', 'ofiq') para o MathService."""
        if self.use_quality_weighting and self.quality_weight_type:
            # Acessa o atributo do Enum (considerando que você tem a estrutura de Enum Rico implementada)
            if hasattr(self.quality_weight_type, 'key_name'):
                return self.quality_weight_type.key_name
            return self.quality_weight_type.name.lower()
        return None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o DTO de Agregação, garantindo o fallback e a conversão correta dos Enums.

        Devolve `cls()` (e não None) para entrada vazia — alinhado com
        `PLDASimilarityModelConfig.from_dict`. Devolver None fazia o editor
        inline exibir "(sem configuração)" quando o JSON trazia `{}`.
        """
        if not data:
            return cls()

        # Puxa os dados para um novo dict para não mutar a entrada original
        processed_data = data.copy()

        # Resiliência: Converte strings de volta para Enums usando 'from_str'
        # (ambos os from_str são idempotentes — membro já hidratado passa reto).
        if 'method' in processed_data:
            processed_data['method'] = AggregationMethod.from_str(processed_data['method'])

        if 'combination_sampling' in processed_data and processed_data['combination_sampling']:
            processed_data['combination_sampling'] = CombinationSamplingMode.from_str(
                processed_data['combination_sampling']
            )

        if 'quality_weight_type' in processed_data and processed_data['quality_weight_type']:
            processed_data['quality_weight_type'] = QualityScoreType.from_str(
                processed_data['quality_weight_type']
            )

        # Descarta chaves desconhecidas: sem isto, um campo renomeado no
        # futuro vira TypeError ao abrir um projeto antigo, em vez de ser
        # ignorado com o default.
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in processed_data.items() if k in valid_keys}
        return cls(**filtered)
