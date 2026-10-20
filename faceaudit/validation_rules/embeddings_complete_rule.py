# faceaudit/validation_rules/embeddings_complete_rule.py

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


# Quantas amostras inválidas a mensagem nomeia. O resto vira contagem —
# a mensagem vai para o log e para a interface, e uma lista de milhares de
# nomes não ajudaria ninguém a achar o problema.
_MAX_NOMEADAS = 10


class EmbeddingsCompleteRule(ValidationRuleBase):
    """
    Verifica que TODA amostra do dataset tem um embedding utilizável.

    Existe porque o carregador aceita linha sem embedding em silêncio: o
    valor nulo do parquet vira `np.array(None, float32)`, um escalar `nan`,
    e o primeiro sinal do problema é o empilhamento das matrizes falhar
    dentro do modelo com "array at index N has size 1" — mensagem que não
    diz qual dataset nem qual imagem. Foi assim que um dataset de vídeo com
    metade das faces sem biometria chegou ao confronto.

    Inválida é a amostra cujo embedding está ausente, não é vetor 1-D, tem
    valor não finito (NaN/inf) ou tem dimensão diferente da maioria. A
    dimensão de referência é a mais frequente entre os vetores válidos:
    o dataset não declara a dele, e a moda é o que a maioria das linhas
    afirma.

    Target esperado: qualquer instrução (o target não é lido).

    Parâmetros:
        dataset_resource_id: id do recurso DATASET a verificar
        max_named: quantas amostras inválidas nomear na mensagem
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        max_named: int = _MAX_NOMEADAS,
        name: str = "Embeddings completos",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que toda amostra de '{dataset_resource_id}' tem "
                "embedding válido (presente, vetor 1-D, finito e com a "
                "mesma dimensão das demais)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._max_named = max(0, int(max_named))

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._dataset_id):
            result.add_error(
                f"[{self.rule_id}] Dataset '{self._dataset_id}' não encontrado no contexto."
            )
            return result

        try:
            resource = context.ensure_payload(self._dataset_id)
        except Exception as exc:
            result.add_error(
                f"[{self.rule_id}] Não foi possível carregar '{self._dataset_id}': {exc}"
            )
            return result
        payload = resource.payload if resource is not None else None
        items = getattr(payload, "items", None)
        if items is None:
            result.add_warning(
                f"[{self.rule_id}] Payload de '{self._dataset_id}' indisponível "
                "ou sem amostras enumeráveis. Pulando checagem de embeddings."
            )
            return result

        invalidas, dim_ref, n_total = self._classificar(items)
        if not invalidas:
            result.mark_ok()
            return result

        result.add_error(
            f"[{self.rule_id}] {len(invalidas)} de {n_total} amostra(s) de "
            f"'{self._dataset_id}' sem embedding válido"
            + (f" (dimensão esperada: {dim_ref})" if dim_ref else "")
            + ". Exemplos: " + self._descrever(invalidas)
            + ". Corrija o dataset (enriquecer de novo ou excluir essas "
            "faces com justificativa) antes de calcular escores."
        )
        return result

    @staticmethod
    def _classificar(items) -> Tuple[List[Tuple[Any, str]], Optional[int], int]:
        """(inválidas com motivo, dimensão de referência, total)."""
        dims: List[Optional[int]] = []
        motivos: List[Optional[str]] = []
        lista = list(items)
        for item in lista:
            emb = getattr(item, "embedding", None)
            if emb is None:
                dims.append(None); motivos.append("ausente")
                continue
            try:
                arr = np.asarray(emb, dtype=np.float64)
            except (TypeError, ValueError):
                dims.append(None); motivos.append("não numérico")
                continue
            if arr.ndim != 1 or arr.size == 0:
                # O caso do parquet com nulo: escalar nan, ndim 0.
                dims.append(None); motivos.append("ausente")
                continue
            if not np.all(np.isfinite(arr)):
                dims.append(None); motivos.append("contém NaN/inf")
                continue
            dims.append(int(arr.size)); motivos.append(None)

        validas = [d for d in dims if d is not None]
        dim_ref = Counter(validas).most_common(1)[0][0] if validas else None

        invalidas: List[Tuple[Any, str]] = []
        for item, d, motivo in zip(lista, dims, motivos):
            if motivo is not None:
                invalidas.append((item, motivo))
            elif dim_ref is not None and d != dim_ref:
                invalidas.append((item, f"dimensão {d}"))
        return invalidas, dim_ref, len(lista)

    def _descrever(self, invalidas: List[Tuple[Any, str]]) -> str:
        partes = []
        for item, motivo in invalidas[: self._max_named]:
            nome = (
                getattr(item, "file_name", None)
                or getattr(item, "image_path", None)
                or (str(getattr(item, "face_hash", "") or "")[:12] or "?")
            )
            partes.append(f"{nome} ({motivo})")
        restantes = len(invalidas) - len(partes)
        texto = "; ".join(partes)
        if restantes > 0:
            texto += f"; e mais {restantes}"
        return texto
