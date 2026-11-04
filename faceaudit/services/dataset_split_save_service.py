"""Grava as duas metades de uma divisão sem perder as imagens.

A divisão (ribbon Dataset → Dividir) produz dois datasets com SUBCONJUNTOS
das faces e os grava numa pasta que em geral não é a do dataset de origem
(`paths.datasets`). As faces continuam com o `image_path` de antes, e num
dataset RELATIVE esse caminho só significa algo junto da âncora
`<pasta do parquet>/<path_root>`. Gravar a metade noutra pasta sem tratar os
caminhos reancora as imagens onde elas não estão — exatamente o que o
`concat` fazia antes de ser corrigido.

Aqui a intenção é declarada: as metades REFERENCIAM as imagens onde elas
estão (`prepare_dataset_for_destination(portable=False)`), que prefere
reapontar o `path_root` para a mesma pasta de imagens e só absolutiza
quando não há caminho relativo possível. Nenhuma imagem é copiada.

Também recusa gravar metade vazia: a divisão por proporção com poucas
identidades pode deixar um lado sem ninguém, e parquet de zero linhas some
dos seletores sem explicação.

Função de regra de negócio — sem Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from faceaudit.services.dataset_localization_service import (
    prepare_dataset_for_destination,
)

if TYPE_CHECKING:
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.services.dataset_path_rebase_service import PathRebaseResult


class EmptySplitHalfError(ValueError):
    """Uma das metades da divisão ficou sem nenhuma face."""


@dataclass(frozen=True)
class SavedSplitHalf:
    """Uma metade gravada: o dataset como foi para o disco e onde."""

    dataset: "FaceDataset"
    path: Path
    rebase: Optional["PathRebaseResult"]


def split_half_path(dest_dir: Union[str, Path], name: str) -> Path:
    """Caminho do parquet de uma metade: ``<destino>/<nome>.parquet``."""
    return Path(dest_dir) / f"{name}.parquet"


def save_split_halves(
    dataset_a: "FaceDataset",
    dataset_b: "FaceDataset",
    dest_dir: Union[str, Path],
) -> "tuple[SavedSplitHalf, SavedSplitHalf]":
    """Prepara e grava as duas metades em `dest_dir`.

    Os datasets recebidos não são mutados; os devolvidos (em
    `SavedSplitHalf.dataset`) são os que foram gravados, já ancorados no
    destino — é esse que a interface deve abrir.

    Raises:
        EmptySplitHalfError: alguma metade não tem faces. Nada é gravado.
    """
    vazias = [ds.name for ds in (dataset_a, dataset_b) if len(ds.faces) == 0]
    if vazias:
        raise EmptySplitHalfError(
            "A divisão deixou sem nenhuma face: "
            + ", ".join(f"'{n}'" for n in vazias)
            + ". Nada foi gravado — ajuste a proporção ou a seleção de "
            "identidades."
        )

    salvos = []
    for ds in (dataset_a, dataset_b):
        destino = split_half_path(dest_dir, ds.name)
        pronto, relatorio = prepare_dataset_for_destination(
            ds, destino, portable=False,
        )
        pronto.to_parquet(destino)
        salvos.append(SavedSplitHalf(pronto, destino, relatorio.rebase))
    return salvos[0], salvos[1]
