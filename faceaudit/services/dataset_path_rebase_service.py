"""Preserva os caminhos de imagem quando o parquet é gravado noutra pasta.

Um dataset RELATIVE resolve as imagens como
`<pasta do parquet>/<path_root>/<image_path>`. Um "Salvar Como" para outra
pasta muda o ponto de partida: o mesmo `path_root` passa a apontar para um
lugar que normalmente não existe, e TODAS as imagens somem de uma vez, sem
erro nenhum — o parquet grava bem, só o ponteiro fica errado.

Caso real: `RGMA-Sepael_9-14_16-21_mulher.parquet` (em `datasets/`, com
`path_root=RGMA-Sepael_9-14_16-21_mulher_images`) salvo como
`datasets/Confronto 2/RGMA-9-14.parquet`. O `path_root` herdado passou a
apontar para `datasets/Confronto 2/RGMA-Sepael_9-14_16-21_mulher_images/`,
que não existe. A importação para um projeto com cópia de imagens descartou
as 300 faces e gravou um dataset vazio.

Função pura de regra de negócio — sem Qt. MUTA o dataset recebido.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.path_rebase_action import PathRebaseAction
from faceaudit.services.image_path_resolver import resolve_image_path


@dataclass(frozen=True)
class PathRebaseResult:
    """O que foi feito, e uma frase pronta para mostrar ao usuário."""

    action: PathRebaseAction
    path_root: Optional[str]
    message: str

    @property
    def changed(self) -> bool:
        return self.action is not PathRebaseAction.NOT_NEEDED


def _faces(dataset: Any) -> List[Any]:
    """Faces do dataset — `faces` no FaceDataset, `items` no EmbeddingDataset."""
    return list(getattr(dataset, "faces", None) or getattr(dataset, "items", None) or [])


def _first_with_image(dataset: Any) -> Optional[Any]:
    return next((f for f in _faces(dataset) if getattr(f, "image_path", None)), None)


def rebase_dataset_paths(
    dataset: Any,
    *,
    source_parquet_dir: Optional[Union[str, Path]],
    target_parquet_dir: Union[str, Path],
) -> PathRebaseResult:
    """Ajusta os caminhos de `dataset` para continuarem válidos em `target_parquet_dir`.

    Preferência por reapontar o `path_root` (mantém o dataset portátil);
    se isso não for possível ou não resolver, absolutiza os caminhos.
    Chame ANTES de `to_parquet(<destino>)`.
    """
    if getattr(dataset, "path_mode", None) != PathMode.RELATIVE:
        return PathRebaseResult(
            PathRebaseAction.NOT_NEEDED, getattr(dataset, "path_root", None),
            "Caminhos absolutos — nada a ajustar.",
        )
    if source_parquet_dir is None:
        return PathRebaseResult(
            PathRebaseAction.NOT_NEEDED, getattr(dataset, "path_root", None),
            "Pasta de origem desconhecida — caminhos mantidos como estão.",
        )

    source_dir = Path(source_parquet_dir).resolve()
    target_dir = Path(target_parquet_dir).resolve()
    if source_dir == target_dir:
        return PathRebaseResult(
            PathRebaseAction.NOT_NEEDED, dataset.path_root,
            "Mesma pasta de origem — nada a ajustar.",
        )

    old_root = dataset.path_root
    images_dir = (source_dir / (old_root or ".")).resolve()
    probe = _first_with_image(dataset)
    resolved_before = (
        resolve_image_path(
            probe, parquet_dir=source_dir,
            path_mode=PathMode.RELATIVE, path_root=old_root,
        )
        if probe is not None else None
    )
    was_resolvable = bool(resolved_before and resolved_before.exists())

    # 1ª escolha: reapontar `path_root` para a MESMA pasta de imagens.
    try:
        new_root = Path(os.path.relpath(images_dir, target_dir)).as_posix()
    except ValueError:
        new_root = None  # drives diferentes no Windows

    if new_root is not None:
        resolved_after = (
            resolve_image_path(
                probe, parquet_dir=target_dir,
                path_mode=PathMode.RELATIVE, path_root=new_root,
            )
            if probe is not None else None
        )
        if resolved_after is not None and resolved_after.exists():
            dataset.path_root = new_root
            dataset._parquet_dir = target_dir
            return PathRebaseResult(
                PathRebaseAction.REBASED, new_root,
                f"A subpasta de imagens foi reapontada de '{old_root or '(raiz)'}' "
                f"para '{new_root}', que é a mesma pasta de imagens de antes.",
            )
        if not was_resolvable:
            # Já estava quebrado na origem: reapontar mantém o ponteiro
            # equivalente. Não cabe absolutizar um caminho inexistente.
            dataset.path_root = new_root
            dataset._parquet_dir = target_dir
            return PathRebaseResult(
                PathRebaseAction.REBASED_UNVERIFIED, new_root,
                f"A subpasta de imagens foi reapontada de '{old_root or '(raiz)'}' "
                f"para '{new_root}'. Atenção: as imagens já não eram encontradas "
                f"na origem ({images_dir}), então continuam sem ser encontradas.",
            )

    # 2ª escolha: absolutizar a partir da resolução ORIGINAL.
    for face in _faces(dataset):
        if not getattr(face, "image_path", None):
            continue
        try:
            face.image_path = str(
                resolve_image_path(
                    face, parquet_dir=source_dir,
                    path_mode=PathMode.RELATIVE, path_root=old_root,
                )
            )
        except Exception:
            continue  # path irresolvível: preserva o que já está lá
    dataset.path_mode = PathMode.ABSOLUTE
    dataset.path_root = None
    dataset._parquet_dir = target_dir
    return PathRebaseResult(
        PathRebaseAction.ABSOLUTIZED, None,
        "Não havia caminho relativo possível até a pasta de imagens; os "
        "caminhos foram gravados como absolutos, apontando para "
        f"{images_dir}.",
    )
