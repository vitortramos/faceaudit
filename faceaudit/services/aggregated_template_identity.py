"""Reconhecer um template agregado pelo que sobrou dele.

`EMBEDDING_AGGREGATION` funde N faces da mesma identidade num template. O
resultado não é uma foto: `image_path`, `pose` e `quality_scores` saem
vazios de propósito, porque a média de N faces não corresponde a imagem
nenhuma. O `face_hash` deixa de ser hash de arquivo e vira um digest
SHA-256 do conjunto ordenado de faces contribuintes, com prefixo que o
distingue de qualquer foto real.

O digest é de mão única: a lista de faces que formaram o template não fica
gravada em lugar nenhum, então não há como mostrá-las. O que dá para
recuperar é **quantas** eram, porque a agregação a escreve no nome que
inventa para a linha, `AGG_<identidade>_k<N>_<digest>`.

Isso basta para a tabela de comparações não mentir. Sem este módulo ela
mostraria célula vazia onde deveria estar a face, ou pior, o hash
sintético travestido de identificador de arquivo. Com ele a linha diz o
que de fato é: um template de N faces, sem imagem.
"""

from __future__ import annotations

import re
from typing import Optional

# Prefixo do `face_hash` de template agregado. Espelha `_SYNTH_PREFIX` do
# `embedding_math_service`, que é privado e caro de importar daqui; o teste
# sentinela compara os dois para que não possam divergir em silêncio.
SYNTH_PREFIX = "SYNTH_"

# O nome que a agregação inventa: `AGG_<identidade>_k<N>_<digest12>`.
_N_NO_NOME = re.compile(r"_k(\d+)_")


def is_synthetic_template(face_hash: Optional[str]) -> bool:
    """True quando o hash é de um template agregado, e não de uma foto."""
    return bool(face_hash) and str(face_hash).startswith(SYNTH_PREFIX)


def contributing_faces(file_name: Optional[str]) -> Optional[int]:
    """Quantas faces formaram o template, lidas do nome que a agregação deu.

    `None` quando o nome não segue a convenção — template de versão antiga,
    ou nome editado à mão. Chutar um número seria pior que não dizer.
    """
    if not file_name:
        return None
    achado = _N_NO_NOME.search(str(file_name))
    if achado is None:
        return None
    try:
        quantas = int(achado.group(1))
    except ValueError:
        return None
    return quantas if quantas > 0 else None


def template_label(
    face_hash: Optional[str],
    file_name: Optional[str] = None,
) -> Optional[str]:
    """Rótulo da face quando ela é um template; `None` quando é foto.

    Quem chama usa o `None` para seguir com o nome de arquivo normal.
    """
    if not is_synthetic_template(face_hash):
        return None
    quantas = contributing_faces(file_name)
    if quantas is None:
        return "Template agregado"
    return f"Template de {quantas} face" + ("s" if quantas != 1 else "")


def template_tooltip(
    face_hash: Optional[str],
    file_name: Optional[str] = None,
) -> str:
    """Explicação para a dica do mouse, em uma frase.

    Um template é objeto estatisticamente distinto de um embedding de
    imagem única, e quem lê a tabela precisa saber que aquela linha não tem
    foto para conferir.
    """
    quantas = contributing_faces(file_name)
    de_quantas = f"de {quantas} faces " if quantas else ""
    return (
        f"Template agregado {de_quantas}da mesma identidade. Não "
        "corresponde a imagem nenhuma: é a média dos embeddings, e as faces "
        "que o formaram não ficam registradas."
    )
