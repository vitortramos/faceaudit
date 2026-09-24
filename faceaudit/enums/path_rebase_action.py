from enum import Enum


class PathRebaseAction(Enum):
    """O que foi feito com os caminhos de imagem ao gravar o dataset noutra pasta.

    Um dataset RELATIVE resolve as imagens como
    `<pasta do parquet>/<path_root>/<image_path>`. Gravar o parquet noutra
    pasta sem tocar no `path_root` move o ponto de partida e quebra todos os
    caminhos silenciosamente.
    """

    NOT_NEEDED = "not_needed"
    """Nada a fazer: dataset ABSOLUTE, ou destino na mesma pasta da origem."""

    REBASED = "rebased"
    """`path_root` reapontado para a mesma pasta de imagens, e verificado."""

    REBASED_UNVERIFIED = "rebased_unverified"
    """`path_root` reapontado, mas as imagens já não eram encontradas na
    origem — o ponteiro ficou equivalente, sem piorar nem consertar."""

    ABSOLUTIZED = "absolutized"
    """Sem caminho relativo possível (outro drive, ou reapontamento não
    resolveu): cada `image_path` virou absoluto e o modo passou a ABSOLUTE."""
