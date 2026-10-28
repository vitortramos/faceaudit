"""Cache de mini-thumbnails para tooltips do hover na tabela.

Cada `image_path` é decodificado uma única vez para uma string HTML
contendo `<img src="data:image/png;base64,...">` e cacheado in-memory.
A tabela retorna essa string em `Qt.ToolTipRole`; o Qt detecta o `<` e
renderiza como rich text, mostrando a thumbnail enquanto o usuário
mantém o mouse sobre a linha.

Por que não QPixmap direto no tooltip? O Qt não aceita pixmaps em
QToolTip — só HTML. O caminho via data URI é o jeito padrão de embutir
imagem em rich-text tooltips, e funciona offline (não exige acesso a
filesystem além do já feito uma vez).

Performance: encoding PNG + base64 é O(thumb_size²); a chave é o cache.
Em mouse-move típico, o usuário passa por dezenas de linhas — sem cache
isso seria inviável.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PyQt6.QtGui import QPixmap
from interface.oriented_image import ler_pixmap_orientado


_DEFAULT_THUMB_SIZE = 200  # lado máximo (mantém aspect ratio)


class TooltipThumbCache:
    """Encode lazy de paths em HTML data-URI cacheado.

    `get_html(path)` retorna a string pronta pra `Qt.ToolTipRole` ou None
    quando o path falha (arquivo não existe, decode quebra). Chamadas
    subsequentes pro mesmo path batem o cache.
    """

    def __init__(self, thumb_size: int = _DEFAULT_THUMB_SIZE):
        self._size = thumb_size
        # Cache: path → html string OU None (path quebrado, evita retry)
        self._cache: Dict[str, Optional[str]] = {}

    def get_html(self, path: Optional[str]) -> Optional[str]:
        """Retorna HTML pronto para tooltip ou None.

        Cacheia mesmo o resultado None — paths quebrados não viram retry
        em cada hover.
        """
        if not path:
            return None
        if path in self._cache:
            return self._cache[path]
        html = self._build_html(path)
        self._cache[path] = html
        return html

    def clear(self) -> None:
        """Esvazia o cache. Chamar quando o dataset trocar."""
        self._cache.clear()

    # ── Internals ──────────────────────────────────────────────────────

    def _build_html(self, path: str) -> Optional[str]:
        pix = ler_pixmap_orientado(path)
        if pix.isNull():
            return None
        scaled = pix.scaled(
            self._size, self._size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        ok = scaled.save(buffer, "PNG")
        buffer.close()
        if not ok:
            return None
        b64 = bytes(ba.toBase64()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}">'
