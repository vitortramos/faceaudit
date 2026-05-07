"""Diálogo de comparação visual de duas faces lado a lado.

Estrutura:
  ┌───────────────────────────────────────────────────────────────────┐
  │ Comparação de Faces                                                │
  ├───────────────────────────────────────────────────────────────────┤
  │ ┌─────────────────────────┐    ┌─────────────────────────┐        │
  │ │   [imagem 1, ~400px]    │    │   [imagem 2, ~400px]    │        │
  │ │                         │    │                         │        │
  │ └─────────────────────────┘    └─────────────────────────┘        │
  │ alice                            alice2                            │
  │ /path/img_0.jpg                  /path/img_1.jpg                   │
  │ FIQA: 85.0                       FIQA: 72.0                        │
  │ ...                              ...                               │
  ├───────────────────────────────────────────────────────────────────┤
  │ Similaridade (cosine): 0.823                                       │
  ├───────────────────────────────────────────────────────────────────┤
  │                                                          [Fechar]  │
  └───────────────────────────────────────────────────────────────────┘

Stateless — o controller monta os dados e injeta via setter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6 import QtCore, QtGui, QtWidgets


_THUMB_SIZE = 360            # lado do quadro de imagem em pixels
_PROPS_LABEL_WIDTH = 80      # largura fixa do label "FIQA:" etc.


class ComparePairView(QtWidgets.QDialog):
    """View do diálogo de comparação. Sem lógica — só renderização."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Comparação de Faces")
        self.setModal(True)
        self.resize(900, 660)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Header
        self.lbl_header = QtWidgets.QLabel("Comparação de Faces")
        self.lbl_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a3d6e;"
        )
        outer.addWidget(self.lbl_header)

        # Linha de panels (2 lado-a-lado)
        panels_row = QtWidgets.QHBoxLayout()
        panels_row.setSpacing(20)
        self.panel_left = _FacePanel(self)
        self.panel_right = _FacePanel(self)
        panels_row.addWidget(self.panel_left, 1)
        panels_row.addWidget(self.panel_right, 1)
        outer.addLayout(panels_row, 1)

        # Bloco de comparação inter-pares (similarity, etc.)
        self.lbl_similarity = QtWidgets.QLabel("Similaridade: —")
        self.lbl_similarity.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_similarity.setStyleSheet(
            "font-size: 13px; padding: 8px; "
            "background-color: #f3f6fa; border-radius: 4px; color: #333;"
        )
        outer.addWidget(self.lbl_similarity)

        # Botão fechar
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        outer.addLayout(bottom)

    # ── API pública ───────────────────────────────────────────────────

    def set_pair(
        self,
        face_left: Any,
        face_right: Any,
        similarity_label: Optional[str] = None,
    ) -> None:
        """Popula os 2 painéis com objetos `FaceInfo` e o label de
        comparação. `similarity_label` é uma string já formatada (ex:
        "Cosine = 0.823 (par genuíno provável)") — o controller decide
        a semântica."""
        self.panel_left.set_face(face_left)
        self.panel_right.set_face(face_right)
        if similarity_label:
            self.lbl_similarity.setText(similarity_label)
        else:
            self.lbl_similarity.setText("Similaridade: indisponível "
                                        "(sem embedding em uma das faces)")


class _FacePanel(QtWidgets.QFrame):
    """Painel individual: imagem grande + bloco de propriedades."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #fafafa; border: 1px solid #ddd; "
            "border-radius: 6px; }"
        )
        self._build_ui()

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Imagem
        self.lbl_img = QtWidgets.QLabel()
        self.lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setMinimumSize(_THUMB_SIZE, _THUMB_SIZE)
        self.lbl_img.setStyleSheet(
            "background: #eee; border: 1px solid #ccc; border-radius: 4px;"
        )
        v.addWidget(self.lbl_img, 0, QtCore.Qt.AlignmentFlag.AlignCenter)

        # Identidade em destaque
        self.lbl_identity = QtWidgets.QLabel("(sem identidade)")
        self.lbl_identity.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1a3d6e; padding-top: 4px;"
        )
        self.lbl_identity.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_identity)

        # Path em fonte pequena
        self.lbl_path = QtWidgets.QLabel("")
        self.lbl_path.setStyleSheet("color: #666; font-size: 10px;")
        self.lbl_path.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_path.setWordWrap(True)
        v.addWidget(self.lbl_path)

        # Bloco de propriedades em form layout
        self.props_form = QtWidgets.QFormLayout()
        self.props_form.setSpacing(2)
        self.props_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.props_form.setHorizontalSpacing(8)
        # Embrulha em widget pra estilizar (background + padding)
        props_widget = QtWidgets.QWidget()
        props_widget.setLayout(self.props_form)
        props_widget.setStyleSheet(
            "QWidget { background: #fff; border: 1px solid #eee; "
            "border-radius: 3px; }"
        )
        v.addWidget(props_widget)

    def set_face(self, face: Any) -> None:
        """Recarrega o painel com os dados de uma FaceInfo."""
        # Imagem
        path = getattr(face, "image_path", "") or ""
        pix = QtGui.QPixmap(path) if path else QtGui.QPixmap()
        if not pix.isNull():
            scaled = pix.scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_img.setPixmap(scaled)
            self.lbl_img.setStyleSheet(
                "background: #fff; border: 1px solid #ccc; border-radius: 4px;"
            )
        else:
            self.lbl_img.setPixmap(QtGui.QPixmap())
            self.lbl_img.setText("⚠️ arquivo\nnão encontrado")
            self.lbl_img.setStyleSheet(
                "background: #fff3e0; border: 1px solid #ffb74d; "
                "color: #c62828; font-size: 12px; border-radius: 4px;"
            )

        # Identidade + path
        identity = getattr(face, "identity", None) or "(sem identidade)"
        self.lbl_identity.setText(str(identity))
        path_short = Path(path).name if path else "(sem path)"
        self.lbl_path.setText(path_short)
        self.lbl_path.setToolTip(path)

        # Limpa form de propriedades anteriores
        while self.props_form.rowCount() > 0:
            self.props_form.removeRow(0)

        # Adiciona linhas conforme atributos disponíveis
        for label_text, value in _extract_properties(face):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setFixedWidth(_PROPS_LABEL_WIDTH)
            lbl.setStyleSheet(
                "font-size: 11px; color: #555; font-weight: bold;"
            )
            val_lbl = QtWidgets.QLabel(value)
            val_lbl.setStyleSheet(
                "font-size: 11px; color: #222;"
            )
            val_lbl.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.props_form.addRow(lbl, val_lbl)


# ── Helpers livres ────────────────────────────────────────────────────

def _extract_properties(face: Any):
    """Itera sobre atributos relevantes de FaceInfo e produz
    `(label, value_str)` pairs pra o form de propriedades.

    Cobre: hash, gender/ethnicity/origin, FIQA/det_score (FFInfo),
    OFIQ score, SERFIQ score, yaw/pitch/roll. Pula campos None ou ""
    pra evitar poluição visual.
    """
    rows = []
    h = getattr(face, "hash", None)
    if h:
        rows.append(("Hash:", _short_hash(h)))

    for attr_name, label in [("gender", "Gênero:"),
                             ("ethnicity", "Etnia:"),
                             ("origin", "Origem:")]:
        v = getattr(face, attr_name, None)
        if v:
            rows.append((label, str(v)))

    ff = getattr(face, "ff_info", None)
    if ff is not None:
        for attr, fmt, label in [
            ("fiqa_score", "{:.2f}", "FIQA:"),
            ("det_score", "{:.3f}", "Det.:"),
            ("yaw", "{:.1f}°", "Yaw:"),
            ("pitch", "{:.1f}°", "Pitch:"),
            ("roll", "{:.1f}°", "Roll:"),
        ]:
            v = getattr(ff, attr, None)
            if v is not None:
                try:
                    rows.append((label, fmt.format(float(v))))
                except (TypeError, ValueError):
                    rows.append((label, str(v)))

    ofiq = getattr(face, "ofiq_info", None)
    if ofiq is not None:
        s = getattr(ofiq, "unified_quality_score_scalar", None)
        if s is not None:
            try:
                rows.append(("OFIQ:", f"{float(s):.2f}"))
            except (TypeError, ValueError):
                pass

    serfiq = getattr(face, "serfiq_info", None)
    if serfiq is not None:
        s = getattr(serfiq, "score", None)
        if s is not None:
            try:
                rows.append(("SER-FIQ:", f"{float(s):.2f}"))
            except (TypeError, ValueError):
                pass

    return rows


def _short_hash(h: str, n: int = 12) -> str:
    """Trunca hashes longos pra caberem na UI sem perder distinção visual."""
    s = str(h)
    return s if len(s) <= n + 3 else s[:n] + "…"
