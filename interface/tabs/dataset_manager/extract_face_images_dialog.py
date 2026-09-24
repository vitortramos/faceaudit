"""Gera uma pasta de imagens de face a partir de outra pasta.

Só coleta a intenção — pasta de origem, pasta de destino, forma da imagem
e margem. Nada de regra aqui: quem produz as imagens é o
`FaceImageMaterializationService`, no backend.

Não cria dataset nenhum. Serve para preparar material de laudo (recortes
ilustrativos), alimentar outra ferramenta, ou preparar as imagens antes de
criar o dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets

from faceaudit.enums.dataset_image_kind import DatasetImageKind


class ExtractFaceImagesDialog(QtWidgets.QDialog):
    """Pergunta origem, destino, forma e margem."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        initial_dir: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gerar imagens de face")
        self.setMinimumWidth(620)
        self._initial_dir = initial_dir

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Gera, numa pasta nova, a face de cada imagem da pasta de "
            "origem. <b>Não cria dataset</b> — só as imagens e um "
            "<i>manifest.json</i> com a procedência.",
            parent=self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._ed_origem = QtWidgets.QLineEdit(parent=self)
        self._ed_origem.setPlaceholderText("pasta com as imagens de entrada")
        form.addRow("Origem:", self._linha_com_botao(self._ed_origem, self._escolher_origem))

        self._ed_destino = QtWidgets.QLineEdit(parent=self)
        self._ed_destino.setPlaceholderText("pasta onde as imagens serão gravadas")
        form.addRow("Destino:", self._linha_com_botao(self._ed_destino, self._escolher_destino))

        self._chk_recursivo = QtWidgets.QCheckBox(
            "Incluir subpastas (a estrutura é espelhada no destino)", parent=self,
        )
        self._chk_recursivo.setChecked(True)
        form.addRow("", self._chk_recursivo)

        layout.addLayout(form)
        layout.addWidget(self._grupo_forma())

        self._botoes = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("Gerar")
        self._botoes.accepted.connect(self._on_aceitar)
        self._botoes.rejected.connect(self.reject)
        layout.addWidget(self._botoes)

        self._rb_crop.toggled.connect(self._atualizar_margem)
        self._atualizar_margem()

    # ── Construção ───────────────────────────────────────────────────────

    def _linha_com_botao(self, campo, slot) -> QtWidgets.QWidget:
        linha = QtWidgets.QWidget(parent=self)
        h = QtWidgets.QHBoxLayout(linha)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(campo, 1)
        btn = QtWidgets.QPushButton("Escolher…", parent=linha)
        btn.clicked.connect(slot)
        h.addWidget(btn)
        return linha

    def _grupo_forma(self) -> QtWidgets.QGroupBox:
        grupo = QtWidgets.QGroupBox("Forma da imagem", parent=self)
        v = QtWidgets.QVBoxLayout(grupo)

        self._rb_crop = QtWidgets.QRadioButton(
            DatasetImageKind.CROP.label_pt, parent=grupo,
        )
        self._rb_crop.setChecked(True)
        det_crop = QtWidgets.QLabel(
            DatasetImageKind.CROP.description_pt, parent=grupo,
        )
        det_crop.setWordWrap(True)
        det_crop.setIndent(22)

        self._rb_aligned = QtWidgets.QRadioButton(
            DatasetImageKind.ALIGNED.label_pt, parent=grupo,
        )
        det_aligned = QtWidgets.QLabel(
            DatasetImageKind.ALIGNED.description_pt, parent=grupo,
        )
        det_aligned.setWordWrap(True)
        det_aligned.setIndent(22)

        for w in (self._rb_crop, det_crop, self._rb_aligned, det_aligned):
            v.addWidget(w)

        margem_linha = QtWidgets.QWidget(parent=grupo)
        h = QtWidgets.QHBoxLayout(margem_linha)
        h.setContentsMargins(22, 0, 0, 0)
        self._lbl_margem = QtWidgets.QLabel("Margem do recorte:", parent=margem_linha)
        self._spn_margem = QtWidgets.QDoubleSpinBox(parent=margem_linha)
        self._spn_margem.setRange(1.0, 5.0)
        self._spn_margem.setSingleStep(0.5)
        self._spn_margem.setValue(2.0)
        self._spn_margem.setToolTip(
            "Fator de expansão da caixa da face: 1,0 = caixa exata, "
            "2,0 = o dobro em cada dimensão. Mesma unidade da extração "
            "de faces de vídeo."
        )
        h.addWidget(self._lbl_margem)
        h.addWidget(self._spn_margem)
        h.addStretch(1)
        v.addWidget(margem_linha)

        self._aviso = QtWidgets.QLabel(
            "Imagens em que nenhuma face for detectada são <b>puladas</b> — "
            "não entram na pasta como estão. A contagem e os caminhos ficam "
            "no manifesto.",
            parent=grupo,
        )
        self._aviso.setWordWrap(True)
        v.addWidget(self._aviso)
        return grupo

    # ── Interação ────────────────────────────────────────────────────────

    def _atualizar_margem(self) -> None:
        recorte = self._rb_crop.isChecked()
        self._lbl_margem.setEnabled(recorte)
        self._spn_margem.setEnabled(recorte)

    def _escolher_origem(self) -> None:
        pasta = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pasta com as imagens de origem",
            self._ed_origem.text() or self._initial_dir,
        )
        if pasta:
            self._ed_origem.setText(pasta)
            if not self._ed_destino.text():
                sufixo = "recortes" if self._rb_crop.isChecked() else "alinhadas"
                self._ed_destino.setText(str(Path(pasta).parent / f"{Path(pasta).name}_{sufixo}"))

    def _escolher_destino(self) -> None:
        pasta = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pasta de destino",
            self._ed_destino.text() or self._initial_dir,
        )
        if pasta:
            self._ed_destino.setText(pasta)

    def _on_aceitar(self) -> None:
        origem = self.source_dir()
        destino = self.target_dir()
        if origem is None or not origem.is_dir():
            QtWidgets.QMessageBox.warning(
                self, "Origem inválida",
                "Escolha uma pasta de origem existente.",
            )
            return
        if destino is None:
            QtWidgets.QMessageBox.warning(
                self, "Destino inválido", "Escolha a pasta de destino.",
            )
            return
        if destino.resolve() == origem.resolve():
            QtWidgets.QMessageBox.warning(
                self, "Destino inválido",
                "O destino precisa ser diferente da origem — gravar por cima "
                "das imagens de entrada destruiria o material original.",
            )
            return
        self.accept()

    # ── Resultado ────────────────────────────────────────────────────────

    def source_dir(self) -> Optional[Path]:
        texto = self._ed_origem.text().strip()
        return Path(texto) if texto else None

    def target_dir(self) -> Optional[Path]:
        texto = self._ed_destino.text().strip()
        return Path(texto) if texto else None

    def image_kind(self) -> DatasetImageKind:
        return (
            DatasetImageKind.CROP if self._rb_crop.isChecked()
            else DatasetImageKind.ALIGNED
        )

    def margin_factor(self) -> float:
        return float(self._spn_margem.value())

    def recursive(self) -> bool:
        return self._chk_recursivo.isChecked()
