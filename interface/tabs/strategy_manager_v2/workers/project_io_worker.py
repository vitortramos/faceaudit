# interface/tabs/strategy_manager_v2/workers/project_io_worker.py

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore

from faceaudit.services.project_service import FaceAuditProject, FaceAuditProjectService


class ProjectOpenWorker(QtCore.QObject):
    """
    Worker para abertura de projeto em thread separada.

    A abertura com verify_integrity=True pode ser lenta em projetos grandes
    (recalcula SHA-256 de todos os artefatos). Rodar em thread evita congelar a UI.

    Sinais:
        log_message(str)
        finished(FaceAuditProject)
        error(str)
    """

    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)   # FaceAuditProject
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        fa_path: Path,
        verify_integrity: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fa_path = fa_path
        self._verify_integrity = verify_integrity

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.log_message.emit(f"Abrindo projeto: {self._fa_path}")
            project = FaceAuditProjectService.open(
                self._fa_path,
                verify_integrity=self._verify_integrity,
            )
            if self._verify_integrity and project.has_integrity_problems:
                problems = [
                    r.describe()
                    for r in (project.integrity_results or [])
                    if r.is_problem
                ]
                for p in problems:
                    self.log_message.emit(f"⚠ {p}")
                self.log_message.emit(
                    f"Projeto aberto com {len(problems)} problema(s) de integridade."
                )
            else:
                self.log_message.emit("Projeto aberto com sucesso. Integridade OK.")

            self.finished.emit(project)

        except Exception as exc:
            msg = f"Erro ao abrir projeto: {exc}"
            self.log_message.emit(msg)
            self.error.emit(msg)
