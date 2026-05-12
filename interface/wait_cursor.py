"""Context manager + decorator pra exibir o cursor de espera durante
operações síncronas lentas.

Centraliza o padrão `setOverrideCursor + processEvents + try/finally
restoreOverrideCursor` usado em vários lugares (filtrar, set/replace
batch, abrir dialogs pesados, etc.).

Uso típico:
    from interface.wait_cursor import wait_cursor

    def _on_apply(self):
        with wait_cursor():
            # trabalho síncrono lento — cursor já é ampulheta
            self._apply_batch_replace(...)

Pra slots que demoram pra ter o primeiro repaint (clique → trabalho
pesado imediato), o `processEvents()` no enter força o cursor a
renderizar ANTES do trabalho começar. Sem isso, mesmo o
`setOverrideCursor` chamado primeiro pode demorar pra aparecer porque
o paint event fica enfileirado atrás do trabalho síncrono.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def wait_cursor():
    """Mostra `WaitCursor` enquanto o bloco roda. Força repaint
    imediato via `processEvents` no enter. Restaura no exit (mesmo
    em caso de exceção).

    Stack-safe — múltiplos níveis aninhados funcionam porque
    Qt usa pilha de overrideCursor.
    """
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    QApplication.processEvents()
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def with_wait_cursor(fn: F) -> F:
    """Decorator que envelopa um método/função com `wait_cursor()`.

    Útil pra slots simples onde envolver o corpo todo num `with` é
    redundante:

        @with_wait_cursor
        def _action_set_value(self):
            ...
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with wait_cursor():
            return fn(*args, **kwargs)
    return wrapper  # type: ignore[return-value]
