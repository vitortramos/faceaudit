"""Avaliador AST seguro para colunas computadas (PR 23 — Tema 4.3).

Permite ao usuário definir uma fórmula simbólica (ex: `yaw_abs = abs(yaw)`)
que é avaliada por linha no `DatasetTableModel.data()`. Read-only — o
resultado nunca é gravado no DataFrame nem persistido na coluna; só o
`computed_expr` (string) round-trippa via `column_specs` na metadata do
parquet.

**Sandbox**: NÃO usa `eval`/`exec`. Walk explícito do AST do Python com
whitelist de tipos de nó e funções. Sem acesso a `__builtins__`,
`__import__`, `getattr`, atributos ou métodos. Variáveis vêm do `row`
dict (colunas do DataFrame da face).

**Limitações deliberadas**:
- Sem chamadas encadeadas (`a.b.c`) — não há AST Attribute permitido.
- Sem comprehensions/lambdas/generators.
- Sem subscripts (`a[0]`).
- Funções: apenas as listadas em `_ALLOWED_FUNCTIONS`.
- Colunas computadas referenciando outras colunas computadas: não suportado.
  O `row` dict é construído só do DataFrame raw — referências a outras
  computadas resolvem como NameError → exibido como "#ERR" pela UI.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, Optional


# Funções built-in seguras. Sem `open`, `eval`, `exec`, `__import__`, etc.
_ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    # Constantes / utilitários numéricos comuns
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_COMPARE_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class ExpressionError(ValueError):
    """Erro de avaliação ou validação da expressão."""


def validate_expression(expr: str) -> None:
    """Faz parse + walk inicial sem avaliar (sem row context). Levanta
    `ExpressionError` se a sintaxe for inválida ou contiver nós/funções
    não permitidos. Identifica nomes desconhecidos? Não — `Name` é
    permitido sempre, lookup falha em runtime se a coluna não existir.

    Útil pra validação no momento de criar a coluna (UI pode mostrar
    erro antes de aceitar).
    """
    if not expr or not expr.strip():
        raise ExpressionError("Expressão vazia.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Sintaxe inválida: {e.msg}") from e

    # Walk validation — chama _validate_node em cada nó, levanta se
    # encontrar tipo proibido.
    _validate_node(tree.body)


def evaluate_expression(expr: str, row: Dict[str, Any]) -> Any:
    """Avalia a expressão `expr` usando `row` como namespace de variáveis.

    Args:
        expr: fórmula simbólica (ex: `"abs(yaw) + 1.0"`).
        row: dict {nome_coluna: valor} — tipicamente um row do DataFrame
             convertido via `df.iloc[i].to_dict()`.

    Returns:
        Resultado da avaliação. Tipo depende da expressão.

    Raises:
        ExpressionError: erro de sintaxe, nó proibido, ou função não
        permitida. NameError vira ExpressionError pra capturar referência
        a coluna ausente.
    """
    if not expr or not expr.strip():
        raise ExpressionError("Expressão vazia.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Sintaxe inválida: {e.msg}") from e
    try:
        return _eval_node(tree.body, row)
    except ExpressionError:
        raise
    except (TypeError, ValueError, ZeroDivisionError, KeyError, NameError) as e:
        raise ExpressionError(str(e)) from e


# ──────────────────────────────── Internos ───────────────────────────────

def _validate_node(node: ast.AST) -> None:
    """Walk recursivo sem avaliação, valida tipos de nó."""
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ExpressionError(f"Operador binário não permitido: {type(node.op).__name__}")
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ExpressionError(f"Operador unário não permitido: {type(node.op).__name__}")
        _validate_node(node.operand)
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ExpressionError(f"BoolOp não permitido: {type(node.op).__name__}")
        for v in node.values:
            _validate_node(v)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE_OPS:
                raise ExpressionError(f"Comparação não permitida: {type(op).__name__}")
        _validate_node(node.left)
        for c in node.comparators:
            _validate_node(c)
        return
    if isinstance(node, ast.IfExp):
        _validate_node(node.test)
        _validate_node(node.body)
        _validate_node(node.orelse)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Apenas chamadas a funções nomeadas (sem `obj.metodo()`).")
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise ExpressionError(
                f"Função '{node.func.id}' não está na whitelist. "
                f"Disponíveis: {sorted(_ALLOWED_FUNCTIONS.keys())}"
            )
        if node.keywords:
            raise ExpressionError("Argumentos nomeados não suportados.")
        for arg in node.args:
            _validate_node(arg)
        return
    raise ExpressionError(f"Nó AST não permitido: {type(node).__name__}")


def _eval_node(node: ast.AST, row: Dict[str, Any]) -> Any:
    """Avaliação recursiva. Pressupõe `_validate_node` já passou (via
    `evaluate_expression` ou checagem prévia)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in row:
            return row[node.id]
        # Permite constantes no namespace (math.pi etc) — não suportado por
        # ora pra manter sandbox simples.
        raise ExpressionError(f"Coluna '{node.id}' não existe no dataset.")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"Operador binário não permitido: {type(node.op).__name__}")
        return op(_eval_node(node.left, row), _eval_node(node.right, row))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"Operador unário não permitido: {type(node.op).__name__}")
        return op(_eval_node(node.operand, row))
    if isinstance(node, ast.BoolOp):
        # Short-circuit: avalia da esquerda pra direita.
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _eval_node(v, row)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = _eval_node(v, row)
                if result:
                    return result
            return result
        raise ExpressionError(f"BoolOp não permitido: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        # Suporta cadeias: a < b <= c (curto-circuito)
        left = _eval_node(node.left, row)
        for op_node, right_node in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise ExpressionError(f"Comparação não permitida: {type(op_node).__name__}")
            right = _eval_node(right_node, row)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        if _eval_node(node.test, row):
            return _eval_node(node.body, row)
        return _eval_node(node.orelse, row)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Apenas chamadas a funções nomeadas.")
        fn = _ALLOWED_FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ExpressionError(f"Função '{node.func.id}' não está na whitelist.")
        if node.keywords:
            raise ExpressionError("Argumentos nomeados não suportados.")
        args = [_eval_node(a, row) for a in node.args]
        return fn(*args)
    raise ExpressionError(f"Nó AST não permitido: {type(node).__name__}")
