# tests/test_strategy_folder.py
"""Helpers de naming da subpasta hierárquica de estratégia.

A subpasta dentro de `models/`, `scores/`, `results/`, `graphics/` segue
o padrão `<sanitize(strategy_name)>_<strategy_id>`. Esses helpers são
usados tanto pelos templates de `StrategyFileType.build_path` quanto
pelos PNG path resolvers e pelo sweep de `_on_remove_strategy`.
"""

from __future__ import annotations

from faceaudit.models.strategy.strategy_folder import (
    sanitize_name,
    strategy_folder_name,
)


class TestSanitizeName:

    def test_replaces_invalid_chars(self):
        assert sanitize_name("a/b\\c:d") == "a_b_c_d"

    def test_replaces_spaces(self):
        assert sanitize_name("PLDA - EMB V4") == "PLDA_-_EMB_V4"

    def test_empty_returns_unknown(self):
        assert sanitize_name("") == "unknown"
        assert sanitize_name(None) == "unknown"  # type: ignore

    def test_idempotent(self):
        once = sanitize_name("a:b c\\d")
        twice = sanitize_name(once)
        assert once == twice


class TestStrategyFolderName:

    def test_combines_name_and_id(self):
        assert (
            strategy_folder_name("ST_01", "PLDA Cosseno")
            == "PLDA_Cosseno_ST_01"
        )

    def test_sanitizes_both_components(self):
        assert (
            strategy_folder_name("uuid:with/slash", "My/Strat")
            == "My_Strat_uuid_with_slash"
        )

    def test_empty_name_falls_back_to_id_only(self):
        """Sem name (chamada pré-rename ou repositório que só tem id),
        folder é apenas `<strategy_id>` — sem prefix `unknown_`."""
        assert strategy_folder_name("ST_01", "") == "ST_01"
        assert strategy_folder_name("ST_01") == "ST_01"

    def test_brazilian_chars_passthrough(self):
        """Caracteres válidos (acentos, ç) NÃO são sanitizados."""
        folder = strategy_folder_name("ST_01", "Padrões")
        # Apenas a SUBSTITUIÇÃO de chars inválidos — acentos sobrevivem.
        assert folder == "Padrões_ST_01"
