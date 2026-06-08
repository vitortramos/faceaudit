"""PR 1a — Testes do rename TypicalityModel → MoVMFTypicalityModel.

Cobre:
  1. Import direto do novo módulo + nome de classe correto.
  2. Re-exports do pacote raiz e do subpacote services.
  3. Shim emite DeprecationWarning sem quebrar funcionalidade.
  4. Alias TypicalityModel === MoVMFTypicalityModel (mesma classe).
  5. Pickle round-trip do modelo treinado.
  6. JSON to_dict/from_dict round-trip.
  7. Pickle legado (qualified name antigo) resolve via shim.
  8. Default `model_name` reflete novo nome.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Fixture comum
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_movmf():
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    rng = np.random.default_rng(20260608)
    X = rng.standard_normal((60, 8))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    m = MoVMFTypicalityModel(n_components=1)
    m.fit(X)
    return m, X


# ─────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────

def test_new_import_works():
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    assert MoVMFTypicalityModel.__name__ == "MoVMFTypicalityModel"


def test_root_reexport():
    from faceaudit import MoVMFTypicalityModel
    from faceaudit.services.movmf_typicality_model import (
        MoVMFTypicalityModel as M2,
    )
    assert MoVMFTypicalityModel is M2


def test_services_reexport():
    from faceaudit.services import MoVMFTypicalityModel
    from faceaudit.services.movmf_typicality_model import (
        MoVMFTypicalityModel as M2,
    )
    assert MoVMFTypicalityModel is M2


def test_root_legacy_alias():
    from faceaudit import TypicalityModel, MoVMFTypicalityModel
    assert TypicalityModel is MoVMFTypicalityModel


def test_services_legacy_alias():
    from faceaudit.services import TypicalityModel, MoVMFTypicalityModel
    assert TypicalityModel is MoVMFTypicalityModel


# ─────────────────────────────────────────────────────────────────────
# Shim deprecation
# ─────────────────────────────────────────────────────────────────────

def test_shim_emits_deprecation_warning():
    # Força re-import do shim
    import importlib
    import faceaudit.services.typicality_model as shim_mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(shim_mod)
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)
                and "movmf_typicality_model" in str(w.message)]
        assert deps, "Shim deveria emitir DeprecationWarning"


def test_shim_class_alias_resolves():
    from faceaudit.services.typicality_model import TypicalityModel as ShimAlias
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    assert ShimAlias is MoVMFTypicalityModel


def test_shim_typicality_info_reexport():
    from faceaudit.services.typicality_model import TypicalityInfo
    assert TypicalityInfo.__name__ == "TypicalityInfo"


# ─────────────────────────────────────────────────────────────────────
# Default model_name
# ─────────────────────────────────────────────────────────────────────

def test_default_model_name_reflects_new_class():
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    m = MoVMFTypicalityModel()
    assert m.model_name == "MoVMFTypicalityModel"


# ─────────────────────────────────────────────────────────────────────
# Round-trips
# ─────────────────────────────────────────────────────────────────────

def test_pickle_round_trip(trained_movmf):
    m, X = trained_movmf
    buf = pickle.dumps(m)
    m2 = pickle.loads(buf)
    np.testing.assert_allclose(m.score_samples(X), m2.score_samples(X))


def test_json_round_trip(trained_movmf):
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    m, X = trained_movmf
    d = m.to_dict()
    m2 = MoVMFTypicalityModel.from_dict(d)
    np.testing.assert_allclose(m.score_samples(X), m2.score_samples(X))


# ─────────────────────────────────────────────────────────────────────
# Legacy pickle resolution via shim
# ─────────────────────────────────────────────────────────────────────

def test_legacy_pickle_with_old_qualified_name_resolves():
    """Pickle gerado antes do rename apontava para
    `faceaudit.services.typicality_model.TypicalityModel`. Após rename,
    esse path resolve para `MoVMFTypicalityModel` via shim.
    """
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel

    # Pickle handcrafted que aponta para o nome antigo via STACK_GLOBAL
    crafted = (
        b"\x80\x04"
        b"\x95"
        + (b"\x00" * 8)
        + b"\x8c"
        + bytes([len("faceaudit.services.typicality_model")])
        + b"faceaudit.services.typicality_model"
        + b"\x94"
        + b"\x8c"
        + bytes([len("TypicalityModel")])
        + b"TypicalityModel"
        + b"\x94"
        + b"\x93\x94"
        + b"\x29"
        + b"\x81\x94"
        + b"."
    )
    obj = pickle.loads(crafted)
    assert isinstance(obj, MoVMFTypicalityModel)
