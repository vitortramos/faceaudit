"""Deixa as DLLs do CUDA/cuDNN visíveis para o ONNX Runtime no Windows.

**O problema.** Com `onnxruntime-gpu` instalado e as bibliotecas da NVIDIA
presentes como pacotes pip no ambiente, a criação da sessão falhava assim:

    Error loading "onnxruntime_providers_cuda.dll" which depends on
    "cudnn64_9.dll" which is missing. (Error 126)

E caía para CPU — em silêncio para quem não lesse o terminal. A GPU existia,
o cuDNN existia, e nada funcionava.

**A causa.** A partir do Python 3.8, o CPython chama
`SetDefaultDllDirectories` no Windows, o que **retira o `PATH` da ordem de
busca** das DLLs dependentes. As bibliotecas da NVIDIA instaladas por pip
ficam em `site-packages/nvidia/<componente>/bin/`, que não está em nenhuma
lista que o carregador consulte. Pôr no `PATH` não resolve, justamente pela
mudança do 3.8.

**Por que `os.add_dll_directory` sozinho não basta.** Ele registra o
diretório, mas o ONNX Runtime carrega o provider pelo caminho completo, e a
resolução das dependências *desse* módulo não passa pelos diretórios do
usuário de forma confiável. Medido nesta máquina: com os sete diretórios
registrados, o erro 126 persistia.

**O que funciona.** Pré-carregar as DLLs com `ctypes.WinDLL`, pelo caminho
completo, antes de o ORT pedir o provider. Com o módulo já na memória do
processo, o carregador o reaproveita pelo nome-base ao resolver as
dependências. A ordem importa: runtime do CUDA primeiro, bibliotecas de
matemática depois, cuDNN por último — o cuDNN depende das anteriores.

Nada disso é necessário no Linux (busca por `RPATH`/`LD_LIBRARY_PATH`) nem
no macOS (CoreML, sem CUDA), então a função é um no-op fora do Windows.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Ordem de dependência. O cuDNN é o último porque depende do runtime do CUDA
# e das bibliotecas de álgebra; invertê-la faz o pré-carregamento falhar em
# silêncio e o problema volta.
_ORDEM_COMPONENTES: Tuple[str, ...] = (
    "cuda_runtime",
    "cublas",
    "cufft",
    "curand",
    "nvjitlink",
    "cuda_nvrtc",
    "cusparse",
    "cusolver",
    "cudnn",
)


@dataclass
class CudaBootstrapReport:
    """O que o preparo conseguiu fazer. Para log e diagnóstico."""

    applied: bool = False
    skipped_reason: Optional[str] = None
    dll_directories: List[str] = field(default_factory=list)
    loaded_dlls: int = 0
    failed_dlls: List[str] = field(default_factory=list)

    @property
    def resumo(self) -> str:
        if not self.applied:
            return f"preparo do CUDA não aplicado ({self.skipped_reason})"
        return (
            f"{self.loaded_dlls} DLL(s) do CUDA pré-carregadas de "
            f"{len(self.dll_directories)} diretório(s)"
        )


_relatorio: Optional[CudaBootstrapReport] = None


def ensure_cuda_dlls_loaded(verbose: bool = False) -> CudaBootstrapReport:
    """Torna o CUDA/cuDNN do ambiente visível ao ORT. Idempotente.

    Chamar antes de criar qualquer sessão de inferência com
    `CUDAExecutionProvider`. Chamar mais de uma vez é livre — o resultado da
    primeira vez fica guardado.

    Nunca levanta: ambiente sem GPU, sem os pacotes da NVIDIA ou com DLL
    corrompida apenas não ganha aceleração, e isso não pode impedir o
    programa de abrir.
    """
    global _relatorio
    if _relatorio is not None:
        return _relatorio

    _relatorio = _aplicar(verbose)
    return _relatorio


def reset_for_tests() -> None:
    """Esquece o resultado guardado. Só para teste."""
    global _relatorio
    _relatorio = None


def _aplicar(verbose: bool) -> CudaBootstrapReport:
    if sys.platform != "win32":
        # No Linux a busca é por RPATH/LD_LIBRARY_PATH e no macOS não há
        # CUDA — em nenhum dos dois o problema existe.
        return CudaBootstrapReport(skipped_reason="não é Windows")

    raiz = _raiz_dos_pacotes_nvidia()
    if raiz is None:
        return CudaBootstrapReport(
            skipped_reason="pacotes nvidia-*-cu* não encontrados no ambiente",
        )

    relatorio = CudaBootstrapReport(applied=True)

    # 1. Registra os diretórios. Não basta sozinho (ver o cabeçalho), mas
    #    ajuda nas dependências que o pré-carregamento não cobre.
    for diretorio in sorted(glob.glob(os.path.join(raiz, "*", "bin"))):
        try:
            os.add_dll_directory(diretorio)
            relatorio.dll_directories.append(diretorio)
        except OSError:
            # Diretório sumiu entre o glob e a chamada; segue.
            pass

    # 2. Pré-carrega, na ordem de dependência.
    import ctypes

    for componente in _ORDEM_COMPONENTES:
        padrao = os.path.join(raiz, componente, "bin", "*.dll")
        for caminho in sorted(glob.glob(padrao)):
            try:
                ctypes.WinDLL(caminho)
                relatorio.loaded_dlls += 1
            except OSError:
                # DLL que não carrega isolada é comum (variantes ".alt",
                # módulos que exigem um contexto que ainda não existe) e não
                # impede o conjunto de funcionar.
                relatorio.failed_dlls.append(os.path.basename(caminho))

    if verbose:
        print(f"⚙️  [CUDA] {relatorio.resumo}")
        if relatorio.failed_dlls:
            print(
                f"   (não carregaram isoladas, normalmente inofensivo: "
                f"{', '.join(relatorio.failed_dlls[:5])})"
            )
    return relatorio


def _raiz_dos_pacotes_nvidia() -> Optional[str]:
    """`site-packages/nvidia`, se existir e tiver algum componente.

    Usa `sysconfig` e não `os.__file__`: num ambiente virtual, o segundo
    aponta para a `Lib` do Python base, e a busca erra o alvo silenciosamente
    (foi exatamente o que aconteceu no diagnóstico deste problema).
    """
    import sysconfig

    candidatos = []
    for chave in ("purelib", "platlib"):
        caminho = sysconfig.get_paths().get(chave)
        if caminho:
            candidatos.append(os.path.join(caminho, "nvidia"))
    # Fallback: alguma entrada do sys.path que contenha nvidia/.
    candidatos.extend(
        os.path.join(p, "nvidia") for p in sys.path if p and os.path.isdir(p)
    )

    for raiz in candidatos:
        if os.path.isdir(raiz) and glob.glob(os.path.join(raiz, "*", "bin")):
            return raiz
    return None


def cuda_provider_disponivel() -> bool:
    """True quando o ORT consegue, de fato, criar o provider de CUDA.

    Diferente de `get_available_providers()`, que lista o que foi compilado
    e não o que carrega — foi essa diferença que escondeu o problema: a
    lista mostrava `CUDAExecutionProvider` enquanto toda sessão rodava em
    CPU.
    """
    try:
        import onnxruntime as ort
    except Exception:            # noqa: BLE001
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    ensure_cuda_dlls_loaded()
    try:
        # Grafo mínimo: só para ver se o provider inicializa.
        import numpy as np
        from onnx import TensorProto, helper

        no = helper.make_node("Identity", ["x"], ["y"])
        grafo = helper.make_graph(
            [no], "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        modelo = helper.make_model(grafo)
        sessao = ort.InferenceSession(
            modelo.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return "CUDAExecutionProvider" in sessao.get_providers()
    except Exception:            # noqa: BLE001
        return False
