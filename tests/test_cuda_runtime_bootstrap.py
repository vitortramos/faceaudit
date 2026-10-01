"""DLLs do CUDA visíveis ao ONNX Runtime no Windows.

O defeito que isto corrige era silencioso e caro: com `onnxruntime-gpu`
instalado e a GPU presente, TODA sessão de inferência rodava em CPU. Medido
nesta máquina pelo caminho real do `FFService`: 32,7 ms/imagem em CPU contra
8,4 ms/imagem em GPU — 3,9× de diferença paga sem ninguém saber.

A causa não é falta de biblioteca: o Python 3.8+ chama
`SetDefaultDllDirectories` no Windows, o que **retira o `PATH` da ordem de
busca** das DLLs dependentes, e os pacotes pip da NVIDIA ficam em
`site-packages/nvidia/*/bin`, fora de qualquer lista que o carregador
consulte. Registrar os diretórios não bastou (medido: erro 126 persistia);
o que resolve é pré-carregar as DLLs antes de o ORT pedir o provider.

O que os testes travam:

- **a ordem dos componentes**, com o cuDNN por último. Ele depende do
  runtime do CUDA e das bibliotecas de álgebra; inverter faz o
  pré-carregamento falhar em silêncio e o problema volta sem aviso;
- **idempotência**, porque o preparo é chamado do bootstrap da interface E
  do FFService;
- **nunca levantar**: ambiente sem GPU apenas não ganha aceleração, e isso
  não pode impedir o programa de abrir;
- **`sysconfig` e não `os.__file__`** para achar o `site-packages` — num
  ambiente virtual o segundo aponta para a `Lib` do Python base, e foi
  exatamente esse erro que escondeu o diagnóstico por um bom tempo.
"""

from __future__ import annotations

import sys

import pytest

from faceaudit.services import cuda_runtime_bootstrap as boot


@pytest.fixture(autouse=True)
def _limpar():
    boot.reset_for_tests()
    yield
    boot.reset_for_tests()


# ── Ordem de dependência ─────────────────────────────────────────────────

def test_cudnn_e_o_ultimo_componente():
    """Ele depende dos anteriores. Carregar antes falha em silêncio."""
    assert boot._ORDEM_COMPONENTES[-1] == "cudnn"


def test_runtime_do_cuda_vem_primeiro():
    assert boot._ORDEM_COMPONENTES[0] == "cuda_runtime"


def test_ordem_cobre_os_componentes_que_o_ort_exige():
    """`onnxruntime_providers_cuda.dll` depende destes."""
    for componente in ("cuda_runtime", "cublas", "cufft", "curand", "cudnn"):
        assert componente in boot._ORDEM_COMPONENTES, componente


def test_sem_componentes_repetidos():
    assert len(boot._ORDEM_COMPONENTES) == len(set(boot._ORDEM_COMPONENTES))


# ── Idempotência ─────────────────────────────────────────────────────────

def test_segunda_chamada_devolve_o_mesmo_relatorio():
    """Chamado do bootstrap da interface E do FFService — recarregar 20
    DLLs a cada construção de serviço seria desperdício puro."""
    primeiro = boot.ensure_cuda_dlls_loaded()
    segundo = boot.ensure_cuda_dlls_loaded()

    assert primeiro is segundo


def test_reset_permite_reaplicar():
    boot.ensure_cuda_dlls_loaded()
    boot.reset_for_tests()

    assert boot._relatorio is None


# ── Fora do Windows ──────────────────────────────────────────────────────

def test_fora_do_windows_e_no_op(monkeypatch):
    """No Linux a busca é por RPATH; no macOS não há CUDA."""
    monkeypatch.setattr(sys, "platform", "linux")
    boot.reset_for_tests()

    r = boot.ensure_cuda_dlls_loaded()

    assert r.applied is False
    assert "Windows" in r.skipped_reason


# ── Nunca impedir o programa de abrir ────────────────────────────────────

def test_sem_nenhuma_fonte_apenas_nao_acelera(monkeypatch):
    """Precisa zerar as DUAS fontes: sem os pacotes pip o preparo cai para a
    instalação do sistema, então só a ausência de ambas significa "sem
    aceleração"."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: None)
    monkeypatch.setattr(boot, "_PADROES_DO_SISTEMA", ())
    boot.reset_for_tests()

    r = boot.ensure_cuda_dlls_loaded()

    assert r.applied is False
    assert "CUDA não encontrado" in r.skipped_reason
    assert r.loaded_dlls == 0


def test_dll_que_nao_carrega_nao_interrompe(monkeypatch, tmp_path):
    """DLL corrompida ou que exige contexto ausente é caso real; o conjunto
    continua funcionando sem ela."""
    raiz = tmp_path / "nvidia"
    for comp in ("cuda_runtime", "cudnn"):
        d = raiz / comp / "bin"
        d.mkdir(parents=True)
        (d / f"{comp}_falsa.dll").write_bytes(b"nao e uma dll")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: str(raiz))
    boot.reset_for_tests()

    r = boot.ensure_cuda_dlls_loaded()      # não levanta

    assert r.applied is True
    assert r.loaded_dlls == 0
    assert len(r.failed_dlls) == 2


def test_relatorio_tem_resumo_legivel(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    boot.reset_for_tests()

    assert "não aplicado" in boot.ensure_cuda_dlls_loaded().resumo


# ── A busca do site-packages ─────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="específico do Windows")
def test_acha_a_raiz_no_ambiente_virtual():
    """`os.__file__` num venv aponta para a `Lib` do Python BASE — foi esse
    erro que fez a busca falhar silenciosamente no diagnóstico."""
    import os
    import sysconfig

    raiz = boot._raiz_dos_pacotes_nvidia()
    if raiz is None:
        pytest.skip("ambiente sem os pacotes nvidia-*-cu*")

    esperado = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    assert os.path.normcase(raiz) == os.path.normcase(esperado)
    assert os.path.isdir(raiz)


@pytest.mark.skipif(sys.platform != "win32", reason="específico do Windows")
def test_nesta_maquina_o_provider_de_cuda_funciona():
    """Teste de ambiente, não de código: confirma que o preparo entrega o
    que prometeu. Pula onde não há GPU, em vez de falhar."""
    try:
        import onnxruntime as ort
    except Exception:
        pytest.skip("onnxruntime ausente")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime sem suporte a CUDA compilado")

    assert boot.cuda_provider_disponivel() is True, (
        "o provider de CUDA deveria inicializar depois do preparo — se este "
        "teste falhar nesta máquina, a aceleração voltou a estar quebrada"
    )


def test_disponibilidade_nao_confia_na_lista_do_ort():
    """`get_available_providers()` lista o que foi COMPILADO, não o que
    carrega. Foi essa diferença que escondeu o problema: a lista mostrava
    CUDA enquanto toda sessão rodava em CPU."""
    import inspect

    fonte = inspect.getsource(boot.cuda_provider_disponivel)
    assert "InferenceSession" in fonte, (
        "a checagem tem de CRIAR uma sessão, não só ler a lista"
    )


# ── Costura ──────────────────────────────────────────────────────────────

def test_ffservice_prepara_antes_de_criar_sessao():
    import inspect

    from faceaudit.services.ff_service import FFService

    fonte = inspect.getsource(FFService.__init__)
    assert "ensure_cuda_dlls_loaded" in fonte
    # Antes do patch do CoreML, que é onde as sessões começam a nascer.
    assert fonte.index("ensure_cuda_dlls_loaded") < fonte.index(
        "_patch_onnxruntime_for_coreml_static_shapes(",
    )


def test_bootstrap_da_interface_tambem_prepara():
    """Qualquer caminho que crie sessão sofre o mesmo — OFIQ, SER-FIQ, e
    bibliotecas de terceiros que não passam pelo FFService."""
    from pathlib import Path

    fonte = Path("interface/run.py").read_text(encoding="utf-8")
    assert "ensure_cuda_dlls_loaded" in fonte


# ── Escala do CR-FIQA no catálogo ────────────────────────────────────────

def test_escala_do_fiqa_cobre_os_valores_reais():
    """Estava (0, 100), faixa que o CR-FIQA nunca produz: medido em 599
    faces reais, vai de 1,46 a 2,56. Declaração errada não dava dano hoje
    (nada consome `scale`), mas qualquer normalização futura mostraria todas
    as faces comprimidas em 2% da barra."""
    from faceaudit.models.column_catalog import DatasetType, get_default_specs

    spec = get_default_specs(DatasetType.FACE)["fiqa_score"]

    minimo, maximo = spec.scale
    assert minimo == 0.0
    assert 2.6 <= maximo <= 5.0, (
        f"o teto {maximo} tem de cobrir os valores observados (até ~2,6) sem "
        f"exagerar — (0, 100) comprimia tudo no começo da escala"
    )


# ── As duas fontes de CUDA ───────────────────────────────────────────────

def test_prefere_os_pacotes_pip(monkeypatch, tmp_path):
    """O ambiente virtual tem um conjunto coerente entre si; o sistema pode
    ter outro. Misturar deixaria a combinação de versões à mercê de quem
    chega primeiro na memória."""
    raiz = tmp_path / "nvidia"
    (raiz / "cudnn" / "bin").mkdir(parents=True)
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: str(raiz))

    origem, dirs = boot._escolher_fonte()

    assert origem == "pacotes pip"
    assert len(dirs) == 1


def test_cai_para_o_sistema_sem_os_pacotes_pip(monkeypatch):
    """Máquina que instalou o toolkit do CUDA e não tem os pacotes pip —
    caminho verificado nesta máquina: isolado, faz o provider carregar."""
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: None)
    monkeypatch.setattr(
        boot, "_PADROES_DO_SISTEMA", (r"C:\Program Files\NVIDIA*\**\bin",),
    )

    origem, _ = boot._escolher_fonte()

    assert origem in (None, "instalação do sistema")


def test_nenhuma_fonte_e_motivo_explicito(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: None)
    monkeypatch.setattr(boot, "_PADROES_DO_SISTEMA", ())
    boot.reset_for_tests()

    r = boot.ensure_cuda_dlls_loaded()

    assert r.applied is False
    assert "nem" in r.skipped_reason, (
        "o motivo tem de dizer que as DUAS fontes foram procuradas"
    )


def test_padroes_do_sistema_so_pegam_cuda_12():
    """onnxruntime-gpu atual exige CUDA 12.x. Carregar as DLLs de um toolkit
    11 instalado ao lado levaria ao erro oposto do que corrigimos."""
    toolkit = [p for p in boot._PADROES_DO_SISTEMA if "Toolkit" in p]
    assert toolkit and all("v12" in p for p in toolkit)


def test_cudnn_por_ultimo_mesmo_vindo_de_um_diretorio_unico(tmp_path):
    """Na instalação do sistema tudo vem de dois diretórios, não de um por
    componente — a ordenação tem de funcionar pelo nome do arquivo."""
    d = tmp_path / "bin"
    d.mkdir()
    for nome in ("cudnn64_9.dll", "cublas64_12.dll", "cudart64_12.dll"):
        (d / nome).write_bytes(b"")

    ordem = [p.lower() for p in boot._dlls_em_ordem([str(d)])]

    assert "cudnn" in ordem[-1], f"o cuDNN tem de ser o último: {ordem}"


def test_origem_aparece_no_resumo(monkeypatch, tmp_path):
    raiz = tmp_path / "nvidia"
    (raiz / "cudnn" / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(boot, "_raiz_dos_pacotes_nvidia", lambda: str(raiz))
    boot.reset_for_tests()

    r = boot.ensure_cuda_dlls_loaded()

    assert r.source == "pacotes pip"
    assert "pacotes pip" in r.resumo, (
        "saber de QUAL fonte veio é a primeira coisa a olhar quando funciona "
        "numa máquina e não noutra"
    )
