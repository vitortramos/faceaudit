"""Contrato com o forensicface: o que o FFService usa de interno, conferido.

O FFService chama partes internas da biblioteca (`_aligned_face_runner`).
Na 0.8 o `RecognitionRunner.try_compute_batch` (que a 0.7 usava) passou a
devolver só os embeddings, e desmontar isso como par num lote de 2 faces
gravava um número no lugar do embedding, sem erro.
"""

import numpy as np
import pytest

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.services import ff_service as ffs


def _svc_sem_modelos(batch_size=32):
    svc = ffs.FFService.__new__(ffs.FFService)
    svc._use_gpu = False
    svc._batch_size = batch_size
    svc._extended = True
    return svc


class _Runner:
    def __init__(self, resultado):
        self._resultado = resultado

    def try_compute_batch(self, bgr, aligned_keypoints_batch=None):
        return self._resultado(len(bgr)) if callable(self._resultado) else self._resultado


class _Engine:
    def __init__(self, resultado):
        self._runner = _Runner(resultado)

    def _aligned_face_runner(self):
        return self._runner


def _rodar(svc, engine, n):
    erros = []
    emb, fiqa = svc._run_engine_batch(
        engine=engine, model_name="sepaelv2",
        bgr_aligned_faces=np.zeros((n, 112, 112, 3), np.uint8),
        aligned_keypoints_batch=None, error_callback=erros.append,
    )
    return emb, fiqa, erros


def test_par_correto_passa():
    emb = np.random.rand(3, 512).astype(np.float32)
    fiqa = np.array([1.0, 2.0, 3.0])
    e, f, erros = _rodar(_svc_sem_modelos(), _Engine((emb, fiqa)), 3)
    assert erros == [] and np.array_equal(e, emb) and np.array_equal(f, fiqa)


@pytest.mark.parametrize("n", [2, 5])
def test_so_embeddings_vira_erro_e_nunca_lixo(n):
    """O formato da 0.8 no runner errado: (N, 512) sem FIQA. Com N=2, desmontar
    como par 'funcionava' — agora é erro reportado e nada é gravado."""
    emb = np.random.rand(n, 512).astype(np.float32)
    e, f, erros = _rodar(_svc_sem_modelos(), _Engine(emb), n)
    assert e is None and f is None
    assert len(erros) == 1 and "esperado o par" in erros[0]


def test_numero_de_linhas_errado_vira_erro():
    e, f, erros = _rodar(_svc_sem_modelos(), _Engine((np.zeros((2, 512)), None)), 3)
    assert e is None and erros and "forma" in erros[0]


def test_fiqa_com_tamanho_errado_vira_erro():
    e, f, erros = _rodar(
        _svc_sem_modelos(), _Engine((np.zeros((3, 512)), np.zeros(2))), 3,
    )
    assert e is None and erros and "FIQA" in erros[0]


def test_lista_de_um_modelo_e_desembrulhada():
    emb = np.random.rand(2, 512)
    e, _, erros = _rodar(_svc_sem_modelos(), _Engine(([emb], None)), 2)
    assert erros == [] and np.array_equal(e, emb)


def test_padding_e_descartado_depois_da_conferencia():
    """Com o lote completado até batch_size, a conferência vê as N linhas
    preenchidas e o retorno sai com só as reais."""
    svc = _svc_sem_modelos(batch_size=4)
    svc._use_gpu = True  # força o preenchimento
    e, f, erros = _rodar(
        svc, _Engine(lambda n: (np.arange(n * 2.0).reshape(n, 2), np.arange(n * 1.0))), 3,
    )
    assert erros == [] and e.shape == (3, 2) and f.shape == (3,)


@pytest.mark.parametrize("versao, ok", [("0.7.1", False), ("0.8.1", True), ("0.9.0", True), ("1.0", True)])
def test_versao_minima_do_forensicface(monkeypatch, versao, ok):
    monkeypatch.setattr(ffs, "_forensicface_version", lambda: versao)
    if ok:
        ffs._exigir_forensicface_compativel()
    else:
        with pytest.raises(ImportError, match="pip install -e"):
            ffs._exigir_forensicface_compativel()


def test_biblioteca_instalada_tem_o_que_o_ffservice_usa():
    """Pega no CI a troca de versão que remove ou muda o interno usado."""
    pytest.importorskip("forensicface")
    from forensicface.app import ForensicFace
    from forensicface.aligned_processing import AlignedFaceRunner
    assert callable(getattr(ForensicFace, "_aligned_face_runner", None))
    assert callable(getattr(AlignedFaceRunner, "try_compute_batch", None))
    ffs._exigir_forensicface_compativel()


def _tocar(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.mark.parametrize("detector", ["detection/scrfd/det_10g.onnx", "detection/det_10g.onnx"])
def test_modelo_instalado_nos_layouts_compartilhados(tmp_path, detector):
    _tocar(tmp_path / detector)
    _tocar(tmp_path / "recognition" / "sepaelv2" / "adaface_face.onnx")
    assert SepaelModel.SEPAEL_V2.installed_on_disk(tmp_path)
    assert not SepaelModel.SEPAEL_V4.installed_on_disk(tmp_path)


def test_modelo_instalado_no_layout_antigo(tmp_path):
    _tocar(tmp_path / "sepaelv2" / "det_10g.onnx")
    _tocar(tmp_path / "sepaelv2" / "adaface" / "model.onnx")
    assert SepaelModel.SEPAEL_V2.installed_on_disk(tmp_path)


def test_sem_detector_nao_esta_instalado(tmp_path):
    _tocar(tmp_path / "recognition" / "sepaelv2" / "adaface_face.onnx")
    assert not SepaelModel.SEPAEL_V2.installed_on_disk(tmp_path)
