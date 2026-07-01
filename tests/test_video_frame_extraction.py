"""Testes da Fase 1 — produtor mínimo de extração de faces de vídeo.

Usa um FFService/engine FAKE (sem GPU/modelos ForensicFace) e um vídeo
sintético, de modo que o laço, os gates de qualidade e a proveniência
sejam exercitados de forma determinística.

Truque: o engine fake codifica um `face_id` no pixel [0,0] da
`aligned_face`; o FFService fake decodifica esse id de volta no batch
para devolver o embedding/fiqa que o teste escolheu — controlando a
similaridade com a âncora sem rodar reconhecimento real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.models.ff_info import FFInfo
from faceaudit.models.video.video_extraction_config import VideoExtractionConfig
from faceaudit.models.video.video_target import VideoTarget
from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.enums.quality_gate_policy import QualityGatePolicy
from faceaudit.services.image_io import imwrite_unicode_safe, imread_unicode_safe
from faceaudit.services.hash_service import HashService
from faceaudit.services import video_frame_face_extraction_service as svc_mod
from faceaudit.services.video_frame_face_extraction_service import (
    VideoFrameFaceExtractionService,
    _gate_decision,
    _pose_to_dict,
)

DIM = 8
EMB_TARGET = np.zeros(DIM, dtype=np.float32); EMB_TARGET[0] = 1.0
EMB_OTHER = np.zeros(DIM, dtype=np.float32); EMB_OTHER[1] = 1.0


# ───────────────────────────── fakes ─────────────────────────────

class _FakeBackend:
    def estimate_norm(self, kps):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


class _FakeEngine:
    """detect_and_align scriptável por chamada."""
    def __init__(self, script):
        self.backend = _FakeBackend()
        self._script = script
        self._call = 0

    def detect_and_align(self, frame, single_face=False):
        faces = self._script(self._call)
        self._call += 1
        return faces


def _face(face_id, bbox, det_score, pose=(0.0, 0.0, 0.0)):
    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    aligned[0, 0, :] = face_id  # id em todos os canais sobrevive ao swap RGB↔BGR
    return {
        "aligned_face": aligned,
        "aligned_keypoints": np.zeros((5, 2), dtype=float),
        "bbox": np.asarray(bbox, dtype=float),
        "keypoints": np.zeros((5, 2), dtype=float),
        "det_score": float(det_score),
        "pose": pose,   # ForensicFace detect_and_align: cru = [pitch, yaw, roll]
        "gender": "M",
        "age": 30,
    }


class _FakeFFService:
    def __init__(self, engine, registry):
        self.app = engine
        self.default_model = "sepaelv2"
        self._registry = registry  # face_id -> (embedding, fiqa)

    def extract_embeddings_from_aligned_faces_batch(self, bgr_aligned_faces, *, models=None, aligned_keypoints_batch=None, error_callback=None):
        out = []
        for k in range(bgr_aligned_faces.shape[0]):
            fid = int(bgr_aligned_faces[k, 0, 0, 0])
            emb, fiqa = self._registry[fid]
            out.append(FFInfo(fiqa_score=fiqa, embedding_sepaelv2=np.asarray(emb, dtype=np.float32)))
        return out

    def extract_embeddings_from_aligned_face(self, bgr, *, aligned_keypoints=None, model=None):
        fid = int(bgr[0, 0, 0])
        emb, fiqa = self._registry[fid]
        return FFInfo(fiqa_score=fiqa, embedding_sepaelv2=np.asarray(emb, dtype=np.float32))


def _make_video(path: Path, n=5, size=(64, 64), fps=5) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not vw.isOpened():
        return False
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), (i * 10) % 256, np.uint8))
    vw.release()
    return path.exists() and path.stat().st_size > 0


@pytest.fixture
def video_path(tmp_path):
    p = tmp_path / "clip.avi"
    if not _make_video(p):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    return p


def _target(threshold=0.5):
    return VideoTarget(label="alvo", anchor_embeddings=[EMB_TARGET.copy()],
                       similarity_threshold=threshold, recognition_model="sepaelv2")


def _cfg(video_path, tmp_path, **kw):
    base = dict(video_path=str(video_path), output_root=str(tmp_path / "out"),
                every_n_frames=1, similarity_threshold=0.5)
    base.update(kw)
    return VideoExtractionConfig(**base)


def _service(script, registry):
    return VideoFrameFaceExtractionService(_FakeFFService(_FakeEngine(script), registry))


def _read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


# ───────────────────────── testes puros ─────────────────────────

def test_gate_decision_matrix():
    P = QualityGatePolicy
    assert _gate_decision(P.DISABLED, 0.1, 0.5) == "disabled"
    assert _gate_decision(P.TAG_ONLY, 0.7, 0.5) == "passed"
    assert _gate_decision(P.TAG_ONLY, 0.3, 0.5) == "tag_low"
    assert _gate_decision(P.ROUTE_TO_REVIEW, 0.3, 0.5) == "route_to_review"
    assert _gate_decision(P.HARD_REJECT, 0.3, 0.5) == "hard_reject"
    # política ativa mas sem limiar → nunca derruba
    assert _gate_decision(P.HARD_REJECT, 0.3, None) == "passed"


def test_detection_containing_point_picks_clicked_not_largest():
    big = _face(0, [0, 0, 100, 100], 0.9)
    small = _face(1, [40, 40, 60, 60], 0.9)
    dets = [big, small]
    # clique dentro da face pequena (também dentro da grande) → escolhe a menor
    idx = VideoFrameFaceExtractionService.detection_containing_point(dets, 50, 50)
    assert idx == 1
    # clique só dentro da grande
    assert VideoFrameFaceExtractionService.detection_containing_point(dets, 10, 10) == 0
    # clique fora de tudo
    assert VideoFrameFaceExtractionService.detection_containing_point(dets, 200, 200) is None


def test_imwrite_unicode_safe_accented_path_roundtrip(tmp_path):
    p = tmp_path / "Política Federal" / "alvo_f0000001.png"
    img = (np.random.default_rng(0).integers(0, 255, (32, 32, 3))).astype(np.uint8)
    assert imwrite_unicode_safe(p, img) is True
    assert p.exists()
    back = imread_unicode_safe(p)
    assert back is not None and back.shape == (32, 32, 3)
    assert HashService().calculate_file_hash(p) is not None


def test_face_provenance_sidecar_and_scalar_projection():
    fp = FaceProvenance(run_id="r", run_manifest="_run_r.json", identity_label="alvo",
                        image_file="alvo_f0000001.png", det_score=0.9, fiqa_score=0.4,
                        similarity_to_anchor=0.7, frame_native_index=1, source_video_sha256="abc",
                        det_score_decision="passed", fiqa_score_decision="route_to_review",
                        det_score_policy="TAG_ONLY", fiqa_score_policy="ROUTE_TO_REVIEW")
    side = fp.to_sidecar_dict()
    assert side["detection"]["det_score"] == 0.9
    assert side["quality_gates"]["fiqa_score_decision"] == "route_to_review"
    scal = fp.scalar_projection().to_extra_data()
    # namespaced, não colide com det_score/fiqa_score canônicos
    assert "video_extraction_det_score" in scal and "det_score" not in scal
    assert scal["video_extraction_fiqa_score"] == 0.4
    assert scal["similarity_to_anchor"] == 0.7


# ───────────────────── testes de extract (laço) ─────────────────────

def test_extract_no_detection_writes_frame_log(video_path, tmp_path):
    service = _service(lambda call: [], {})
    res = service.extract(_cfg(video_path, tmp_path), [_target()])
    assert res.n_accepted_auto == 0
    frames = _read_jsonl(res.frames_path)
    assert len(frames) >= 1
    assert all(f["status"] == "no_detection" for f in frames)
    # nenhuma pasta de identidade criada
    assert not (Path(res.run_dir) / "alvo").exists()


def test_extract_accepts_target_and_writes_crop_and_sidecar(video_path, tmp_path):
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    res = service.extract(_cfg(video_path, tmp_path), [_target(threshold=0.5)])
    assert res.n_accepted_auto >= 1
    crops = list((Path(res.run_dir) / "alvo").glob("*.png"))
    assert crops, "deveria ter salvo recortes"
    sidecars = list((Path(res.run_dir) / "_meta").glob("alvo_v*_f*.json"))
    assert sidecars
    side = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert side["reidentification"]["similarity_to_anchor"] == pytest.approx(1.0, abs=1e-5)
    assert side["human_review"]["status"] == "pending"
    assert side["artifact"]["image_sha256"]
    # Cadeia de custódia: a extração mantém o SHA-256 do vídeo de origem mesmo
    # depois de a prévia da UI passar a carregar com compute_hash=False.
    assert len(side["source_video"]["sha256"]) == 64
    # manifesto com status de ciclo de vida
    manifest = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "completed"
    assert len(manifest["source_video"]["sha256"]) == 64   # custódia no manifesto
    assert manifest["review_status"] == "not_started"
    assert manifest["dataset_status"] == "not_imported"


def test_extract_below_identity_threshold_writes_no_crop(video_path, tmp_path):
    reg = {0: (EMB_OTHER, 0.8)}  # ortogonal à âncora → sim ~0
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    res = service.extract(_cfg(video_path, tmp_path), [_target(threshold=0.5)])
    assert res.n_accepted_auto == 0
    assert not (Path(res.run_dir) / "alvo").exists()
    frames = _read_jsonl(res.frames_path)
    assert all(f["status"] in ("no_match", "no_detection") for f in frames)


def test_extract_det_hard_reject_pre_embedding(video_path, tmp_path):
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.40)], reg)
    cfg = _cfg(video_path, tmp_path, det_score_policy=QualityGatePolicy.HARD_REJECT, min_det_score=0.6)
    res = service.extract(cfg, [_target(threshold=0.5)])
    assert res.n_accepted_auto == 0
    decisions = _read_jsonl(res.decisions_path)
    assert any(d.get("decision") == "hard_reject_det_pre_embedding" for d in decisions)
    assert not (Path(res.run_dir) / "alvo").exists()


def test_extract_fiqa_route_to_review_flags_reason(video_path, tmp_path):
    reg = {0: (EMB_TARGET, 0.30)}  # fiqa baixo
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    cfg = _cfg(video_path, tmp_path, fiqa_score_policy=QualityGatePolicy.ROUTE_TO_REVIEW, min_fiqa_score=0.5)
    res = service.extract(cfg, [_target(threshold=0.5)])
    assert res.n_accepted_auto >= 1
    sidecars = list((Path(res.run_dir) / "_meta").glob("alvo_v*_f*.json"))
    side = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert "low_fiqa" in side["review_reasons"]
    assert side["quality_gates"]["fiqa_score_decision"] == "route_to_review"


def test_extract_reproducible_crop_names_and_hashes(video_path, tmp_path):
    reg = {0: (EMB_TARGET, 0.8)}
    script = lambda call: [_face(0, [10, 10, 40, 40], 0.95)]
    fixed_clock = lambda: "2026-01-01T00:00:00Z"
    r1 = _service(script, reg).extract(_cfg(video_path, tmp_path / "a"), [_target()], now_utc=fixed_clock)
    r2 = _service(script, reg).extract(_cfg(video_path, tmp_path / "b"), [_target()], now_utc=fixed_clock)
    def crop_hashes(res):
        h = HashService()
        return {p.name: h.calculate_file_hash(p) for p in sorted((Path(res.run_dir) / "alvo").glob("*.png"))}
    assert crop_hashes(r1) == crop_hashes(r2)


def test_build_target_from_click_records_anchor_frame_and_bbox():
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.9)], reg)
    frame = np.zeros((64, 64, 3), np.uint8)
    built = service.build_target_from_click(frame, 25, 25, label="alvo",
                                            similarity_threshold=0.5, frame_index=7)
    assert built is not None
    target, idx, _dets = built
    assert target.anchor_frame_index == 7
    assert target.anchor_bbox == [10.0, 10.0, 40.0, 40.0]


def test_extract_multi_target_assigns_each_to_own_folder(video_path, tmp_path):
    # Fase 4: pipeline N-ready — 2 alvos, 2 rostos por quadro → 1-pra-1.
    reg = {0: (EMB_TARGET, 0.8), 1: (EMB_OTHER, 0.8)}
    script = lambda call: [_face(0, [5, 5, 25, 25], 0.95), _face(1, [35, 35, 60, 60], 0.95)]
    service = _service(script, reg)
    t_a = VideoTarget(label="alvo_a", anchor_embeddings=[EMB_TARGET.copy()],
                      similarity_threshold=0.5, recognition_model="sepaelv2")
    t_b = VideoTarget(label="alvo_b", anchor_embeddings=[EMB_OTHER.copy()],
                      similarity_threshold=0.5, recognition_model="sepaelv2")
    res = service.extract(_cfg(video_path, tmp_path), [t_a, t_b])
    assert res.n_accepted_auto >= 2
    assert list((Path(res.run_dir) / "alvo_a").glob("*.png"))
    assert list((Path(res.run_dir) / "alvo_b").glob("*.png"))
    # cada alvo recebe a face do seu próprio embedding (sem cruzamento)
    side_a = json.loads(sorted((Path(res.run_dir) / "_meta").glob("alvo_a_v*_f*.json"))[0].read_text(encoding="utf-8"))
    assert side_a["reidentification"]["similarity_to_anchor"] == pytest.approx(1.0, abs=1e-5)


def test_extract_multivideo_combines_into_one_run(tmp_path):
    """Fase 3 — 2 vídeos numa run só: nomes namespaced (sem colisão), manifesto
    com source_videos[], cada face marcada pela sua origem, run nomeada pelo caso."""
    v1, v2 = tmp_path / "camA.avi", tmp_path / "camB.avi"
    if not (_make_video(v1) and _make_video(v2)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    cfg = _cfg(v1, tmp_path, video_paths=[str(v1), str(v2)], case_name="caso_x")
    res = service.extract(cfg, [_target(threshold=0.5)])

    assert res.n_accepted_auto >= 2
    crops = sorted(p.name for p in (Path(res.run_dir) / "alvo").glob("*.png"))
    assert any(n.startswith("alvo_v00_") for n in crops)     # vídeo 0
    assert any(n.startswith("alvo_v01_") for n in crops)     # vídeo 1
    assert len(crops) == len(set(crops))                     # sem colisão de nome

    manifest = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
    assert len(manifest["source_videos"]) == 2
    assert manifest["summary"]["n_videos"] == 2

    sidecars = [json.loads(p.read_text(encoding="utf-8"))
                for p in (Path(res.run_dir) / "_meta").glob("alvo_v*_f*.json")]
    origins = {s["source_video"]["filename"] for s in sidecars}
    assert origins == {"camA.avi", "camB.avi"}               # faces marcadas por origem
    assert Path(res.run_dir).name.startswith("caso_x_")      # run nomeada pelo caso


def test_extract_multivideo_cancel_mid_second_video_keeps_partial(tmp_path):
    """Cancelar no 2º vídeo deixa o run parcial auditável (1º vídeo extraído)."""
    v1, v2 = tmp_path / "camA.avi", tmp_path / "camB.avi"
    if not (_make_video(v1) and _make_video(v2)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    cfg = _cfg(v1, tmp_path, video_paths=[str(v1), str(v2)], case_name="caso_y")
    # cancela assim que o 1º vídeo terminar (≥1 face já aceita)
    state = {"calls": 0}
    def cancel():
        state["calls"] += 1
        return state["calls"] > 6        # deixa o 1º vídeo (5 quadros) completar
    res = service.extract(cfg, [_target(threshold=0.5)], cancel_check=cancel)
    assert res.extraction_status == "cancelled"
    manifest = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "cancelled"
    assert res.n_accepted_auto >= 1      # parcial preservado


def test_extract_multivideo_failure_still_writes_manifest(tmp_path):
    """Falha DURANTE o 2º vídeo: o run parcial continua auditável (manifesto
    gravado com extraction_status='failed') antes do erro propagar."""
    v1, v2 = tmp_path / "camA.avi", tmp_path / "camB.avi"
    if not (_make_video(v1) and _make_video(v2)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    # falha determinística no 2º vídeo (independe da contagem exata de quadros)
    orig = service._process_frame
    def boom(*a, **k):
        if k.get("video_index") == 1:
            raise RuntimeError("falha simulada no vídeo 2")
        return orig(*a, **k)
    service._process_frame = boom
    cfg = _cfg(v1, tmp_path, video_paths=[str(v1), str(v2)], case_name="caso_err")
    with pytest.raises(RuntimeError):
        service.extract(cfg, [_target(threshold=0.5)])
    manifests = list((tmp_path / "out").glob("*/_meta/_run_*.json"))
    assert manifests, "run parcial deve ter manifesto mesmo após falha no laço"
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "failed"
    assert manifest["summary"]["n_accepted_auto"] >= 1   # vídeo 1 preservado


def test_extract_multivideo_per_video_window(tmp_path):
    """Janela POR vídeo: o vídeo 0 é limitado a um trecho, o vídeo 1 é inteiro."""
    v1, v2 = tmp_path / "camA.avi", tmp_path / "camB.avi"
    if not (_make_video(v1, n=10, fps=5) and _make_video(v2, n=10, fps=5)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    # vídeo 0: trecho [0.0s, 0.4s] (≈ 3 quadros a 5 fps); vídeo 1: inteiro
    cfg = _cfg(v1, tmp_path, video_paths=[str(v1), str(v2)],
               video_windows=[[0.0, 0.4], None], case_name="caso_win")
    res = service.extract(cfg, [_target(threshold=0.5)])
    v0 = list((Path(res.run_dir) / "alvo").glob("alvo_v00_*.png"))
    v1c = list((Path(res.run_dir) / "alvo").glob("alvo_v01_*.png"))
    assert 1 <= len(v0) < len(v1c)   # a janela reduziu o vídeo 0


def test_import_run_image_kind_crop_vs_aligned(video_path, tmp_path):
    """Tipo de imagem: 'aligned' usa a face 112×112 (is_aligned=True); 'crop'
    usa o recorte com margem (is_aligned=False)."""
    from faceaudit.services.video_review_service import VideoReviewService
    from faceaudit.services.video_run_importer import VideoRunImporter
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(lambda call: [_face(0, [10, 10, 40, 40], 0.95)], reg)
    res = service.extract(_cfg(video_path, tmp_path), [_target(threshold=0.5)])
    # a extração deixa pending; o importador só pega 'accepted'
    rvs = VideoReviewService()
    for f in rvs.list_faces(res.run_dir):
        rvs.set_face_decision(res.run_dir, f.image_file, status="accepted", operator="t")

    imp = VideoRunImporter()
    ds_aligned = imp.import_run(res.run_dir, image_kind="aligned")
    ds_crop = imp.import_run(res.run_dir, image_kind="crop")
    assert ds_aligned.is_aligned is True
    assert ds_crop.is_aligned is False
    assert all("_aligned/" in f.image_path for f in ds_aligned.faces)
    assert all("_aligned/" not in f.image_path for f in ds_crop.faces)
    # Custódia preservada nos DOIS tipos de imagem (crop = imagem de evidência):
    # cada face rastreia o vídeo de origem por hash + origem rotulada.
    for f in (*ds_aligned.faces, *ds_crop.faces):
        assert f.extra_data.get("source_video_sha256")
        assert f.origin.startswith("Vídeo:")


# ─────────── prévia do scrubber: hash adiado + capture cacheado (perf) ───────────

def test_probe_video_skips_hash_when_requested(video_path):
    """A prévia da UI carrega com compute_hash=False (rápido): sem hash do
    arquivo inteiro. A extração mantém o hash para a cadeia de custódia."""
    service = _service(lambda call: [], {})
    fast = service.probe_video(video_path, compute_hash=False)
    full = service.probe_video(video_path)  # default = com hash
    assert fast.sha256 is None
    assert full.sha256 and len(full.sha256) == 64
    # demais metadados continuam disponíveis no carregamento rápido
    assert fast.frame_count_reported == full.frame_count_reported
    assert fast.fps_reported == full.fps_reported


def test_read_frame_reuses_cached_capture(video_path):
    """read_frame reusa UM capture aberto por caminho (não reabre o vídeo a
    cada quadro) — é o que destrava o arraste do slider em vídeos longos."""
    service = _service(lambda call: [], {})
    assert getattr(service, "_pv_cap", None) is None
    f0 = service.read_frame(video_path, 0)
    f1 = service.read_frame(video_path, 1)
    assert f0 is not None and f1 is not None
    cap_after = service._pv_cap
    assert cap_after is not None
    assert service._pv_cap_path == str(video_path)
    # uma segunda leitura não recria o capture
    service.read_frame(video_path, 2)
    assert service._pv_cap is cap_after
    # e fechar a prévia libera o handle
    service.close_preview_cap()
    assert service._pv_cap is None and service._pv_cap_path is None


def test_preview_capture_switches_on_new_path(tmp_path):
    """Trocar de vídeo recria o capture cacheado para o novo caminho."""
    a, b = tmp_path / "a.avi", tmp_path / "b.avi"
    if not (_make_video(a) and _make_video(b)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    service = _service(lambda call: [], {})
    service.read_frame(a, 0)
    cap_a = service._pv_cap
    service.read_frame(b, 0)
    assert service._pv_cap is not cap_a
    assert service._pv_cap_path == str(b)
    service.close_preview_cap()


# ─────────── pose: ordem yaw/pitch/roll (ForensicFace cru = [pitch, yaw, roll]) ───────────

def test_pose_to_dict_reorders_pitch_yaw():
    # detect_and_align devolve [pitch, yaw, roll]; _pose_to_dict deve mapear
    # para {yaw, pitch, roll} — NÃO ler pose[0] como yaw.
    assert _pose_to_dict([10.0, 20.0, 30.0]) == {"yaw": 20.0, "pitch": 10.0, "roll": 30.0}
    assert _pose_to_dict(None) is None
    assert _pose_to_dict("lixo") is None
    assert _pose_to_dict([1.0, 2.0]) is None          # tamanho errado → None


def test_extract_sidecar_pose_order(video_path, tmp_path):
    # End-to-end: face com pose crua [pitch=10, yaw=20, roll=30] → o sidecar
    # deve gravar yaw=20, pitch=10, roll=30 (sem a troca yaw↔pitch).
    reg = {0: (EMB_TARGET, 0.8)}
    service = _service(
        lambda call: [_face(0, [10, 10, 40, 40], 0.95, pose=(10.0, 20.0, 30.0))], reg)
    res = service.extract(_cfg(video_path, tmp_path), [_target(threshold=0.5)])
    sidecars = list((Path(res.run_dir) / "_meta").glob("alvo_v*_f*.json"))
    assert sidecars
    pose = json.loads(sidecars[0].read_text(encoding="utf-8"))["detection"]["pose"]
    assert pose == {"yaw": 20.0, "pitch": 10.0, "roll": 30.0}
