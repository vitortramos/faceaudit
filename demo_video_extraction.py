"""Demo CLI da Fase 1 — produtor de extração de faces de vídeo.

Sem UI: escolhe a âncora por clique (coordenada na imagem) ou, na
ausência de `--click`, pela maior face detectada no quadro-âncora.
Roda o laço sequencial e grava recortes + `_meta/` (sidecars, manifesto,
decisões, frame log) sob `<output>/<video>_<runtag>/`.

Uso:
    python demo_video_extraction.py CAMINHO_DO_VIDEO.mp4 \
        --anchor-frame 30 --click 640 360 --label suspeito_01 \
        --threshold 0.30 --target-fps 2 --output video_runs

Em GPU Pascal desta workstation, use SEM --gpu (use_gpu=False).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrai faces de um alvo ao longo de um vídeo.")
    ap.add_argument("video", help="Caminho do arquivo de vídeo.")
    ap.add_argument("--output", default="video_runs", help="Pasta raiz de saída.")
    ap.add_argument("--anchor-frame", type=int, default=0, help="Índice do quadro-âncora.")
    ap.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"),
                    help="Coordenada do clique no quadro-âncora (px). Sem isto, usa a maior face.")
    ap.add_argument("--label", default="alvo", help="Rótulo da identidade (nome da pasta).")
    ap.add_argument("--threshold", type=float, default=0.30, help="Limiar de identidade (cosseno).")
    ap.add_argument("--target-fps", type=float, default=2.0, help="FPS-alvo de amostragem.")
    ap.add_argument("--margin", type=float, default=2.0, help="Fator de margem do recorte.")
    ap.add_argument("--start", type=float, default=0.0, help="Início (s).")
    ap.add_argument("--end", type=float, default=None, help="Fim (s).")
    ap.add_argument("--model", default=None, help="Modelo de recognition (default do FFService).")
    ap.add_argument("--gpu", action="store_true", help="Usar GPU (evite em Pascal/ort>=1.20).")
    args = ap.parse_args()

    # Imports tardios: carregam ForensicFace/ONNX só quando o demo roda.
    from faceaudit.services.ff_service import FFService
    from faceaudit.services.video_frame_face_extraction_service import (
        VideoFrameFaceExtractionService,
    )
    from faceaudit.models.video.video_extraction_config import VideoExtractionConfig

    models = [args.model] if args.model else None
    ff = FFService(models=models, use_gpu=args.gpu)
    service = VideoFrameFaceExtractionService(ff)

    info = service.probe_video(args.video)
    print(f"📹 {info.filename} | {info.resolution[0]}x{info.resolution[1]} | "
          f"{info.fps_reported:.2f} fps | {info.frame_count_reported} quadros | "
          f"sha256={(info.sha256 or '')[:12]}…")

    frame = service.read_frame(args.video, args.anchor_frame)
    if frame is None:
        print(f"❌ Não foi possível ler o quadro-âncora {args.anchor_frame}.", file=sys.stderr)
        return 2

    if args.click:
        cx, cy = args.click
    else:
        dets = service.detect_in_frame(frame)
        if not dets:
            print("❌ Nenhuma face detectada no quadro-âncora; informe --anchor-frame com uma "
                  "face visível ou --click.", file=sys.stderr)
            return 3
        # maior face → centro da bbox como 'clique'
        def _area(d):
            b = np.asarray(d["bbox"]).astype(float).flatten()
            return (b[2] - b[0]) * (b[3] - b[1])
        big = max(dets, key=_area)
        b = np.asarray(big["bbox"]).astype(float).flatten()
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        print(f"ℹ️  Sem --click: usando a maior face, centro=({cx:.0f}, {cy:.0f}).")

    built = service.build_target_from_click(
        frame, cx, cy, label=args.label, similarity_threshold=args.threshold, model=args.model,
    )
    if built is None:
        print("❌ Nenhuma detecção contém o ponto de âncora informado.", file=sys.stderr)
        return 4
    target, det_idx, _dets = built
    print(f"🎯 Âncora: detecção #{det_idx} | fingerprint={target.anchor_embedding_fingerprint()[:24]}…")

    cfg = VideoExtractionConfig(
        video_path=args.video, output_root=args.output, target_fps=args.target_fps,
        start_s=args.start, end_s=args.end, margin=args.margin,
        similarity_threshold=args.threshold, recognition_model=args.model,
    )
    res = service.extract(
        cfg, [target],
        progress_callback=lambda m, p: print(f"  [{p:3d}%] {m}", end="\r", flush=True),
    )
    print()
    print(f"✅ Status: {res.extraction_status}")
    print(f"📁 Run: {res.run_dir}")
    print(f"   quadros amostrados={res.n_frames_sampled} | detecções={res.n_detections} | "
          f"aceitos={res.n_accepted_auto} | rejeitados={res.n_rejected_auto} | "
          f"sem detecção={res.n_no_detection_frames}")
    print(f"   manifesto: {res.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
