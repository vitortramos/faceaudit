from pathlib import Path

from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.enums.sepael_model import SepaelModel
from .enrich_dataset_view import EnrichDatasetView


class EnrichDatasetController:
    """Controlador da janela de Enriquecimento (redesenho por tipo de coluna).

    Analisa o dataset atual POR COLUNA (por modelo de embedding e por atributo)
    para mostrar o status (presente/parcial/ausente) e sugerir o que falta. O
    get_config mapeia para o backend: `ff_kept_attributes` (atributos a adicionar),
    `use_gpu`/`batch_size` (override por execução), `is_aligned` e `ff_models`.
    """

    def __init__(self, dataset, parent=None, ff_available: bool = True,
                 ofiq_available: bool = True, default_use_gpu: bool = False,
                 default_batch_size: int = 32):
        self.dataset = dataset
        self.view = EnrichDatasetView(
            parent, default_use_gpu=default_use_gpu, default_batch_size=default_batch_size)

        # Modelos com cobertura total (usados como `ff_models` quando o usuário
        # não marca nenhum — evita recalcular embeddings ao adicionar só atributos).
        self._fully_present_models: list = []

        # Pré-preenche o override de "faces já alinhadas" com o valor do dataset.
        self.view.set_ff_is_aligned(bool(getattr(dataset, "is_aligned", False)))
        self.view.set_stage_availability(ff_available=ff_available, ofiq_available=ofiq_available)
        self._analyze_and_populate()

    def exec(self):
        return self.view.exec()

    # ───────────────────────── análise por coluna ─────────────────────────
    def _analyze_and_populate(self):
        faces = getattr(self.dataset, "faces", [])
        total = len(faces)
        aligned = self.view.is_input_aligned()
        if total == 0:
            self.view.set_info("Dataset vazio — nada a enriquecer.")
            return

        kind = "faces já alinhadas 112×112" if aligned else "fotos originais"
        self.view.set_info(
            f"{total} face(s) · {kind}. Os selos mostram o que já existe; "
            "marque o que deseja adicionar.")

        def cov(pred) -> float:
            return sum(1 for f in faces if pred(f)) / total

        def ff(f):
            return getattr(f, "ff_info", None)

        # Embeddings por modelo.
        for model in self.view._ff_model_checks:
            c = cov(lambda f, m=model: getattr(ff(f), f"embedding_{m}", None) is not None)
            if c >= 0.999:
                self._fully_present_models.append(model)
            self._apply_status("emb:" + model, c)

        # FIQA (sobrevive a faces alinhadas).
        self._apply_status("fiqa", cov(lambda f: getattr(ff(f), "fiqa_score", None) is not None))

        if aligned:
            return   # pose/outros/OFIQ não existem em crops alinhados

        self._apply_status("ofiq", cov(lambda f: getattr(f, "ofiq_info", None) is not None))
        # Pose e Outros: status POR item (cada um pode ter cobertura diferente —
        # ex.: det_score presente mas gênero/idade ausentes).
        for key, fld in (("attr:yaw", "yaw"), ("attr:pitch", "pitch"), ("attr:roll", "roll"),
                         ("attr:det_score", "det_score"), ("attr:ipd", "ipd"),
                         ("attr:gender", "gender"), ("attr:age", "age")):
            self._apply_status(key, cov(lambda f, fl=fld: getattr(ff(f), fl, None) is not None))

    def _apply_status(self, key: str, coverage: float):
        """Selo + auto-check. Pré-marca SÓ o que está parcial (completa dado
        inconsistente). Presente fica desmarcado (não re-adiciona) e totalmente
        ausente também (opt-in — o selo "ausente" já avisa; não presume que o
        usuário quer adicionar tudo que falta)."""
        pct = int(round(coverage * 100))
        if coverage >= 0.999:
            self.view.set_item_status(key, "present", "presente")
            self.view.set_item_checked(key, False)
        elif coverage <= 0.001:
            self.view.set_item_status(key, "absent", "ausente")
            self.view.set_item_checked(key, False)
        else:
            self.view.set_item_status(key, "partial", f"parcial {pct}%")
            self.view.set_item_checked(key, True)

    # ───────────────────────── config ─────────────────────────
    def get_config(self) -> DatasetIngestionConfig:
        v = self.view
        flags = v.get_content_flags()
        aligned = v.is_input_aligned()
        kept = [k for k, on in flags.items() if on]

        # FF roda se há embedding OU atributo pedido. Extended liga se algum
        # atributo do bundle (fiqa/pose/gênero/idade) é pedido — necessário para
        # o detect_and_align devolver gênero/idade/pose.
        compute_ff = v.any_model_checked() or any(flags.values())
        ff_extended = bool(flags["fiqa"] or flags["yaw"] or flags["pitch"]
                           or flags["roll"] or flags["gender"] or flags["age"])

        # ff_models: os marcados; se nenhum, os já 100% presentes (para a
        # idempotência não recomputar embeddings ao adicionar só atributos);
        # fallback no default (__post_init__ nunca deixa vazio).
        ff_models = (v.get_ff_models() or self._fully_present_models
                     or [SepaelModel.default().ff_name])

        return DatasetIngestionConfig(
            input_path=Path("."),                     # ignorado no enrich
            dataset_name=self.dataset.name,
            compute_ff=compute_ff,
            compute_ofiq=(v.chk_ofiq.isChecked() and not aligned),   # OFIQ exige a original
            compute_serfiq=False,                     # SER-FIQ fora da UI
            ff_models=ff_models,
            ff_det_size=v.get_ff_det_size(),
            ff_extended=ff_extended,
            ff_kept_attributes=kept,                  # atributos a adicionar
            use_gpu=v.get_use_gpu(),
            batch_size=v.get_batch_size(),
            is_aligned=aligned,
            force_reprocess=v.chk_force_reprocess.isChecked(),
        )
