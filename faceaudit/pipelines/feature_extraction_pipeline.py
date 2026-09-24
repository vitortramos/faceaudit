import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# Imports do Domínio
from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models import FaceDataset
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

# Imports dos Serviços
from faceaudit.services import (
    FFService,
    HashService,
    OFIQService,
    SERFIQService
)
from faceaudit.services.image_io import imwrite_unicode_safe

# Mapeia a chave de `DatasetIngestionConfig.ff_kept_attributes` (UI) para o
# campo correspondente no `FFInfo`. Usado só no fluxo de criação para zerar os
# atributos não pedidos (a coluna some via skip-None em to_flat_dict).
_FF_KEPT_ATTR_TO_FIELD = {
    "ipd": "ipd",
    "fiqa": "fiqa_score",
    "det_score": "det_score",
    "gender": "gender",
    "age": "age",
    "yaw": "yaw",
    "pitch": "pitch",
    "roll": "roll",
}

# Atributos que vêm de `detect_and_align` (detector + landmark_3d_68 + genderage),
# SEM o embedding. `fiqa` NÃO está aqui — é calculado pelo runner de reconhecimento
# (acoplado ao embedding). Base do caminho de enriquecimento "atributos-só".
_FF_DETECT_ATTRS = frozenset({"yaw", "pitch", "roll", "gender", "age", "det_score", "ipd"})


class FeatureExtractionPipeline:
    """
    Orquestrador ETL (Extract, Transform, Load) para Datasets Faciais.

    Esta classe é responsável por coordenar todo o ciclo de vida da ingestão de um dataset,
    desde a leitura dos arquivos em disco até a extração de características biométricas
    e persistência final.

    Responsabilidades:
        1. **Extract (Extração):** Utiliza o `FaceDatasetAdapter` para varrer diretórios,
           aplicar templates de metadados ou Regex, e criar a estrutura inicial do dataset.
        2. **Transform (Transformação):** Orquestra a chamada sequencial dos serviços de IA
           (Biometria, Qualidade ISO, Robustez) conforme configurado no objeto `DatasetIngestionConfig`.
        3. **Load (Carga):** Persiste o resultado processado em um arquivo Parquet otimizado
           para leitura rápida futura.

    Attributes:
        storage_path (Path): Diretório base onde os arquivos .parquet processados serão salvos.
        logger (logging.Logger): Logger para registro de eventos e erros do pipeline.
    """

    def __init__(self, 
                 hash_service: HashService,
                 ff_service: FFService,
                 ofiq_service: OFIQService,
                 serfiq_service: SERFIQService,
                 storage_path: Path):
        """
        Inicializa o pipeline injetando as dependências de serviço.

        Args:
            hash_service (HashService): Serviço para cálculo de integridade (SHA256).
            ff_service (FFService): Serviço de IA (ForensicFace) para detecção e embeddings.
            ofiq_service (OFIQService): Serviço de cálculo de qualidade padrão ISO/IEC 29794-5.
            serfiq_service (SERFIQService): Serviço de cálculo de robustez estocástica (SER-FIQ).
            storage_path (Path): Caminho do diretório onde os datasets processados serão salvos.
        """
        self.hash_service = hash_service
        self.ff_service = ff_service
        self.ofiq_service = ofiq_service
        self.serfiq_service = serfiq_service
        
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger("FeatureExtractionPipeline")

        # Garante a existência do diretório de armazenamento na inicialização
        if not self.storage_path.exists():
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.critical(f"Falha ao criar diretório de storage {self.storage_path}: {e}")

    def execute(self, 
                config: DatasetIngestionConfig, 
                progress_callback: Optional[Callable[[str, int], None]] = None
                ) -> FaceDataset:
        """
        Executa o fluxo completo de ingestão (ETL) baseado no objeto de configuração.
        (Sempre varre os diretórios do zero, ignorando caches antigos).

        Args:
            config (DatasetIngestionConfig): DTO contendo todas as regras de ingestão
                (caminhos, flags de processamento, templates, metadados padrão).
            progress_callback (Optional[Callable[[str, int], None]]): Função de callback para
                atualizar a interface gráfica. Deve aceitar `(mensagem: str, porcentagem: int)`.
                Para cancelar a operação, este callback deve lançar `InterruptedError`.

        Returns:
            FaceDataset: O objeto dataset processado e enriquecido.

        Raises:
            Exception: Propaga qualquer erro inesperado ocorrido durante o processamento
                       (exceto InterruptedError, que é tratado salvando o estado parcial).
        """
        
        # 1. Setup e Definição de Caminhos
        # `config.dataset_dir` (campo "Destino" do diálogo) manda; sem ele,
        # cai na biblioteca global — o default histórico.
        dataset_name = config.dataset_name
        dataset_dir = Path(config.dataset_dir) if config.dataset_dir else self.storage_path
        if getattr(config, "self_contained", False):
            # Autocontido é UNIDADE: o dataset ganha pasta própria, com o
            # parquet e as imagens dentro. Assim a pasta inteira pode ser
            # movida/compactada sem quebrar os caminhos relativos.
            from faceaudit.services.dataset_localization_service import (
                self_contained_dataset_dir,
            )
            dataset_dir = self_contained_dataset_dir(dataset_dir, dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / f"{dataset_name}.parquet"

        # 2. Extract: Ingestão de Dados
        if progress_callback: 
            progress_callback("Lendo imagens e extraindo metadados...", 0)
        
        print(f"🚀 [Pipeline] Iniciando ingestão: {dataset_name}")
        
        # Delega a complexidade de leitura de disco/regex para o Adapter
        dataset = FaceDatasetAdapter.dataset_from_dir(
            dir_path=config.input_path,
            name=dataset_name,
            recursive=config.recursive,
            template=config.template,
            default_identity=config.default_identity,
            origin=config.origin,
            callback=progress_callback
        )

        # Propaga flag is_aligned do config pro dataset criado.
        # FFService.process_dataset usa esse flag pra decidir entre
        # process_face (foto inteira) e process_face_aligned (crop
        # 112×112 já alinhado).
        dataset.is_aligned = bool(getattr(config, "is_aligned", False))

        if len(dataset) == 0:
            print("⚠️ [Pipeline] Dataset vazio ou nenhuma face encontrada.")
            return dataset

        # 3. Transform (Parte A): Aplicação de Metadados Manuais
        if config.default_gender:
            print(f" ℹ️ [Pipeline] Aplicando Gênero Fixo: {config.default_gender}")
            dataset._df['gender'] = config.default_gender
            
        if config.default_ethnicity:
             print(f" ℹ️ [Pipeline] Aplicando Etnia Fixa: {config.default_ethnicity}")
             dataset._df['ethnicity'] = config.default_ethnicity

        # 4. Transform (Parte B): Processamento Biométrico e de Qualidade
        active_stages = []
        if config.compute_ff: active_stages.append("Biometria (ForensicFace)")
        if config.compute_ofiq: active_stages.append("Qualidade (OFIQ)")
        if config.compute_serfiq: active_stages.append("Robustez (SER-FIQ)")
        
        total_stages = len(active_stages)
        current_stage_idx = 0

        # Tracking de falhas por face — em process_dataset (criação) o
        # `dataset` inteiro é enviado a cada estágio, então todas as
        # faces são tocadas. Mesma semântica do enrich.
        per_face_failures: List[tuple] = []
        touched_face_ids: set = {id(f) for f in dataset.faces}
        # Sinaliza cancellation pelo usuário — relançado depois do save
        # parcial pra que o Worker trate corretamente como cancelado.
        interrupted = False

        try:
            # A. ForensicFace (Embeddings, Detecção, Pose, Idade)
            if config.compute_ff:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)

                ff_svc = self._resolve_ff_service(config)
                wanted_models = list(config.ff_models or [SepaelModel.default().ff_name])

                # Callback adaptativo: escala o % interno do FFService (0-100)
                # para caber no slot atual do pipeline.
                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                ff_svc.process_dataset(
                    dataset,
                    progress_callback=ff_callback,
                    models=wanted_models,
                )
                for f in dataset.faces:
                    if f.ff_info is None:
                        per_face_failures.append(
                            (f, "ff", "nenhum embedding gerado "
                             "(face não detectada ou imagem ilegível)"),
                        )

                # ── Modo "alinhar durante criação" ──
                # Após FF rodar nos originais, salva os crops alinhados
                # (que vivem em ff_info.aligned_face) como PNG no
                # target_dir e reconfigura o dataset pra apontar pros
                # PNGs em modo RELATIVE. Estágios OFIQ/SERFIQ subsequentes
                # rodam nas faces ALINHADAS — sem precisar re-alinhar
                # via Image Pipeline depois.
                # Em autocontido os crops nascem já dentro da pasta de imagens
                # do dataset — assim a portabilidade sai de um REWRITE_RELATIVE
                # no final, sem uma segunda cópia de todos os PNGs.
                crops_dir = config.aligned_target_dir
                if config.self_contained:
                    from faceaudit.services.dataset_localization_service import (
                        SELF_CONTAINED_IMAGES_SUBDIR,
                    )
                    crops_dir = dataset_dir / SELF_CONTAINED_IMAGES_SUBDIR
                if crops_dir and config.image_kind is DatasetImageKind.ALIGNED:
                    self._materialize_aligned_crops(
                        dataset, crops_dir, progress_callback,
                    )
                elif crops_dir and config.image_kind is DatasetImageKind.CROP:
                    # Recorte com margem sai da bbox que o FF já devolveu ao
                    # extrair o embedding — sem segunda passada de detecção.
                    self._materialize_face_crops(
                        dataset, crops_dir, config.crop_margin_factor,
                        progress_callback,
                    )
                current_stage_idx += 1

            # B. OFIQ (Qualidade ISO)
            if config.compute_ofiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                ofiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self.ofiq_service.process_dataset(dataset, callback=ofiq_tick)
                for f in dataset.faces:
                    if f.ofiq_info is None:
                        per_face_failures.append(
                            (f, "ofiq", "OFIQ não produziu resultado"),
                        )
                current_stage_idx += 1

            # C. SER-FIQ (Qualidade Estocástica)
            if config.compute_serfiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                serfiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self._resolve_serfiq_service(config).process_dataset(
                    dataset,
                    callback=serfiq_tick,
                    t=config.serfiq_t,
                    alpha=config.serfiq_alpha,
                )
                for f in dataset.faces:
                    if f.serfiq_info is None:
                        per_face_failures.append(
                            (f, "serfiq", "SER-FIQ não produziu score"),
                        )
                current_stage_idx += 1

        except InterruptedError:
            print("\n🛑 [Pipeline] Processamento interrompido pelo usuário.")
            print("   ⚠️ Salvando estado parcial do dataset...")
            interrupted = True
            # Não relançamos AQUI para permitir o bloco de persistência
            # abaixo executar. Relançamento ocorre DEPOIS do save (mais
            # abaixo) — Worker recebe a InterruptedError e trata como
            # cancelamento legítimo em vez de "sucesso".

        except Exception as e:
            self.logger.error(f"Erro inesperado durante a execução do pipeline: {e}", exc_info=True)
            raise e # Erros não previstos devem ser tratados pelo Worker/Controller.

        # Consolida marcas de erro nas colunas error/error_stage/
        # error_message das faces (mesma lógica do enrich).
        self._apply_error_marks_to_dataset(
            dataset, touched_face_ids, per_face_failures,
        )

        # Seleção de colunas de atributo FF (só na CRIAÇÃO): descarta os
        # atributos NÃO pedidos. O engine calcula gênero/idade/pose em BLOCO
        # (`extended`), então isto joga fora o que o usuário não marcou — a
        # coluna some do parquet (via skip-None em to_flat_dict) e não reaparece
        # com dado real num re-save. O enrich é aditivo/não-destrutivo e NÃO
        # passa por aqui.
        self._apply_ff_column_selection(dataset, config.ff_kept_attributes)

        # 5. Load: Persistência (Sobrescreve se existir).
        # O parquet vai pra `config.dataset_dir` quando informado; senão pra
        # `storage_path` da biblioteca global. Antes de salvar, o modo de
        # armazenamento é materializado: autocontido copia/relativiza as
        # imagens pra junto do parquet; caminho absoluto só resolve os paths.
        dataset = self._finalize_storage_mode(
            dataset, config, dataset_dir, progress_callback,
        )

        if progress_callback:
            progress_callback("Salvando arquivo Parquet...", 99)

        try:
            print(f"💾 [Pipeline] Salvando dataset em: {cache_file}")
            dataset.to_parquet(cache_file)
        except Exception as e:
            msg = f"Erro crítico ao salvar o dataset em {cache_file}: {e}"
            print(f"❌ {msg}")
            self.logger.error(msg)
            raise e

        # Re-lança DEPOIS do save parcial, pra que o Worker trate como
        # cancelado em vez de "sucesso". Estado parcial fica no disco.
        if interrupted:
            raise InterruptedError("Processamento interrompido pelo usuário.")

        if progress_callback:
            progress_callback("Processamento Concluído!", 100)

        return dataset

    def _calc_pct(self, current_stage: int, total_stages: int) -> int:
        if total_stages == 0: return 0
        # Reserva os 5% finais para persistência, distribuindo 95% entre as fases
        return int((current_stage / total_stages) * 95)

    # =========================================================================
    # Resolução dinâmica de serviços a partir do DatasetIngestionConfig
    # =========================================================================

    def _resolve_ff_service(self, config: DatasetIngestionConfig) -> FFService:
        """Retorna o FFService Singleton se ele já cobre os params do
        config; caso contrário, instancia um temporário com os params
        pedidos. O temporário paga ~5-10s de carga ONNX uma vez por
        execução — se você executar várias vezes seguidas com os mesmos
        params, o helper continua reutilizando."""
        wanted_models = list(config.ff_models or [SepaelModel.default().ff_name])
        det_size = int(config.ff_det_size)
        extended = bool(config.ff_extended)

        # use_gpu/batch_size: override POR execução (config) com fallback no
        # default da máquina (singleton). None no config = usar o do singleton.
        # O match do singleton também compara gpu/batch — senão um override que
        # difere do singleton seria ignorado pelo early-return.
        singleton_gpu = getattr(self.ff_service, "_use_gpu", False)
        singleton_batch = getattr(self.ff_service, "_batch_size", 32)
        use_gpu = config.use_gpu if config.use_gpu is not None else singleton_gpu
        batch_size = config.batch_size if config.batch_size is not None else singleton_batch

        if (self.ff_service.matches_config(
                models=wanted_models, det_size=det_size, extended=extended)
                and use_gpu == singleton_gpu and batch_size == singleton_batch):
            return self.ff_service

        print(
            f"⚙️ [Pipeline] FFService Singleton não cobre os params pedidos "
            f"(models={wanted_models}, det_size={det_size}, extended={extended}, "
            f"use_gpu={use_gpu}, batch_size={batch_size}). Instanciando temporário…"
        )
        return FFService(
            models=wanted_models,
            det_size=det_size,
            use_gpu=use_gpu,
            extended=extended,
            batch_size=batch_size,
        )

    def _build_single_model_ff_service(self, config: DatasetIngestionConfig, model: str) -> FFService:
        """FFService de UM modelo só, para o processamento sequencial multi-modelo
        em faces alinhadas — o pico de VRAM fica limitado a ~1 pipeline (em vez
        de N simultâneos), garantindo execução em GPUs menores sem OOM/thrashing.
        Reusa o singleton quando ele JÁ é exatamente esse modelo (evita recarga);
        senão constrói um temporário que o caller libera após usar."""
        det_size = int(config.ff_det_size)
        extended = bool(config.ff_extended)
        singleton_gpu = getattr(self.ff_service, "_use_gpu", False)
        singleton_batch = getattr(self.ff_service, "_batch_size", 32)
        use_gpu = config.use_gpu if config.use_gpu is not None else singleton_gpu
        batch_size = config.batch_size if config.batch_size is not None else singleton_batch
        if (self.ff_service.matches_config(models=[model], det_size=det_size, extended=extended)
                and use_gpu == singleton_gpu and batch_size == singleton_batch):
            return self.ff_service
        return FFService(
            models=[model], det_size=det_size, use_gpu=use_gpu,
            extended=extended, batch_size=batch_size,
        )

    def _resolve_serfiq_service(
        self, config: DatasetIngestionConfig
    ) -> SERFIQService:
        """Retorna o SERFIQService Singleton, ajustando T/alpha in-place
        antes do uso. SERFIQService é leve (não carrega modelo extra além
        do FF embedded), então mutar atributos é mais barato que rebuild."""
        # SERFIQService.process_dataset aceita `t=` e `alpha=` por chamada
        # (override por execução), então não precisa rebuild — basta passar
        # adiante na chamada. Retorna o Singleton.
        return self.serfiq_service

    def _make_noarg_stage_callback(
        self,
        stage_msg: str,
        base_pct: int,
        next_pct: int,
        total_items: int,
        progress_callback: Optional[Callable[[str, int], None]],
    ) -> Callable[[], None]:
        """
        Adaptador para serviços que emitem callback sem args (OFIQ, SER-FIQ),
        equivalente ao `ff_callback` adaptativo usado pelo FFService.

        Conta itens internamente, escala o progresso pro slot do estágio
        (`base_pct` → `next_pct`) e chama `progress_callback(msg, pct)`.

        Por que isso existe: o `EnrichWorker._callback` checa o flag de
        cancelamento dentro do progress_callback. Se OFIQ/SER-FIQ rodam
        sem callback, o cancel fica preso até a fase terminar. Esse
        adaptador garante que cada item processado dispare a verificação.
        """
        if total_items <= 0 or progress_callback is None:
            return lambda: None

        state = {"count": 0}
        slot_width = max(0, next_pct - base_pct)

        def _tick() -> None:
            state["count"] += 1
            internal_pct = min(100, int((state["count"] / total_items) * 100))
            scaled = base_pct + int((internal_pct / 100) * slot_width)
            progress_callback(stage_msg, scaled)

        return _tick
    
    def enrich_existing_dataset(self,
                                dataset: FaceDataset,
                                config: DatasetIngestionConfig,
                                progress_callback: Optional[Callable[[str, int], None]] = None
                                ) -> FaceDataset:
        """
        Processa um dataset já existente em memória, aplicando apenas os serviços de IA solicitados.
        Ignora as etapas de varredura de diretório (Extract) e carregamento de cache.

        PR 18 — idempotência: por padrão (`config.force_reprocess=False`),
        cada estágio recebe um sub-dataset com APENAS as faces que ainda
        não têm aquele enrichment, economizando GPU/CPU em datasets
        parcialmente enriquecidos. As mutações nos `FaceInfo` propagam
        pra `dataset` original via referência. `force_reprocess=True`
        ignora o filtro e processa tudo (sobrescreve os *_info existentes).
        """
        print(f"🚀 [Pipeline] Iniciando enriquecimento do dataset: {dataset.name}")

        # Propaga o override de `is_aligned` do config pro dataset antes
        # do FF rodar. Esse override existe pra corrigir datasets cuja
        # metadata está errada (parquets legados, gerados por scripts
        # externos, etc.) — sem isso, FF rodaria detect+align num crop
        # 112×112 e produziria embedding lixo silenciosamente.
        # O flag persiste no `dataset` em memória e vai pra metadata no
        # próximo save.
        dataset.is_aligned = bool(getattr(config, "is_aligned", False))

        active_stages = []
        if config.compute_ff: active_stages.append("Biometria (ForensicFace)")
        if config.compute_ofiq: active_stages.append("Qualidade (OFIQ)")
        if config.compute_serfiq: active_stages.append("Robustez (SER-FIQ)")

        total_stages = len(active_stages)
        current_stage_idx = 0

        if total_stages == 0:
            print("⚠️ [Pipeline] Nenhum processamento foi selecionado para enriquecimento.")
            return dataset

        # Rastreamento de falhas por face — preenchido após cada estágio
        # comparando quem foi enviado pro service (`target.faces`) vs.
        # quem ficou com o `*_info` esperado. Cada entrada é
        # `(face, stage_name, message)`. As marcas são consolidadas na
        # face via `_apply_error_marks_to_dataset` antes do save.
        per_face_failures: List[tuple] = []
        # IDs (via id()) das faces tocadas em algum estágio nesta
        # execução — usadas pra limpar as colunas de erro quando todas
        # as fases passaram (re-run convergente).
        touched_face_ids: set = set()
        # Sinaliza cancellation pelo usuário — relançado depois do save
        # parcial pra que o Worker trate corretamente como cancelado.
        interrupted = False

        try:
            if config.compute_ff:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)

                wanted_models = list(config.ff_models or [SepaelModel.default().ff_name])

                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                # Idempotência por ATRIBUTO (não só por embedding): faces sem
                # embedding (ou sem o fiqa pedido) vão pro caminho completo;
                # faces que já têm embedding mas faltam atributos de detecção
                # (pose/gênero/idade/det/ipd) vão pro caminho "atributos-só"
                # (detect_and_align, sem recomputar o embedding).
                wanted_attrs = self._resolve_wanted_ff_attrs(config)
                aligned = bool(getattr(dataset, "is_aligned", False))
                wanted_da = set() if aligned else {
                    a for a in wanted_attrs if a in _FF_DETECT_ATTRS}
                wants_fiqa = "fiqa" in wanted_attrs
                faces_full, faces_attrs_only = self._route_ff_faces(
                    dataset, wanted_models, wanted_da, wants_fiqa, config.force_reprocess)

                target = self._select_target_for_stage(
                    dataset, active_stages[current_stage_idx],
                    faces_full,
                    base_pct, progress_callback,
                    force=config.force_reprocess,
                )
                if target is not None:
                    for f in target.faces:
                        touched_face_ids.add(id(f))
                    # Multi-modelo em faces ALINHADAS: processa UM modelo por vez,
                    # liberando a VRAM entre eles → pico ≈ 1 pipeline (em vez de N
                    # simultâneos), o que garante rodar em GPUs menores sem
                    # OOM/thrashing e torna o cancelamento responsivo (checado a
                    # cada face/modelo). Em faces NÃO-alinhadas a detecção seria
                    # refeita por modelo (e com 1 modelo não há ganho) → nesses
                    # casos mantém o caminho único multi-modelo.
                    if bool(getattr(dataset, "is_aligned", False)) and len(wanted_models) > 1:
                        n_models = len(wanted_models)
                        for mi, model in enumerate(wanted_models):
                            m_base = base_pct + int((mi / n_models) * (next_pct - base_pct))
                            m_next = base_pct + int(((mi + 1) / n_models) * (next_pct - base_pct))

                            def ff_callback_seq(msg, internal_pct, _b=m_base, _n=m_next, _m=model):
                                if progress_callback:
                                    scaled = _b + int((internal_pct / 100) * (_n - _b))
                                    progress_callback(f"[{_m}] {msg}", scaled)

                            ff_svc = self._build_single_model_ff_service(config, model)
                            try:
                                ff_svc.process_dataset(
                                    target, progress_callback=ff_callback_seq, models=[model])
                            finally:
                                # Libera a VRAM do modelo antes de carregar o próximo
                                # (não toca no singleton da app, que é compartilhado).
                                if ff_svc is not self.ff_service:
                                    del ff_svc
                                    gc.collect()
                    else:
                        self._resolve_ff_service(config).process_dataset(
                            target, progress_callback=ff_callback, models=wanted_models)
                    self._log_stage_summary(
                        "FF", target, lambda f: f.ff_info is not None,
                    )
                    # Coleta falhas: face foi enviada mas ff_info ficou None
                    # (provável falha de leitura/detecção — service já loga
                    # o motivo via `_safe_read_image`).
                    for f in target.faces:
                        if f.ff_info is None:
                            per_face_failures.append(
                                (f, "ff", "nenhum embedding gerado "
                                 "(face não detectada ou imagem ilegível)"),
                            )

                # Caminho "atributos-só": adiciona pose/gênero/idade/det/ipd a
                # faces que JÁ têm embedding, rodando só o detector na original —
                # sem recomputar o embedding (pula o modelo de reconhecimento).
                if faces_attrs_only and wanted_da:
                    for f in faces_attrs_only:
                        touched_face_ids.add(id(f))
                    ff_svc_da = self._resolve_ff_service(config)
                    n_ok = self._enrich_attributes_only(
                        faces_attrs_only, dataset, ff_svc_da, wanted_da, ff_callback)
                    print(
                        f"🧭 [Pipeline] Atributos-só: {n_ok}/{len(faces_attrs_only)} "
                        "face(s) atualizada(s) sem recomputar embedding."
                    )
                current_stage_idx += 1

            if config.compute_ofiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                target = self._select_target_for_stage(
                    dataset, active_stages[current_stage_idx],
                    self._faces_needing_ofiq(dataset),
                    base_pct, progress_callback,
                    force=config.force_reprocess,
                )
                if target is not None:
                    for f in target.faces:
                        touched_face_ids.add(id(f))
                    ofiq_tick = self._make_noarg_stage_callback(
                        msg, base_pct, next_pct, len(target), progress_callback
                    )
                    self.ofiq_service.process_dataset(target, callback=ofiq_tick)
                    self._log_stage_summary(
                        "OFIQ", target, lambda f: f.ofiq_info is not None,
                    )
                    for f in target.faces:
                        if f.ofiq_info is None:
                            per_face_failures.append(
                                (f, "ofiq", "OFIQ não produziu resultado "
                                 "(verifique se o executável está OK)"),
                            )
                current_stage_idx += 1

            if config.compute_serfiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                target = self._select_target_for_stage(
                    dataset, active_stages[current_stage_idx],
                    self._faces_needing_serfiq(dataset),
                    base_pct, progress_callback,
                    force=config.force_reprocess,
                )
                if target is not None:
                    for f in target.faces:
                        touched_face_ids.add(id(f))
                    serfiq_tick = self._make_noarg_stage_callback(
                        msg, base_pct, next_pct, len(target), progress_callback
                    )
                    self._resolve_serfiq_service(config).process_dataset(
                        target,
                        callback=serfiq_tick,
                        t=config.serfiq_t,
                        alpha=config.serfiq_alpha,
                    )
                    self._log_stage_summary(
                        "SER-FIQ", target, lambda f: f.serfiq_info is not None,
                    )
                    for f in target.faces:
                        if f.serfiq_info is None:
                            per_face_failures.append(
                                (f, "serfiq", "SER-FIQ não produziu score"),
                            )
                current_stage_idx += 1

        except InterruptedError:
            print("\n🛑 [Pipeline] Enriquecimento interrompido pelo usuário.")
            interrupted = True

        except Exception as e:
            self.logger.error(f"Erro ao enriquecer dataset: {e}", exc_info=True)
            raise e

        # Aplica marcas de erro nas faces tocadas nesta execução
        # (semantic: a coluna `error*` reflete o resultado da última
        # run; re-run convergente limpa as falhas anteriores).
        self._apply_error_marks_to_dataset(
            dataset, touched_face_ids, per_face_failures,
        )

        # Persistência. Prefere o diretório original do dataset
        # (`_parquet_dir`, set por `from_parquet`) — sem isso, datasets
        # criados pelo Image Pipeline em subdir do projeto seriam
        # regravados na raiz da biblioteca global, separados das suas
        # `images/` e invisíveis pro usuário que abriu a versão do subdir.
        if dataset.name:
            base_dir = dataset._parquet_dir or self.storage_path
            cache_file = Path(base_dir) / f"{dataset.name}.parquet"
            if progress_callback and not interrupted:
                progress_callback("Salvando alterações...", 99)
            try:
                print(f"💾 [Pipeline] Salvando dataset enriquecido em: {cache_file}")
                dataset.to_parquet(cache_file)
            except Exception as e:
                self.logger.error(f"Erro ao salvar pós-enriquecimento: {e}")

        # Re-lança DEPOIS do save parcial, pra que o Worker/Controller
        # saiba que foi cancelado e não trate como sucesso. Sem isso, o
        # EnrichWorker emite `finished(dataset)` e a UI fecha o diálogo
        # como "concluído", ignorando o clique do usuário no Cancelar.
        if interrupted:
            raise InterruptedError("Enriquecimento interrompido pelo usuário.")

        if progress_callback:
            progress_callback("Enriquecimento Concluído!", 100)

        return dataset

    # =========================================================================
    # PR 18 — Helpers de idempotência por estágio
    # =========================================================================

    def _select_target_for_stage(
        self,
        dataset: FaceDataset,
        stage_label: str,
        missing_faces: list,
        base_pct: int,
        progress_callback: Optional[Callable[[str, int], None]],
        force: bool,
    ) -> Optional[FaceDataset]:
        """Decide o que enviar pro service do estágio:

          - `force=True` → o dataset original inteiro (reprocessa tudo).
          - `missing_faces` vazio → None (skip do estágio inteiro).
          - Caso intermediário → sub-FaceDataset com APENAS os faltantes.
            Mutações nos `FaceInfo` propagam ao original via referência.

        Loga skip/processed via `progress_callback` quando aplicável.
        """
        total = len(dataset)
        if force:
            if progress_callback:
                progress_callback(
                    f"Enriquecendo (force): {stage_label} — {total} face(s)…",
                    base_pct,
                )
            return dataset

        n_missing = len(missing_faces)
        n_skipped = total - n_missing
        if n_missing == 0:
            if progress_callback:
                progress_callback(
                    f"Pulando: {stage_label} — todas as {total} face(s) já enriquecidas",
                    base_pct,
                )
            print(
                f"⏭ [Pipeline] {stage_label}: pulando — todas as {total} faces "
                "já têm este enrichment."
            )
            return None

        if progress_callback:
            progress_callback(
                f"Enriquecendo: {stage_label} — {n_missing} pendentes "
                f"({n_skipped} já enriquecidas)",
                base_pct,
            )
        print(
            f"➡ [Pipeline] {stage_label}: processando {n_missing} de {total} faces "
            f"(pulando {n_skipped} já enriquecidas)."
        )
        sub = FaceDataset(
            name=f"{dataset.name}_subset",
            faces=missing_faces,
            path_mode=dataset.path_mode,
            path_root=dataset.path_root,
        )
        sub._parquet_dir = dataset._parquet_dir
        # Propaga is_aligned ao subset — sem isto o FFService re-detectaria
        # faces já alinhadas (112×112), o que falha em crops apertados (ex.:
        # datasets de vídeo). Bug afetava enrich em QUALQUER dataset alinhado.
        sub.is_aligned = dataset.is_aligned
        return sub

    @staticmethod
    def _materialize_aligned_crops(
        dataset: FaceDataset,
        aligned_target_dir: Path,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """Salva os crops alinhados (ff_info.aligned_face, RGB 112×112
        — ForensicFace retorna em RGB via cvtColor BGR2RGB) como PNG
        em `aligned_target_dir/` e reconfigura o dataset pra apontar
        pros PNGs em modo ABSOLUTE.

        Pré-requisito: FFService.process_dataset já rodou — cada
        face com `ff_info.aligned_face` populado vira PNG no destino.
        Faces sem aligned_face (falha de detecção) ficam com o
        image_path original (apontando pra imagem desalinhada) e são
        marcadas como erro pelo caller.

        Side-effects:
          - Cria `aligned_target_dir/` (mkdir parents=True).
          - Salva 1 PNG por face com aligned_face direto em
            `aligned_target_dir/<file>.png` (sem subpasta intermediária).
          - Muta cada face com aligned_face: `image_path` =
            caminho absoluto do PNG, `file_name` = basename do PNG,
            `hash` = SHA-256 do PNG, `ff_info.aligned_face = None`
            (libera memória — embedding já foi extraído).
          - Se ao menos uma face foi materializada, muta o dataset:
            `path_mode=ABSOLUTE`, `path_root=None`, `is_aligned=True`.
            Se zero faces foram materializadas (FF não rodou ou todas as
            detecções falharam), **não** muta o dataset — evita corromper
            o estado declarando `is_aligned`/`path_mode` enquanto os
            paths ainda apontam pra imagens desalinhadas.

        Por que ABSOLUTE e não RELATIVE: cada `image_path` fica
        self-describing — perito abre o parquet em qualquer lugar e os
        paths apontam direto pros PNGs alinhados onde foram criados.
        Para portabilidade (mover o `.fa` pra outra máquina), o fluxo
        canônico é "Selar projeto", que reescreve paths como relativos
        e copia as imagens pra dentro do projeto.

        Nota: o **parquet em si** não fica aqui — vai pra `storage_path`
        da biblioteca global de datasets (decisão do caller). Esta função
        cuida apenas dos PNGs alinhados direto em `aligned_target_dir/`.
        """
        import cv2
        from faceaudit.services.hash_service import HashService

        aligned_target_dir = Path(aligned_target_dir).resolve()
        aligned_target_dir.mkdir(parents=True, exist_ok=True)
        hash_service = HashService()
        used_names: set = set()
        n_materialized = 0

        n = len(dataset.faces)
        print(
            f"📐 [Pipeline] Materializando {n} face(s) alinhada(s) em "
            f"{aligned_target_dir}"
        )
        for idx, face in enumerate(dataset.faces):
            if (
                face.ff_info is None
                or face.ff_info.aligned_face is None
            ):
                # Face falhou no FF — sem crop alinhado pra salvar.
                # Mantém path original; caller já marca como erro.
                continue

            # ForensicFace._assemble_result armazena `aligned_face` em
            # RGB (cvtColor BGR2RGB). cv2.imencode/imwrite esperam BGR
            # — sem essa conversão os PNGs salvariam com R↔B trocados.
            rgb = face.ff_info.aligned_face
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Nome do PNG: usa o file_name original (sanitizado) com
            # extensão .png forçada. Sufixa em caso de colisão.
            base = face.file_name or Path(face.image_path).name or f"face_{idx}"
            stem = Path(base).stem
            rel_name = f"{stem}.png"
            i = 1
            while rel_name in used_names:
                rel_name = f"{stem}_{i}.png"
                i += 1
            used_names.add(rel_name)
            dst = aligned_target_dir / rel_name

            # `cv2.imwrite` falha SILENCIOSAMENTE em paths com chars não-ASCII
            # no Windows. Usa o helper compartilhado (image_io), que encoda
            # em memória e grava bytes via pathlib (API Unicode do Windows).
            if not imwrite_unicode_safe(dst, bgr):
                print(f"⚠️  [Pipeline] gravação do crop alinhado falhou: {dst}")
                continue

            # Atualiza FaceInfo: image_path absoluto pro PNG alinhado,
            # file_name = basename do PNG.
            face.image_path = dst.resolve().as_posix()
            face.file_name = rel_name
            face.hash = hash_service.calculate_file_hash(dst)
            # Libera memória — embedding e demais campos já estão em
            # ff_info; aligned_face foi persistido em disco.
            face.ff_info.aligned_face = None
            n_materialized += 1

            if progress_callback and (idx % 50 == 0 or idx + 1 == n):
                progress_callback(
                    f"Materializando crops alinhados ({idx + 1}/{n})…",
                    95 + int((idx + 1) / n * 4),  # 95..99% slot
                )

        if n_materialized == 0:
            # Nenhum PNG salvo — manter o estado original do dataset.
            # Sem isso, o dataset ficava declarado is_aligned/RELATIVE
            # enquanto os paths apontavam pras imagens desalinhadas, em
            # estado inconsistente.
            print(
                "⚠️  [Pipeline] Nenhuma face alinhada foi materializada "
                "(FF não retornou `aligned_face`). Dataset preservado em "
                "modo ABSOLUTE original — sem marcar is_aligned."
            )
            return

        # Reconfigura o dataset pra ABSOLUTE — cada face já tem path
        # self-describing. `path_root` é irrelevante em ABSOLUTE.
        # `_parquet_dir` será populado por `from_parquet` quando o
        # dataset for relido do disco (apontando pro storage_path
        # global, não pro aligned_target_dir).
        dataset.path_mode = PathMode.ABSOLUTE
        dataset.path_root = None
        dataset.is_aligned = True
        print(
            f"📐 [Pipeline] {n_materialized}/{n} face(s) materializada(s). "
            f"Dataset marcado: is_aligned=True, path_mode=ABSOLUTE."
        )

    @staticmethod
    def _finalize_storage_mode(
        dataset: FaceDataset,
        config,
        dataset_dir: Path,
        progress_callback=None,
    ) -> FaceDataset:
        """Materializa o modo de armazenamento escolhido, ANTES do save.

        - **Autocontido**: copia/relativiza as imagens pra
          `<dataset_dir>/<nome>_images/` via `DatasetLocalizationService`
          (`COPY_ORIGINALS`), deixando o dataset portátil (RELATIVE).
          Crops já materializados nessa pasta pelo "alinhar durante a
          criação" são pulados por hash (`skip_existing=True`), então esse
          caminho NÃO paga uma segunda cópia. Faces que falharam no
          alinhamento seguem apontando pro original — a cópia as traz
          junto, em vez de descartá-las (as linhas `error=True` são o que
          o perito usa pra inspecionar falhas).
        - **Caminho absoluto**: resolve cada path e marca ABSOLUTE, sem
          cópia — o dataset referencia as imagens onde elas estão.

        Retorna o dataset a persistir (objeto NOVO no modo autocontido —
        `localize` não muta a entrada).
        """
        from faceaudit.enums.path_mode import PathMode

        if not getattr(config, "self_contained", False):
            for face in dataset.faces:
                try:
                    face.image_path = str(dataset.resolve_image_path(face))
                except Exception:
                    continue    # path irresolvível: preserva o que já está lá
            dataset.path_mode = PathMode.ABSOLUTE
            dataset.path_root = None
            dataset._parquet_dir = Path(dataset_dir).resolve()
            return dataset

        from faceaudit.enums.localization_mode import LocalizationMode
        from faceaudit.services.dataset_localization_service import (
            SELF_CONTAINED_IMAGES_SUBDIR, DatasetLocalizationService,
        )

        def _cb(idx, total, _msg):
            if progress_callback and total and (idx % 50 == 0 or idx + 1 == total):
                progress_callback(
                    f"Copiando imagens para a pasta do dataset… ({idx + 1}/{total})", 98,
                )

        res = DatasetLocalizationService().localize(
            dataset, Path(dataset_dir),
            mode=LocalizationMode.COPY_ORIGINALS,
            copy_subdir=SELF_CONTAINED_IMAGES_SUBDIR,
            progress_callback=_cb,
        )
        final = res.dataset
        # `localize` constrói um FaceDataset novo; estes campos não vêm junto.
        final.name = dataset.name
        final.provenance_metadata = dict(getattr(dataset, "provenance_metadata", {}) or {})
        final.is_aligned = dataset.is_aligned
        final._parquet_dir = Path(dataset_dir).resolve()

        # Nenhuma imagem copiada = dataset vazio. Gravar isso produz um
        # parquet de poucas centenas de bytes, sem colunas, que a UI
        # aceita, lista como recurso e some dos dropdowns depois — o
        # perito só descobre muito adiante. Melhor falhar aqui.
        if dataset.faces and not final.faces:
            motivos = "; ".join(
                f"{caminho}: {motivo}" for caminho, motivo in list(res.failures)[:3]
            )
            raise RuntimeError(
                f"Nenhuma das {res.n_processed} imagens do dataset pôde ser "
                f"copiada para a pasta do dataset — nada foi gravado. "
                f"Confira se os caminhos das imagens ainda são válidos. "
                f"Primeiras falhas: {motivos}"
            )

        # `_copy_originals` DESCARTA a face cuja imagem não pôde ser copiada.
        # Num dataset pericial isso não pode passar silenciosamente.
        if res.n_failed:
            aviso = (
                f"{res.n_failed} imagem(ns) não puderam ser copiadas e ficaram "
                f"DE FORA do dataset autocontido (de {res.n_processed})."
            )
            print(f"⚠️  [Pipeline] {aviso}")
            for caminho, motivo in list(res.failures)[:10]:
                print(f"    - {caminho}: {motivo}")
            if progress_callback:
                progress_callback(f"⚠️ {aviso}", 98)
        return final

    @staticmethod
    def _materialize_face_crops(
        dataset: FaceDataset,
        target_dir: Path,
        margin_factor: float,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """Grava o recorte da face COM MARGEM, sem rotação, como PNG.

        Irmão de `_materialize_aligned_crops`, com três diferenças que
        importam:

        - parte da imagem ORIGINAL e da `ff_info.bbox` que o FF já
          devolveu ao extrair o embedding — não há segunda passada de
          detecção, só leitura e gravação de arquivo;
        - não aplica transformação de similaridade, então a geometria e as
          proporções do rosto são preservadas (é o que serve para laudo e
          comparação visual);
        - **não marca `is_aligned`**. Um recorte não é face alinhada; se o
          dataset se declarasse alinhado, o `FFService` pularia
          detect+align num enriquecimento futuro e produziria embeddings
          errados em silêncio.

        Face sem bbox (detecção falhou) mantém o path original e é marcada
        como erro pelo caller — igual ao caminho do alinhamento.
        """
        import numpy as _np
        from forensicface.geometry import extend_bbox

        from faceaudit.services.hash_service import HashService
        from faceaudit.services.image_io import imread_unicode_safe

        target_dir = Path(target_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        hash_service = HashService()
        used_names: set = set()
        n_materialized = 0

        n = len(dataset.faces)
        print(
            f"✂️  [Pipeline] Materializando {n} recorte(s) com margem "
            f"{margin_factor:g}× em {target_dir}"
        )
        for idx, face in enumerate(dataset.faces):
            bbox = getattr(face.ff_info, "bbox", None) if face.ff_info else None
            if bbox is None:
                continue

            try:
                origem = dataset.resolve_image_path(face)
                imagem = imread_unicode_safe(str(origem))
            except Exception:
                imagem = None
            if imagem is None:
                print(f"⚠️  [Pipeline] imagem ilegível para recorte: {face.image_path}")
                continue

            x1, y1, x2, y2 = extend_bbox(
                _np.asarray(bbox, dtype="float32"), imagem.shape,
                margin_factor=float(margin_factor),
            )
            altura, largura = imagem.shape[:2]
            x1 = max(0, min(largura - 1, int(x1)))
            y1 = max(0, min(altura - 1, int(y1)))
            x2 = max(x1 + 1, min(largura, int(x2)))
            y2 = max(y1 + 1, min(altura, int(y2)))
            recorte = imagem[y1:y2, x1:x2].copy()

            base = face.file_name or Path(face.image_path).name or f"face_{idx}"
            stem = Path(base).stem
            rel_name = f"{stem}.png"
            i = 1
            while rel_name in used_names:
                rel_name = f"{stem}_{i}.png"
                i += 1
            used_names.add(rel_name)
            dst = target_dir / rel_name

            if not imwrite_unicode_safe(dst, recorte):
                print(f"⚠️  [Pipeline] gravação do recorte falhou: {dst}")
                continue

            face.image_path = dst.resolve().as_posix()
            face.file_name = rel_name
            face.hash = hash_service.calculate_file_hash(dst)
            if face.ff_info is not None:
                face.ff_info.aligned_face = None   # já extraído; libera memória
            n_materialized += 1

            if progress_callback and (idx % 50 == 0 or idx + 1 == n):
                progress_callback(
                    f"Materializando recortes ({idx + 1}/{n})…",
                    95 + int((idx + 1) / n * 4),
                )

        if n_materialized == 0:
            print(
                "⚠️  [Pipeline] Nenhum recorte materializado (FF não devolveu "
                "bbox). Dataset preservado em modo ABSOLUTE original."
            )
            return

        dataset.path_mode = PathMode.ABSOLUTE
        dataset.path_root = None
        # is_aligned permanece False DE PROPÓSITO — ver docstring.
        print(
            f"✂️  [Pipeline] {n_materialized} recorte(s) gravado(s) em {target_dir}"
        )

    @staticmethod
    def _apply_ff_column_selection(dataset: FaceDataset, kept_attributes) -> None:
        """Descarta (None) os atributos FF cujas colunas o usuário NÃO pediu
        (`DatasetIngestionConfig.ff_kept_attributes`). None = manter todos
        (compat). Combinado com o skip-None de `FaceInfo.to_flat_dict`, a coluna
        some do parquet e não reaparece com dado real num re-save. Chamado SÓ no
        fluxo de criação — o enrich é aditivo/não-destrutivo e não passa aqui."""
        if kept_attributes is None:
            return
        keep = set(kept_attributes)
        drop_fields = [
            field for user_key, field in _FF_KEPT_ATTR_TO_FIELD.items()
            if user_key not in keep
        ]
        if not drop_fields:
            return
        for face in getattr(dataset, "faces", []) or []:
            ff = getattr(face, "ff_info", None)
            if ff is None:
                continue
            for f in drop_fields:
                setattr(ff, f, None)

    @staticmethod
    def _apply_error_marks_to_dataset(
        dataset: FaceDataset,
        touched_face_ids: set,
        failures: list,
    ) -> None:
        """Consolida as falhas por face nas colunas `error/error_stage/
        error_message` do FaceInfo.

        Semântica:
          - Faces TOCADAS nesta execução (algum estágio rodou) têm
            suas colunas RESETADAS: se passaram em tudo → error=False;
            se falharam em algum estágio → error=True com stage/msg
            consolidados.
          - Faces NÃO TOCADAS (idempotência pulou, force=False e já
            tinham o enrichment) mantêm os valores anteriores
            inalterados.

        Múltiplas falhas na mesma face são concatenadas com `,` em
        `error_stage` e `|` em `error_message` (formato legível e
        fácil de filtrar via str.contains).
        """
        # Indexa falhas por id(face) — listas de (stage, message)
        by_face: Dict[int, List[tuple]] = {}
        for face, stage, msg in failures:
            by_face.setdefault(id(face), []).append((stage, msg))

        for face in dataset.faces:
            fid = id(face)
            if fid not in touched_face_ids:
                continue  # não tocada → preserva estado anterior
            entries = by_face.get(fid)
            if not entries:
                # Passou em todos os estágios desta run — limpa marca.
                face.error = False
                face.error_stage = None
                face.error_message = None
                continue
            # Pelo menos um estágio falhou — consolida.
            stages = ",".join(s for s, _ in entries)
            messages = " | ".join(f"[{s}] {m}" for s, m in entries)
            face.error = True
            face.error_stage = stages
            face.error_message = messages

    @staticmethod
    def _log_stage_summary(
        stage_label: str,
        target: FaceDataset,
        has_enrichment: Callable[[Any], bool],
    ) -> None:
        """Conta quantas faces do `target` ficaram com o enrichment
        populado e loga no terminal. Quando 0/N, marca FALHA TOTAL —
        sintoma típico de path resolution quebrado, modelo não-carregado
        ou exe externo (OFIQ) ausente. Sem esse sumário, falhas
        per-face engolidas pelos services ficavam invisíveis e o
        usuário só percebia que "as colunas não apareceram" sem
        nenhuma pista no terminal.
        """
        total = len(target)
        if total == 0:
            return
        ok = sum(1 for f in target if has_enrichment(f))
        if ok == 0:
            print(
                f"❌ [Pipeline] {stage_label}: 0/{total} faces enriquecidas — "
                f"FALHA TOTAL. Verifique paths das imagens (logs ⚠️ acima) "
                f"e dependências do serviço."
            )
        elif ok < total:
            print(
                f"⚠️ [Pipeline] {stage_label}: {ok}/{total} faces "
                f"enriquecidas ({total - ok} falharam — ver logs ⚠️ acima)."
            )
        else:
            print(f"✅ [Pipeline] {stage_label}: {ok}/{total} faces enriquecidas.")

    @staticmethod
    def _faces_needing_ff(dataset: FaceDataset, wanted_models: list) -> list:
        """Faces sem TODOS os embeddings dos modelos solicitados.

        Convenção do FFInfo: `embedding_<model>` (ex: `embedding_sepaelv2`).
        Se a face tem ff_info mas falta pelo menos um dos `wanted_models`,
        ela é incluída — o serviço preenche o que faltar.
        """
        out = []
        for face in dataset.faces:
            ff = getattr(face, "ff_info", None)
            if ff is None:
                out.append(face)
                continue
            for model in wanted_models:
                attr = f"embedding_{model}"
                if getattr(ff, attr, None) is None:
                    out.append(face)
                    break
        return out

    @staticmethod
    def _resolve_wanted_ff_attrs(config: DatasetIngestionConfig) -> set:
        """Atributos FF pedidos. `ff_kept_attributes` (granular, novo) tem
        prioridade; senão deriva de `ff_extended` (compat: extended=todos os 8,
        senão nenhum)."""
        if getattr(config, "ff_kept_attributes", None) is not None:
            return set(config.ff_kept_attributes)
        if config.ff_extended:
            return {"fiqa", "yaw", "pitch", "roll", "gender", "age", "det_score", "ipd"}
        return set()

    @staticmethod
    def _route_ff_faces(dataset, wanted_models, wanted_da, wants_fiqa, force):
        """Enrich: divide as faces em (full, attrs_only).
          - full = falta embedding OU falta o fiqa pedido (fiqa é acoplado ao
            reconhecimento) → caminho completo (detect+align+extract).
          - attrs_only = tem todos os embeddings e o fiqa pedido, mas falta um
            atributo de detect_and_align (pose/gênero/idade/det/ipd) → roda só o
            detector na imagem original, preservando o embedding.
        `force` joga tudo no full. `wanted_da` deve vir vazio para datasets
        alinhados (esses atributos não existem em crops 112×112)."""
        if force:
            return list(dataset.faces), []
        full, attrs_only = [], []
        for face in dataset.faces:
            ff = getattr(face, "ff_info", None)
            missing_emb = ff is None or any(
                getattr(ff, f"embedding_{m}", None) is None for m in wanted_models)
            if missing_emb:
                full.append(face)
                continue
            if wants_fiqa and getattr(ff, "fiqa_score", None) is None:
                full.append(face)          # fiqa exige o caminho de reconhecimento
                continue
            if any(getattr(ff, _FF_KEPT_ATTR_TO_FIELD[a], None) is None for a in wanted_da):
                attrs_only.append(face)
        return full, attrs_only

    @staticmethod
    def _merge_detect_attrs(ff, det_result: dict, wanted_da: set) -> None:
        """Mescla SÓ os atributos de detecção pedidos no FFInfo, sem tocar no
        embedding. `pose` vem como vetor cru [pitch, yaw, roll] do
        `detect_and_align` (mesma convenção do fluxo de vídeo)."""
        import numpy as np
        pose = det_result.get("pose")
        pmap = {}
        if pose is not None and len(pose) >= 3:
            pmap = {"pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])}
        for attr in wanted_da:
            if attr in ("yaw", "pitch", "roll"):
                if attr in pmap:
                    setattr(ff, attr, pmap[attr])
            elif attr == "det_score":
                v = det_result.get("det_score")
                if v is not None:
                    ff.det_score = float(v)
            elif attr == "gender":
                v = det_result.get("gender")
                if v is not None:
                    ff.gender = v
            elif attr == "age":
                v = det_result.get("age")
                if v is not None:
                    ff.age = float(v)
            elif attr == "ipd":
                kps = det_result.get("keypoints")
                if kps is not None and len(kps) >= 2:
                    ff.ipd = float(np.linalg.norm(
                        np.asarray(kps[0], dtype=float) - np.asarray(kps[1], dtype=float)))

    @staticmethod
    def _enrich_attributes_only(faces, dataset, ff_svc, wanted_da,
                                progress_callback=None) -> int:
        """Adiciona atributos de detect_and_align a faces que JÁ têm embedding —
        roda só o detector na imagem original, sem recomputar o embedding (pula o
        modelo de reconhecimento, o mais pesado). Só os atributos PEDIDOS são
        mesclados. Retorna quantas faces foram atualizadas."""
        from faceaudit.services.image_io import imread_unicode_safe
        from faceaudit.models.ff_info import FFInfo
        engine = getattr(ff_svc, "app", None)
        if engine is None or not wanted_da:
            return 0
        n_ok = 0
        total = len(faces)
        for i, face in enumerate(faces):
            try:
                src = dataset.resolve_image_path(face)
            except Exception:
                src = getattr(face, "image_path", None)
            img = imread_unicode_safe(str(src)) if src else None
            if img is None:
                continue
            try:
                res = engine.detect_and_align(img, single_face=True)
            except Exception:
                res = None
            if not res:
                continue
            ff = face.ff_info or FFInfo()
            FeatureExtractionPipeline._merge_detect_attrs(ff, res, wanted_da)
            face.ff_info = ff
            n_ok += 1
            if progress_callback and (i % 25 == 0 or i == total - 1):
                progress_callback(f"Atributos-só: {i + 1}/{total}",
                                  int((i + 1) / max(1, total) * 100))
        return n_ok

    @staticmethod
    def _faces_needing_ofiq(dataset: FaceDataset) -> list:
        """Faces sem `ofiq_info` populado."""
        return [f for f in dataset.faces if getattr(f, "ofiq_info", None) is None]

    @staticmethod
    def _faces_needing_serfiq(dataset: FaceDataset) -> list:
        """Faces sem `serfiq_info` populado."""
        return [f for f in dataset.faces if getattr(f, "serfiq_info", None) is None]