# Testes

Lista temática dos arquivos de `tests/`. O `CLAUDE.md` da raiz traz só a suíte inteira e as suítes-chave; o resto está aqui. Para rodar tudo:

```bash
python -m pytest tests/
```

Utilitário de conferência de parquet (não é teste do pytest):

```bash
python tests/check_parquet.py
```

## Motor de estratégias, armazenamento e projeto

```bash
python -m pytest tests/test_full_pipeline.py
python -m pytest tests/transformation_pipeline_test.py
python -m pytest tests/test_storage_and_repositories.py
python -m pytest tests/test_instruction_execution_frame.py      # frame de execução por instrução (fase)
python -m pytest tests/test_cascade_invalidation.py             # invalidação transitiva após editar config
python -m pytest tests/test_validation_rule_rebuild.py
python -m pytest tests/test_strategy_folder.py
python -m pytest tests/test_artifact_layout_migration.py
python -m pytest tests/test_datasets_layout_migration.py
python -m pytest tests/test_orphan_cleanup.py
python -m pytest tests/test_score_artifact_type_routing.py
python -m pytest tests/test_software_provenance.py              # versão (fonte única) + commit em toda entrada da custódia
```

## Transformações, similaridade e agregação

```bash
python -m pytest tests/test_lda_transformation.py
python -m pytest tests/test_wccn_transformation.py
python -m pytest tests/test_cross_condition_plda.py
python -m pytest tests/test_shrinkage_two_covariance_plda.py
python -m pytest tests/test_embedding_aggregation_instruction.py
python -m pytest tests/test_embedding_aggregation_semantics.py
python -m pytest tests/test_aggregated_template_label.py
python -m pytest tests/test_embeddings_complete_rule.py         # face sem embedding barra o cálculo de escores
```

## Calibração, arquiteturas LR e monotonicidade

```bash
python -m pytest tests/test_identity_disjoint_pair_kfold.py     # splitter STRICT_IDENTITY
python -m pytest tests/test_logreg_calibration_strict_identity_multivariate.py
python -m pytest tests/test_calibration_metadata_service.py
python -m pytest tests/test_degenerate_score_distribution.py
python -m pytest tests/test_lr_enums.py
python -m pytest tests/test_lr_feature_terms.py
python -m pytest tests/test_lr_architecture_spec.py
python -m pytest tests/test_lr_architecture_state_factory.py
python -m pytest tests/test_lr_architecture_vectorization_dto.py
python -m pytest tests/test_lr_architecture_vectorization_executor.py
python -m pytest tests/test_lr_raw_input_payload.py
python -m pytest tests/test_architecture_feature_builder.py
python -m pytest tests/test_architecture_state_compat_service.py
python -m pytest tests/test_per_fold_transformer.py
python -m pytest tests/test_one_face_family.py
python -m pytest tests/test_s0_resolution_service.py
python -m pytest tests/test_score_interaction_feature.py
python -m pytest tests/test_rarity_transform_registry.py
python -m pytest tests/test_conditional_monotonicity_rule.py
python -m pytest tests/test_piecewise_linear_derivative_provider.py
```

## Tipicidade

```bash
python -m pytest tests/test_knn_typicality_model.py
python -m pytest tests/test_knn_typicality_integration.py
python -m pytest tests/test_typicality_protocol_pr1b.py
python -m pytest tests/test_typicality_model_training_pr2.py
python -m pytest tests/test_typicality_pair_scoring_pr3.py
python -m pytest tests/test_typicality_pr4.py
python -m pytest tests/test_inverse_identity_frequency_pr5.py
python -m pytest tests/test_typicality_pair_scoring_instruction.py
python -m pytest tests/test_typicality_oof_predictive_metrics.py
python -m pytest tests/test_movmf_external_review_p0.py         # e os demais test_movmf_external_review_*.py
```

## Detecção de face: giro, revisão do que não foi detectado, retentativa

```bash
python -m pytest tests/test_image_rotation_geometry.py          # geometria conferida contra pixels
python -m pytest tests/test_rotated_detection_retry.py          # retentativa girando, na extração crua
python -m pytest tests/test_enrichment_undetected.py            # orientação conhecida primeiro (enriquecimento)
python -m pytest tests/test_undetected_set_aside.py             # separar antes dos estágios e da gravação
python -m pytest tests/test_undetected_review.py                # política de inclusão + janela
python -m pytest tests/test_undetected_review_wiring.py
python -m pytest tests/test_undetected_retry.py                 # escalada de det_size
python -m pytest tests/test_undetected_retry_ui.py              # bloqueio que levanta ao recuperar
python -m pytest tests/test_recovered_faces_stages.py           # estágios pendentes das recuperadas
```

## Geometria da face

```bash
python -m pytest tests/test_face_geometry_persistence.py        # bbox/keypoints no parquet + invariante
python -m pytest tests/test_face_geometry_backfill.py           # passada que preenche sem refazer o dataset
python -m pytest tests/test_face_image_materialization.py
python -m pytest tests/test_oriented_image_display.py           # interface exibe a foto com a orientação EXIF, como o OpenCV a vê
```

## Metadados de captura (foto e vídeo) e taxa de compressão

```bash
python -m pytest tests/test_exif_extraction_service.py          # extração, caminho acentuado no Windows
python -m pytest tests/test_exif_detail_tab.py
python -m pytest tests/test_exif_ui_controls.py
python -m pytest tests/test_metadata_dialog_provenance.py
python -m pytest tests/test_video_capture_metadata.py           # data por quadro, fuso, compressão
python -m pytest tests/test_video_metadata_flow.py              # rotação do contêiner, manifesto, colunas
python -m pytest tests/test_jpeg_compression.py                 # qualidade recuperada de valores conhecidos
```

## OFIQ, GPU e ferramentas externas

```bash
python -m pytest tests/test_ofiq_measure_names.py               # CSV real da v1.0.3 e da v1.2.0, mesma imagem
python -m pytest tests/test_ofiq_provenance.py                  # Version.txt, impressão digital dos modelos, versões misturadas
python -m pytest tests/test_ofiq_staging_extensions.py          # cópia ao OFIQ com extensão que ele aceita (.jfif, .JPG, .bmp)
python -m pytest tests/test_image_file_extensions.py
python -m pytest tests/test_cuda_runtime_bootstrap.py           # DLLs do CUDA no Windows
```

## Caminhos de imagem, layout e colunas do dataset

```bash
python -m pytest tests/test_dataset_concat_paths.py             # "Juntar" preserva a resolução das imagens
python -m pytest tests/test_localization_anchor.py              # "Modificar Caminhos" ancora de imediato
python -m pytest tests/test_relative_path_root_on_save_as.py    # nenhum dataset RELATIVE fora da pasta das imagens
python -m pytest tests/test_self_contained_layout.py            # dataset autocontido em pasta própria
python -m pytest tests/test_prepare_dataset_for_destination.py
python -m pytest tests/test_path_root_validation.py
python -m pytest tests/test_dataset_path_rebase_service.py
python -m pytest tests/test_dataset_localization_copy_subdir.py
python -m pytest tests/test_import_copy_images_anchor.py
python -m pytest tests/test_embedding_dataset_path_anchor.py
python -m pytest tests/test_dataset_image_kind_flows.py
python -m pytest tests/test_new_dataset_storage_mode.py
python -m pytest tests/test_template_column_types.py            # coluna de template nasce com o tipo escolhido (inteiro com vazio inclusive)
python -m pytest tests/test_dataset_column_round_trip.py
python -m pytest tests/test_display_dataframe_roundtrip.py
python -m pytest tests/test_embedding_dataset_parquet_roundtrip.py
python -m pytest tests/test_face_info_embedding_columns.py
python -m pytest tests/test_dir_template.py
python -m pytest tests/test_target_assignment.py
python -m pytest tests/test_sync_apply_inplace.py
```

## Enriquecimento

```bash
python -m pytest tests/test_enrich_attribute_only.py
python -m pytest tests/test_enrich_dataset_dialog.py
python -m pytest tests/test_enrich_sequential_models.py
```

## Pipeline de imagens

```bash
python -m pytest tests/test_image_pipeline_self_contained.py    # Processar cria dataset autocontido na pasta escolhida
python -m pytest tests/test_image_pipeline_carries_pose.py      # Processar leva yaw/pitch/roll; FFInfo só com pose não engana o enriquecimento nem some ao reabrir
python -m pytest tests/test_image_pipeline_face_geometry.py     # passos recebem a geometria da face; formulário declarativo; sorteio reprodutível
python -m pytest tests/test_image_pipeline_copy_steps.py        # passos de impressão/cópia (redimensionar por IPD, curva de tons, toner, meio-tom)
python -m pytest tests/test_image_pipeline_questioned_profile.py  # questionada como alvo: medidas, perfil no pipeline, manifesto/custódia, editor
python -m pytest tests/test_image_pipeline_agreement.py         # concordância IMAGEM A IMAGEM (não só a mediana); amostra do editor = conta do Processar
python -m pytest tests/test_image_pipeline_fit.py               # curva por percentis da face, IPD da questionada, ajuste automático (treino × validação), modelo pronto
python -m pytest tests/test_image_pipeline_texture_by_tone.py   # textura por faixa de tom (meio-tom × ruído × liso); perfil antigo completado só com SHA igual
```

## Vídeo

```bash
python -m pytest tests/test_video_crop_biometrics.py            # biometria da face alinhada da extração, sem redetectar
python -m pytest tests/test_realign_from_crop_keypoints.py      # recorte cobre a janela + realinhamento pelos pontos
python -m pytest tests/test_video_frame_extraction.py
python -m pytest tests/test_video_keypoints.py
python -m pytest tests/test_video_run_importer.py
python -m pytest tests/test_video_dataset_finalization.py
python -m pytest tests/test_video_end_to_end.py
python -m pytest tests/test_video_review_service.py
python -m pytest tests/test_video_anchor_match_mode.py
python -m pytest tests/test_video_multi_source_anchor.py
python -m pytest tests/test_video_multi_target_ui.py
python -m pytest tests/test_video_window_reorg.py
```

## Cadeia de custódia, seleção e Salvar Como

```bash
python -m pytest tests/test_save_as_audit_chain.py              # cadeia copiada (não movida) + critério do filtro
python -m pytest tests/test_save_as_options_dialog.py
python -m pytest tests/test_selection_actions.py
```

## Interface (Strategy Manager V2 e Dataset Manager)

```bash
python -m pytest tests/test_score_comparison_table.py           # seleção das linhas: caso completo vs extremos
python -m pytest tests/test_comparison_table_thumbnails.py      # miniaturas localizadas pelo face_hash
python -m pytest tests/test_face_comparison_dialog.py           # duas faces lado a lado + enquadramento
python -m pytest tests/test_results_panel_strategy_isolation.py # estratégia duplicada não herda resultados
python -m pytest tests/test_config_panel_wrapped_label_height.py  # WrappedLabel em coluna estreita
python -m pytest tests/test_plot_axis_labels.py
python -m pytest tests/test_shared_histogram_bins.py
python -m pytest tests/test_dataset_gallery_toolbar.py
python -m pytest tests/test_new_dataset_dialog.py
python -m pytest tests/test_modify_paths_dialog.py
python -m pytest tests/test_extract_face_images_dialog.py
```
