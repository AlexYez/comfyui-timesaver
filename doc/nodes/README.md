# Справочник нод

Каноническая справка находится в `js/docs/<node_id>/en.md` и `ru.md`: эти же файлы открывает интерфейс ComfyUI. Здесь только навигация, без второй копии описаний.

По [снимку контрактов](../../tests/contracts/node_contracts.json) — **88 публичных нод**. Категории ниже сгруппированы по папкам исходников: у HDR категория в статическом снимке может быть `null`.

[Скриншоты](screenshots.md) · [Документация](../README.md) · [Установка](../operations/development.md)

Команды, которым не нужна отдельная нода: [Tidy up, Pack as tiles и Recreate node](../user-guide/README.md). Там же — общий формат запертых моделей и пример HDR workflow.

## Изображения

| Нода | Справка | Исходник |
|---|---|---|
| TS Auto Tile Size | [RU](../../js/docs/TSAutoTileSize/ru.md) / [EN](../../js/docs/TSAutoTileSize/en.md) | [TSAutoTileSize](../../nodes/image/ts_auto_tile_size.py) |
| TS Batch Load Image | [RU](../../js/docs/TS_BatchLoadImage/ru.md) / [EN](../../js/docs/TS_BatchLoadImage/en.md) | [TS_BatchLoadImage](../../nodes/image/ts_batch_load_image.py) |
| TS Color Match | [RU](../../js/docs/TS_Color_Match/ru.md) / [EN](../../js/docs/TS_Color_Match/en.md) | [TS_Color_Match](../../nodes/image/ts_color_match.py) |
| TS Crop To Mask | [RU](../../js/docs/TSCropToMask/ru.md) / [EN](../../js/docs/TSCropToMask/en.md) | [TSCropToMask](../../nodes/image/ts_crop_to_mask.py) |
| TS Edit Align | [RU](../../js/docs/TS_EditAlign/ru.md) / [EN](../../js/docs/TS_EditAlign/en.md) | [TS_EditAlign](../../nodes/image/ts_edit_align.py) |
| TS Film Emulation | [RU](../../js/docs/TS_Film_Emulation/ru.md) / [EN](../../js/docs/TS_Film_Emulation/en.md) | [TS_Film_Emulation](../../nodes/image/film_emulation/ts_film_emulation.py) |
| TS Get Image Megapixels | [RU](../../js/docs/TS_GetImageMegapixels/ru.md) / [EN](../../js/docs/TS_GetImageMegapixels/en.md) | [TS_GetImageMegapixels](../../nodes/image/ts_get_image_megapixels.py) |
| TS Get Image Size | [RU](../../js/docs/TS_GetImageSizeSide/ru.md) / [EN](../../js/docs/TS_GetImageSizeSide/en.md) | [TS_GetImageSizeSide](../../nodes/image/ts_get_image_size_side.py) |
| TS Image Batch Cut | [RU](../../js/docs/TS_ImageBatchCut/ru.md) / [EN](../../js/docs/TS_ImageBatchCut/en.md) | [TS_ImageBatchCut](../../nodes/image/ts_image_batch_cut.py) |
| TS Image Batch to Image List | [RU](../../js/docs/TS_ImageBatchToImageList/ru.md) / [EN](../../js/docs/TS_ImageBatchToImageList/en.md) | [TS_ImageBatchToImageList](../../nodes/image/ts_image_batch_to_list.py) |
| TS Image Depth | [RU](../../js/docs/TS_ImageDepth/ru.md) / [EN](../../js/docs/TS_ImageDepth/en.md) | [TS_ImageDepth](../../nodes/image/ts_image_depth.py) |
| TS Image List to Image Batch | [RU](../../js/docs/TS_ImageListToImageBatch/ru.md) / [EN](../../js/docs/TS_ImageListToImageBatch/en.md) | [TS_ImageListToImageBatch](../../nodes/image/ts_image_list_to_batch.py) |
| TS Image Prompt Injector | [RU](../../js/docs/TS_ImagePromptInjector/ru.md) / [EN](../../js/docs/TS_ImagePromptInjector/en.md) | [TS_ImagePromptInjector](../../nodes/image/ts_image_prompt_injector.py) |
| TS Image Resize | [RU](../../js/docs/TS_ImageResize/ru.md) / [EN](../../js/docs/TS_ImageResize/en.md) | [TS_ImageResize](../../nodes/image/ts_image_resize.py) |
| TS Image Tile Merger | [RU](../../js/docs/TS_ImageTileMerger/ru.md) / [EN](../../js/docs/TS_ImageTileMerger/en.md) | [TS_ImageTileMerger](../../nodes/image/ts_image_tile_merger.py) |
| TS Image Tile Splitter | [RU](../../js/docs/TS_ImageTileSplitter/ru.md) / [EN](../../js/docs/TS_ImageTileSplitter/en.md) | [TS_ImageTileSplitter](../../nodes/image/ts_image_tile_splitter.py) |
| TS Lama Cleanup | [RU](../../js/docs/TS_LamaCleanup/ru.md) / [EN](../../js/docs/TS_LamaCleanup/en.md) | [TS_LamaCleanup](../../nodes/image/lama_cleanup/ts_lama_cleanup.py) |
| TS Langevin Inpaint | [RU](../../js/docs/TS_LangevinInpaint/ru.md) / [EN](../../js/docs/TS_LangevinInpaint/en.md) | [TS_LangevinInpaint](../../nodes/image/ts_langevin_inpaint.py) |
| TS Matting (ViTMatte) | [RU](../../js/docs/TS_Matting_ViTMatte/ru.md) / [EN](../../js/docs/TS_Matting_ViTMatte/en.md) | [TS_Matting_ViTMatte](../../nodes/image/ts_matting_vitmatte.py) |
| TS Model Resize | [RU](../../js/docs/TS_ModelResize/ru.md) / [EN](../../js/docs/TS_ModelResize/en.md) | [TS_ModelResize](../../nodes/image/ts_model_resize.py) |
| TS Remove Background | [RU](../../js/docs/TS_BGRM_BiRefNet/ru.md) / [EN](../../js/docs/TS_BGRM_BiRefNet/en.md) | [TS_BGRM_BiRefNet](../../nodes/image/ts_bgrm_birefnet.py) |
| TS Resolution Selector | [RU](../../js/docs/TS_ResolutionSelector/ru.md) / [EN](../../js/docs/TS_ResolutionSelector/en.md) | [TS_ResolutionSelector](../../nodes/image/ts_resolution_selector.py) |
| TS Restore From Crop | [RU](../../js/docs/TSRestoreFromCrop/ru.md) / [EN](../../js/docs/TSRestoreFromCrop/en.md) | [TSRestoreFromCrop](../../nodes/image/ts_restore_from_crop.py) |
| TS SAM Media Loader | [RU](../../js/docs/TS_SAM_MediaLoader/ru.md) / [EN](../../js/docs/TS_SAM_MediaLoader/en.md) | [TS_SAM_MediaLoader](../../nodes/image/sam_media_loader/ts_sam_media_loader.py) |
| TS Smart Batch | [RU](../../js/docs/TS_SmartBatch/ru.md) / [EN](../../js/docs/TS_SmartBatch/en.md) | [TS_SmartBatch](../../nodes/image/ts_smart_batch.py) |
| TS Smart Inpaint | [RU](../../js/docs/TSSmartInpaint/ru.md) / [EN](../../js/docs/TSSmartInpaint/en.md) | [TSSmartInpaint](../../nodes/image/ts_smart_inpaint.py) |
| TS Universal Inpaint Sampler | [RU](../../js/docs/TS_UniversalInpaintSampler/ru.md) / [EN](../../js/docs/TS_UniversalInpaintSampler/en.md) | [TS_UniversalInpaintSampler](../../nodes/image/ts_universal_inpaint.py) |
| TS VAE Fix | [RU](../../js/docs/TS_VAEFix/ru.md) / [EN](../../js/docs/TS_VAEFix/en.md) | [TS_VAEFix](../../nodes/image/ts_vae_fix.py) |

## Видео и HDR

| Нода | Справка | Исходник |
|---|---|---|
| TS Animation Preview | [RU](../../js/docs/TS_Animation_Preview/ru.md) / [EN](../../js/docs/TS_Animation_Preview/en.md) | [TS_Animation_Preview](../../nodes/video/ts_animation_preview.py) |
| TS DLSS Upscaler | [RU](../../js/docs/TS_DLSSUpscaler/ru.md) / [EN](../../js/docs/TS_DLSSUpscaler/en.md) | [TS_DLSSUpscaler](../../nodes/video/dlss/ts_dlss_upscaler.py) |
| TS Frame Interpolation | [RU](../../js/docs/TS_Frame_Interpolation/ru.md) / [EN](../../js/docs/TS_Frame_Interpolation/en.md) | [TS_Frame_Interpolation](../../nodes/video/ts_frame_interpolation.py) |
| TS H3 Audio Inject | [RU](../../js/docs/TS_H3AudioInject/ru.md) / [EN](../../js/docs/TS_H3AudioInject/en.md) | [TS_H3AudioInject](../../nodes/video/ts_h3_audio_inject.py) |
| TS H3 Fast VAE Decode | [RU](../../js/docs/TS_H3FastVAEDecode/ru.md) / [EN](../../js/docs/TS_H3FastVAEDecode/en.md) | [TS_H3FastVAEDecode](../../nodes/video/latent_upscale/ts_h3_fast_vae.py) |
| TS H3 Regenerate | [RU](../../js/docs/TS_H3Regenerate/ru.md) / [EN](../../js/docs/TS_H3Regenerate/en.md) | [TS_H3Regenerate](../../nodes/video/latent_upscale/ts_h3_regenerate.py) |
| TS H3 Regenerate Model | [RU](../../js/docs/TS_H3RegenerateModel/ru.md) / [EN](../../js/docs/TS_H3RegenerateModel/en.md) | [TS_H3RegenerateModel](../../nodes/video/latent_upscale/ts_h3_regenerate.py) |
| TS LTX Final Latent Selector | [RU](../../js/docs/TS_LTXFinalLatentSelector/ru.md) / [EN](../../js/docs/TS_LTXFinalLatentSelector/en.md) | [TS_LTXFinalLatentSelector](../../nodes/video/hdr/ts_ltx_final_latent_selector.py) |
| TS LTX First/Last Frame | [RU](../../js/docs/TS_LTX_FirstLastFrame/ru.md) / [EN](../../js/docs/TS_LTX_FirstLastFrame/en.md) | [TS_LTX_FirstLastFrame](../../nodes/video/ts_ltx_first_last_frame.py) |
| TS LTX HDR Decode | [RU](../../js/docs/TS_LTXHDRDecode/ru.md) / [EN](../../js/docs/TS_LTXHDRDecode/en.md) | [TS_LTXHDRDecode](../../nodes/video/hdr/ts_ltx_hdr_decode.py) |
| TS LTX HDR Guide | [RU](../../js/docs/TS_LTXHDRGuide/ru.md) / [EN](../../js/docs/TS_LTXHDRGuide/en.md) | [TS_LTXHDRGuide](../../nodes/video/hdr/ts_ltx_hdr_guide.py) |
| TS LTX HDR Settings | [RU](../../js/docs/TS_LTXHDRSettings/ru.md) / [EN](../../js/docs/TS_LTXHDRSettings/en.md) | [TS_LTXHDRSettings](../../nodes/video/hdr/ts_ltx_hdr_settings.py) |
| TS LTX HDR Stats | [RU](../../js/docs/TS_LTXHDRStats/ru.md) / [EN](../../js/docs/TS_LTXHDRStats/en.md) | [TS_LTXHDRStats](../../nodes/video/hdr/ts_ltx_hdr_stats.py) |
| TS LTX HDR VAE | [RU](../../js/docs/TS_LTXHDRVAE/ru.md) / [EN](../../js/docs/TS_LTXHDRVAE/en.md) | [TS_LTXHDRVAE](../../nodes/video/hdr/ts_ltx_hdr_vae.py) |
| TS LTX Load HDR EXR | [RU](../../js/docs/TS_LTXHDRLoadEXR/ru.md) / [EN](../../js/docs/TS_LTXHDRLoadEXR/en.md) | [TS_LTXHDRLoadEXR](../../nodes/video/hdr/ts_ltx_hdr_load_exr.py) |
| TS Latent Upscale | [RU](../../js/docs/TS_LatentUpscale/ru.md) / [EN](../../js/docs/TS_LatentUpscale/en.md) | [TS_LatentUpscale](../../nodes/video/latent_upscale/ts_latent_upscale.py) |
| TS RTX Upscaler | [RU](../../js/docs/TS_RTX_Upscaler/ru.md) / [EN](../../js/docs/TS_RTX_Upscaler/en.md) | [TS_RTX_Upscaler](../../nodes/video/ts_rtx_upscaler.py) |
| TS Video Cut | [RU](../../js/docs/TS_VideoCut/ru.md) / [EN](../../js/docs/TS_VideoCut/en.md) | [TS_VideoCut](../../nodes/video/ts_video_cut.py) |
| TS Video Depth | [RU](../../js/docs/TS_VideoDepthNode/ru.md) / [EN](../../js/docs/TS_VideoDepthNode/en.md) | [TS_VideoDepthNode](../../nodes/video/ts_video_depth.py) |
| TS Video Info | [RU](../../js/docs/TS_VideoInfo/ru.md) / [EN](../../js/docs/TS_VideoInfo/en.md) | [TS_VideoInfo](../../nodes/video/media/ts_video_info.py) |
| TS Video Loader | [RU](../../js/docs/TS_VideoLoader/ru.md) / [EN](../../js/docs/TS_VideoLoader/en.md) | [TS_VideoLoader](../../nodes/video/media/ts_video_loader.py) |
| TS Video Saver | [RU](../../js/docs/TS_VideoSaver/ru.md) / [EN](../../js/docs/TS_VideoSaver/en.md) | [TS_VideoSaver](../../nodes/video/media/ts_video_saver.py) |

## Аудио

| Нода | Справка | Исходник |
|---|---|---|
| TS Audio Loader | [RU](../../js/docs/TS_AudioLoader/ru.md) / [EN](../../js/docs/TS_AudioLoader/en.md) | [TS_AudioLoader](../../nodes/audio/loader/ts_audio_loader.py) |
| TS Audio Preview | [RU](../../js/docs/TS_AudioPreview/ru.md) / [EN](../../js/docs/TS_AudioPreview/en.md) | [TS_AudioPreview](../../nodes/audio/loader/ts_audio_preview.py) |
| TS Audio Visualizer | [RU](../../js/docs/TS_AudioVisualizer/ru.md) / [EN](../../js/docs/TS_AudioVisualizer/en.md) | [TS_AudioVisualizer](../../nodes/audio/ts_audio_visualizer.py) |
| TS Music Stems | [RU](../../js/docs/TS_MusicStems/ru.md) / [EN](../../js/docs/TS_MusicStems/en.md) | [TS_MusicStems](../../nodes/audio/music_stems/ts_music_stems.py) |
| TS Silero TTS | [RU](../../js/docs/TS_SileroTTS/ru.md) / [EN](../../js/docs/TS_SileroTTS/en.md) | [TS_SileroTTS](../../nodes/audio/ts_silero_tts.py) |
| TS Whisper | [RU](../../js/docs/TSWhisper/ru.md) / [EN](../../js/docs/TSWhisper/en.md) | [TSWhisper](../../nodes/audio/ts_whisper.py) |

## Текст и промпты

| Нода | Справка | Исходник |
|---|---|---|
| TS Angle Select | [RU](../../js/docs/TS_AngleSelect/ru.md) / [EN](../../js/docs/TS_AngleSelect/en.md) | [TS_AngleSelect](../../nodes/text/ts_angle_select.py) |
| TS Batch Prompt Loader | [RU](../../js/docs/TS_BatchPromptLoader/ru.md) / [EN](../../js/docs/TS_BatchPromptLoader/en.md) | [TS_BatchPromptLoader](../../nodes/text/ts_batch_prompt_loader.py) |
| TS Prompt Builder | [RU](../../js/docs/TS_PromptBuilder/ru.md) / [EN](../../js/docs/TS_PromptBuilder/en.md) | [TS_PromptBuilder](../../nodes/text/ts_prompt_builder.py) |
| TS Prompt Library | [RU](../../js/docs/TS_PromptLibrary/ru.md) / [EN](../../js/docs/TS_PromptLibrary/en.md) | [TS_PromptLibrary](../../nodes/text/ts_prompt_library.py) |
| TS Silero Stress | [RU](../../js/docs/TS_SileroStress/ru.md) / [EN](../../js/docs/TS_SileroStress/en.md) | [TS_SileroStress](../../nodes/text/ts_silero_stress.py) |
| TS Song Creator | [RU](../../js/docs/TS_SongCreator/ru.md) / [EN](../../js/docs/TS_SongCreator/en.md) | [TS_SongCreator](../../nodes/text/ts_song_creator.py) |
| TS Style Prompt Selector | [RU](../../js/docs/TS_StylePromptSelector/ru.md) / [EN](../../js/docs/TS_StylePromptSelector/en.md) | [TS_StylePromptSelector](../../nodes/text/ts_style_prompt_selector.py) |

## Утилиты

| Нода | Справка | Исходник |
|---|---|---|
| TS Batch Source | [RU](../../js/docs/TS_BatchSource/ru.md) / [EN](../../js/docs/TS_BatchSource/en.md) | [TS_BatchSource](../../nodes/utils/ts_batch_source.py) |
| TS Compare | [RU](../../js/docs/TS_Compare/ru.md) / [EN](../../js/docs/TS_Compare/en.md) | [TS_Compare](../../nodes/utils/ts_compare.py) |
| TS Float Slider | [RU](../../js/docs/TS_FloatSlider/ru.md) / [EN](../../js/docs/TS_FloatSlider/en.md) | [TS_FloatSlider](../../nodes/utils/ts_float_slider.py) |
| TS Free VRAM | [RU](../../js/docs/TS_FreeVRAM/ru.md) / [EN](../../js/docs/TS_FreeVRAM/en.md) | [TS_FreeVRAM](../../nodes/utils/ts_free_vram.py) |
| TS Group Bypasser | [RU](../../js/docs/TS_GroupBypasser/ru.md) / [EN](../../js/docs/TS_GroupBypasser/en.md) | [TS_GroupBypasser](../../nodes/utils/ts_group_bypasser.py) |
| TS Int Slider | [RU](../../js/docs/TS_Int_Slider/ru.md) / [EN](../../js/docs/TS_Int_Slider/en.md) | [TS_Int_Slider](../../nodes/utils/ts_int_slider.py) |
| TS Krea 2 Text Fusion | [RU](../../js/docs/TS_Krea2TextFusion/ru.md) / [EN](../../js/docs/TS_Krea2TextFusion/en.md) | [TS_Krea2TextFusion](../../nodes/utils/ts_krea2_text_fusion.py) |
| TS LoRA Loader | [RU](../../js/docs/TS_LoraLoader/ru.md) / [EN](../../js/docs/TS_LoraLoader/en.md) | [TS_LoraLoader](../../nodes/utils/ts_lora_loader.py) |
| TS LoRA Unmerged | [RU](../../js/docs/TS_LoraUnmerged/ru.md) / [EN](../../js/docs/TS_LoraUnmerged/en.md) | [TS_LoraUnmerged](../../nodes/utils/ts_lora_unmerged.py) |
| TS Math Int | [RU](../../js/docs/TS_Math_Int/ru.md) / [EN](../../js/docs/TS_Math_Int/en.md) | [TS_Math_Int](../../nodes/utils/ts_math_int.py) |
| TS NAG | [RU](../../js/docs/TS_NAG/ru.md) / [EN](../../js/docs/TS_NAG/en.md) | [TS_NAG](../../nodes/utils/ts_nag.py) |
| TS Shifted Sigmas | [RU](../../js/docs/TS_ShiftedSigmas/ru.md) / [EN](../../js/docs/TS_ShiftedSigmas/en.md) | [TS_ShiftedSigmas](../../nodes/utils/ts_shifted_sigmas.py) |
| TS Smart Switch | [RU](../../js/docs/TS_Smart_Switch/ru.md) / [EN](../../js/docs/TS_Smart_Switch/en.md) | [TS_Smart_Switch](../../nodes/utils/ts_smart_switch.py) |

## Языковые модели

| Нода | Справка | Исходник |
|---|---|---|
| TS Qwen 3 | [RU](../../js/docs/TS_Qwen3_VL_V3/ru.md) / [EN](../../js/docs/TS_Qwen3_VL_V3/en.md) | [TS_Qwen3_VL_V3](../../nodes/llm/ts_qwen3_vl.py) |
| TS RT Prompt Enhancer | [RU](../../js/docs/TS_RTPromptEnhancer/ru.md) / [EN](../../js/docs/TS_RTPromptEnhancer/en.md) | [TS_RTPromptEnhancer](../../nodes/llm/ts_rt_prompt_enhancer.py) |
| TS Super Prompt | [RU](../../js/docs/TS_SuperPrompt/ru.md) / [EN](../../js/docs/TS_SuperPrompt/en.md) | [TS_SuperPrompt](../../nodes/llm/super_prompt/ts_super_prompt.py) |
| TS Super Prompt RT | [RU](../../js/docs/TS_SuperPromptRT/ru.md) / [EN](../../js/docs/TS_SuperPromptRT/en.md) | [TS_SuperPromptRT](../../nodes/llm/super_prompt_rt/ts_super_prompt_rt.py) |

## Файлы

| Нода | Справка | Исходник |
|---|---|---|
| TS Batch Write | [RU](../../js/docs/TS_BatchWrite/ru.md) / [EN](../../js/docs/TS_BatchWrite/en.md) | [TS_BatchWrite](../../nodes/files/ts_batch_write.py) |
| TS Files Downloader | [RU](../../js/docs/TS%20Files%20Downloader/ru.md) / [EN](../../js/docs/TS%20Files%20Downloader/en.md) | [TS Files Downloader](../../nodes/files/ts_downloader.py) |
| TS YouTube Chapters | [RU](../../js/docs/TS%20Youtube%20Chapters/ru.md) / [EN](../../js/docs/TS%20Youtube%20Chapters/en.md) | [TS Youtube Chapters](../../nodes/files/ts_edl_chapters.py) |

## Загрузчики

| Нода | Справка | Исходник |
|---|---|---|
| TS Load Checkpoint | [RU](../../js/docs/TS_LockedCheckpoint/ru.md) / [EN](../../js/docs/TS_LockedCheckpoint/en.md) | [TS_LockedCheckpoint](../../nodes/loaders/ts_locked_checkpoint.py) |
| TS Load Diffusion Model | [RU](../../js/docs/TS_LockedDiffusionModel/ru.md) / [EN](../../js/docs/TS_LockedDiffusionModel/en.md) | [TS_LockedDiffusionModel](../../nodes/loaders/ts_locked_diffusion_model.py) |
| TS Load LoRA, model only | [RU](../../js/docs/TS_LockedLoraModelOnly/ru.md) / [EN](../../js/docs/TS_LockedLoraModelOnly/en.md) | [TS_LockedLoraModelOnly](../../nodes/loaders/ts_locked_lora_model_only.py) |

## Conditioning

| Нода | Справка | Исходник |
|---|---|---|
| TS Multi Reference | [RU](../../js/docs/TS_MultiReference/ru.md) / [EN](../../js/docs/TS_MultiReference/en.md) | [TS_MultiReference](../../nodes/conditioning/ts_multi_reference.py) |

## Ideogram

| Нода | Справка | Исходник |
|---|---|---|
| TS Ideogram Designer | [RU](../../js/docs/TS_IdeogramDesigner/ru.md) / [EN](../../js/docs/TS_IdeogramDesigner/en.md) | [TS_IdeogramDesigner](../../nodes/ideogram/ts_ideogram_designer.py) |
