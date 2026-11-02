---
name: reference_dlss_upscaler
description: "DLSS: текущий v9, внешний v21 и отдельные журналы старых протоколов и GPU-ошибок."
metadata:
  type: reference
---

DLSS: текущий v9, внешний v21 и отдельные журналы старых протоколов и GPU-ошибок.

Текущий backend по умолчанию — v9 через float32 host. Опция v21 запускает целое официальное приложение отдельным CLI-процессом. Новые optional-входы добавлены в конец. Skin default = 0: это не блокировка лица. Текущая справка — [TS DLSS Upscaler](../../js/docs/TS_DLSSUpscaler/ru.md); источники — [нода](../../nodes/video/dlss/ts_dlss_upscaler.py), [v21 adapter](../../nodes/video/dlss/_external_v21.py), [assets](../../nodes/video/dlss/_assets.py). Старые v5/v11 протоколы ниже — история, не инструкция менять нынешние DLL или transport.

## Темы

- [История v5: исходный протокол, раскладка, ограничения распространения и вердикты Registry.](dlss_legacy_v5.md)
- [История v9/v11: C ABI, память, битые плитки, возврат на v9 и фактура кожи.](dlss_runtime_history.md)
- [Дополнительный внешний v21: официальное приложение, CLI, форматы транспорта и проверенные ограничения.](dlss_v21.md)
