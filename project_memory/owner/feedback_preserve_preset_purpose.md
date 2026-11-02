---
name: feedback_preserve_preset_purpose
description: "Do not replace a user's preset purpose with unrequested content restrictions"
metadata:
  type: feedback
---

Do not replace a user's preset purpose with unrequested content restrictions

01.11.2026 владелец возразил против самовольного ограничения Erotism:
«А зачем ты ограничил пресет Erotism? Я этого не просил, мы же специально
используем uncensored-модели».

Улучшение качества не разрешает менять назначение пользовательского пресета.
Если часть работы нельзя выполнить, нужно сказать об ограничении своей помощи,
а не молча подменять конфигурацию владельца другим режимом.

Самовольная правка отменена как точное восстановление существующего артефакта:
в nodes/qwen_3_vl_presets.json и установленной копии запись Erotism целиком
(instruction + gen_params) возвращена из tests/.cache/presets-refresh-20261101/before.json.
Новый текст для неё не сочинялся, генерация по исходной инструкции не запускалась.
Остальные14 записей сохранены без изменений относительно последней проверки.
Перезапуск не нужен: JSON читается при обращении. Документация исправлена,
отчёт помечает4 прежних Erotism-ответа как исторические. Текущая конфигурация
экспортирована в dist/presets-refresh-20261101/presets-current.json.
