# TS Whisper

Speech-to-text на нативном движке OpenAI Whisper, общем с голосом TS Super Prompt (общие веса + кэш моделей в памяти). Выбор модели: **Whisper large-v3** (лучшее качество) или **large-v3-turbo** (быстрее). Сразу три формата: SRT (тайм-коды), plain text, TTML; таймкоды по сегментам или словам, язык / перевод в английский, beam search и temperature fallbacks.

**Когда использовать:** транскрипция озвучки, генерация субтитров, выдёргивание текста из подкастов перед LLM-обработкой.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
