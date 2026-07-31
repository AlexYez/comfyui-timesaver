# TS Super Prompt

Нода-улучшайзер промптов со встроенной **голосовой кнопкой** — скажите идею, Whisper её транскрибирует (с грамматикой, заточенной под cinematography), маленький Qwen3 раскрывает в насыщенный промпт. Опциональный image input для image-conditioned промптов. Два режима: быстрый turbo и high-quality. Внутренности в v9.5 разнесены по `nodes/llm/super_prompt/` (`_helpers`, `_voice`, `_qwen` поверх общего Qwen-engine) — путь prompt enhancement идёт в ногу с TS Qwen 3 VL V3.

**Когда использовать:** быстрый брейншторм промптов, голосовые workflow'ы, превращение сырой идеи в production-ready промпт.


<a id="text"></a>
### 📝 Текст и промпты (4 ноды)

Сборка, рандомизация и менеджмент промптов в масштабе.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
