---
name: God files acceptable when one-node-one-file holds
description: Maintainer rejects "large file = debt" framing — only multi-class files count as structural debt
type: feedback
originSessionId: 9496b94f-83f6-408c-a5ad-387291d024d0
---
Большие файлы (1000+ LOC) НЕ считаются техдолгом сами по себе. Приоритетное правило проекта — **«одна публичная нода = один `.py`-файл»** (CLAUDE.md §7).

**Why**: maintainer явно сказал «про размеры файлов не важно» — `ts_super_prompt.py` (1705 LOC), `ts_qwen3_vl.py` (1271), `ts_whisper.py` (1079) считаются ОК, потому что каждый содержит ровно одну публичную ноду + приватные helpers внутри. Декомпозиция через `_*.py` — не приоритет.

**How to apply**:
- Не предлагать рефакторинг крупных файлов «для уменьшения LOC».
- В аудите debt-маркер «god file» применяется только когда в одном `.py` несколько публичных нод (нарушение one-node-one-file).
- Если нода содержит несколько подсистем (voice + qwen + http routes + UI events) — это нормально, пока всё это про одну публичную ноду.
- В TECH_DEBT_AUDIT.md закрывать findings F-12/13/14 как ACCEPTED с причиной «one-node-one-file rule trumps file size».
