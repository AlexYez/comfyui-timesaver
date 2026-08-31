# TS Smart Switch

Type-aware булев свитч между двумя `ANY`-входами. Выбираете `data_type` (images / video / audio / mask / string / int / float), нода валидирует, что входы соответствуют. **Auto-failover**: если выбранный вход отсутствует — fallback на другой. Идеально для опциональных веток.

**Когда использовать:** ветвление workflow по флагу или опциональный вход с разумным fallback'ом.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
