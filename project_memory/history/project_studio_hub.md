---
name: project_studio_hub
description: "История: TS Image Studio — точка входа во всю память и документацию подсистемы: что где лежит, с чем связано, что читать перед какой правкой"
metadata:
  type: reference
  status: historical
---

История: TS Image Studio — точка входа во всю память и документацию подсистемы: что где лежит, с чем связано, что читать перед какой правкой

TS Image Studio снята из текущего пака. Код до удаления доступен в [истории Git](https://github.com/AlexYez/comfyui-timesaver/tree/c1668b3cfa2161e36bf9b9fa91288b949b4b0b1f/nodes/image/studio); удаление зафиксировано [коммитом a2251b1](https://github.com/AlexYez/comfyui-timesaver/commit/a2251b1). Записанное решение владельца датировано 2026-09-27, коммит — 2026-09-28: это разные события.

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

# TS Image Studio — карта памяти

> ⚠️ **Снята из пака 27.09.2026** по решению владельца: идея пока не оправдала
> себя, а подписка с ключами и шифрованием могла мешать паку. Всё — код, тесты,
> инструменты, план, `ARCHITECTURE.md`, вырезанные куски общих документов и
> снимок контрактов — в `archive/removed-ts-image-studio-2026-09-27.zip` с
> `RESTORE.md`. Ссылки ниже на `nodes/image/studio/`, `js/image/studio/`,
> `js/_studio/` (кроме `_dnd.js` и `_compare.js`), `nodes/_pass.py`,
> `nodes/_studio_*.py`, `tools/studio_*.py`, `doc/TESTING_STUDIO.md` ведут в
> архив, а не в дерево пака. Лаунчер, Artius и репозитории наборов не менялись.

Самая большая подсистема пака (до снятия): полноэкранное приложение поверх ComfyUI,
которое запускает workflow-графы как бэкенды. Живёт в трёх репозиториях сразу,
поэтому эта страница — первое, что читать, и отсюда расходятся ссылки.

## Что читать перед правкой

| Собираетесь трогать | Сначала прочтите |
|---|---|
| что угодно в студии | [`nodes/image/studio/ARCHITECTURE.md`](https://github.com/AlexYez/comfyui-timesaver/blob/c1668b3cfa2161e36bf9b9fa91288b949b4b0b1f/nodes/image/studio/ARCHITECTURE.md) — файлы, путь запуска, бэкенды, грабли, точки расширения |
| интерфейс, цвета, локализацию | [`doc/THEME.md`](../../doc/THEME.md) + [reference_theme_system](../engineering/reference_theme_system.md) |
| подписку, наборы, ключи | [reference_studio_subscription](reference_studio_subscription.md) |
| проверку на живой сборке | `doc/TESTING_STUDIO.md` (историческая локальная рабочая таблица; в Git не публиковалась) |
| связь с браузером ассетов | [reference_artius_integration](../engineering/reference_artius_integration.md) |
| редактор Ideogram внутри студии | [studio-hosts-node-editors](studio-hosts-node-editors.md), [ts-ideogram-designer-node](../engineering/ts-ideogram-designer-node.md) |

## Память подсистемы

| Запись | О чём |
|---|---|
| [project_image_studio](project_image_studio.md) | утверждённый план, разрешения владельца, статус |
| [reference_studio_subscription](reference_studio_subscription.md) | пропуск, наборы, два репозитория, шифрование, режим тестирования |
| [reference_studio_persistence](reference_studio_persistence.md) | как параметры и промпт переживают вкладки, модели и перезаход |
| [studio-model-behaviour](studio-model-behaviour.md) | измеренное: какая модель для какой задачи, что делает каждый диапазон denoise |
| [studio-hosts-node-editors](studio-hosts-node-editors.md) | студия открывает чужой редактор, а не переписывает его |
| [reference_artius_integration](../engineering/reference_artius_integration.md) | реестр действий над ассетом и студийный тег на превью |
| [ts-ideogram-designer-node](../engineering/ts-ideogram-designer-node.md) | нода-дизайнер, которую студия хостит |
| [ideogram4-caption-format](../engineering/ideogram4-caption-format.md) | формат подписи Ideogram 4 |
| [project_ideogram_qwen_presets](../decisions/project_ideogram_qwen_presets.md) | пресеты Qwen, отдающие такую подпись |
| [reference_dom_widget_pitfalls](../engineering/reference_dom_widget_pitfalls.md) | грабли DOM-виджетов, на которых стоит вся студия |
| [reference_live_copy_sync](../engineering/reference_live_copy_sync.md) | живая копия ≠ dev-папка, чем сверять |

## Три репозитория, одна подсистема

```
comfyui-timesaver/            сама студия: js/_studio/, js/image/studio/,
                              nodes/image/studio/, наборы читаются отсюда
comfyui-launcher/             админка автора: вкладка «Студия» управляет
                              наборами; ключи подписки общие с workflow
comfyu-artius-browser/        браузер ассетов внутри студии: библиотека,
                              кнопки S и R на превью, студийные бейджи
comfyui-timesaver-studio/     приватные исходники платных наборов (не в git пака)
ts-studio-packs/              публичная доставка: каталог и зашифрованные наборы
```

Что где описано:

- лаунчер: `comfyui-launcher/project_memory/reference_studio_tab.md` — как
  устроена вкладка «Студия», что нельзя ломать, связь с планом админ-консоли;
- Artius: `comfyu-artius-browser/project_memory/project_studio_integration.md` —
  контракт двух точек связи со стороны браузера;
- доставка наборов: `comfyui-timesaver-studio/README.md`.

## Как проверяют целиком

Порядок важен: E2E лаунчера поднимает свой ComfyUI на том же порту и гасит
живой, обрывая прогоны студии.

1. `python tools/verify_live_sync.py --sync` — иначе тест гоняет старый код;
2. `python -m pytest tests` — 1949 офлайн-тестов (2026-08-08, v10.0), из них ~230 про студию;
   GUI-часть — `tests/test_studio_gui_e2e.py`, она ничего не ставит в очередь;
3. живые генерации: все семейства × режимы через интерфейс;
4. цикл автора в админке лаунчера: ключи → сборка → публикация → установка;
5. и только потом E2E лаунчера.

Подробности и команды — `doc/TESTING_STUDIO.md`.

### Три ловушки живых прогонов

Каждая стоила часа и выглядела как поломка студии, а была особенностью:

1. **Инпэйнт открывается в Cleanup.** Мазок сразу уходит в LaMa, вернувшаяся
   картинка обнуляет маску — и запуск отвечает «Paint a mask first» на честном
   мазке. Сперва **Repaint**. Подробности: [studio-model-behaviour](studio-model-behaviour.md).
2. **Порядок отказов.** Без промпта до маски дело не доходит вовсе, поэтому
   «жалобы на маску нет» ≠ «маска есть».
3. **Консоль ComfyUI ругается сама** при каждой загрузке. Проверять надо
   «студия не добавила своих ошибок», а не «ошибок нет».

## Инварианты, которые ломать нельзя

1. **Формат набора реализован один раз** — `tools/studio_pack.py`. Админка
   лаунчера его запускает, а не повторяет.
2. **Заперто отсутствующее, а не платное.** Установленный набор работает без
   ключа и после конца подписки.
3. **Бэкенд — это workflow-файл.** Новая модель = новый JSON плюс строка в
   `index.json`; трогать `_app.js` для этого не надо.
4. **Значение контрола переживает пересборку деки** — см.
   [reference_studio_persistence](reference_studio_persistence.md); порядок «прочитать память → создать
   контрол» обратный тому, что кажется естественным.
5. **Одна публикация.** Наборы уезжают вместе с workflow и ключами из админки
   лаунчера, чтобы подписчик получал согласованное состояние.
6. **Где лежит файл и как его находят — разное.** Работы сохраняются в
   `output/images/<семейство>/` для человека, а сессия узнаётся по снимку
   запуска. Менять раскладку, не тронув поиск, безопасно ровно до перезагрузки
   страницы — так галерея молча пустела до 2026-08-04.

Связано: [project_overview](../decisions/project_overview.md), [reference_tests](../engineering/reference_tests.md), [reference_verification](../engineering/reference_verification.md).
