# Frontend и строковые контракты

ComfyUI импортирует ES modules из js/, а сохранённое состояние связывает UI и Python по устойчивым именам.

## Entry и private modules

WEB_DIRECTORY задаётся в [корневом загрузчике](../../__init__.py). Публичные
entry JS вызывают app.registerExtension с уникальным стабильным name; private
ES modules предоставляют функции и не должны регистрировать ноду случайным
side effect. Файлы не являются единым IIFE и не требуют npm-бандлера.
Существующая раскладка повторяет frontend feature-папки и допускает private
helpers; это не основание дробить простую ноду на десяток файлов.

Браузерные импорты /scripts/app.js и /scripts/api.js — URL host ComfyUI,
а не отсутствующие файлы этого репозитория. Проектные импорты относительны
к JS-файлу. Пример API-пары —
[Audio Loader entry](../../js/audio/loader/ts-audio-loader.js) и его
[Python-нода](../../nodes/audio/loader/ts_audio_loader.py).

## Общие владельцы

| Модуль | Контракт |
| --- | --- |
| [Тема](../../js/_theme.js) | Токены, готовые кнопки/контролы, en/ru и единый launcher. |
| [DOM sizing](../../js/_dom_widget.js) | addResizableDomWidget, hideWidget, holdSize и различия legacy/Vue. |
| [Fullscreen](../../js/_fullscreen.js) | Overlay, focus/keyAnchor, close/restore и одна кнопка закрытия. |
| [Горячие клавиши](../../js/_keys.js) | Отличать typing от команд, включая select/contenteditable. |
| [Playback guard](../../js/_media/_playback_guard.js) | Pause при execution_start и уходе медиа в фон; не auto-resume. |
| [DnD](../../js/_studio/_dnd.js) | Синхронный снимок DataTransfer до первого await. |
| [Compare helper](../../js/_studio/_compare.js) | Оставшаяся общая функция после снятия студии. |
| [Wired source](../../js/_wired_source.js) | Общая работа источника, поданного проводом. |
| [Slider helpers](../../js/utils/sliders/_slider_helpers.js) | Общая фабрика int/float-слайдеров и восстановление состояния. |

## Где живёт состояние

onNodeCreated вызывается до восстановления widgets_values. loadedGraphNode
должен читать уже восстановленные значения через rehydrate, а не пересоздавать
DOM с нуля. В Vue node.widgets и render tree имеют отдельные регистрации:
простой remove/splice не гарантирует уничтожение старого widget и приводит
к двоению интерфейса. Для hidden numeric/string полей допустим согласованный
fallback в node.properties; при чтении ноль, false и sentinel -1 не равны
«значения нет». Подробные причины и шаблоны — [DOM widgets](../engineering/dom-widgets.md).

Resize основывается на graph units node.size, а не на zoom-масштабированном
getBoundingClientRect. Pointer coordinates, напротив, приходят в viewport:
преобразование делается осознанно. Размер после configure/onExecuted сохраняет
общий holdSize, не отдельная копия таймеров в каждой ноде.

## JS ↔ Python

node_id, имена IO.Input, widget IDs, route strings, event names, payload keys
и сохранённые properties — публичная поверхность. Переименование только в одной
стороне ломает интерфейс без синтаксической ошибки. Маршруты в ряде нод
регистрируются через [make_route_registrars](../../nodes/_shared.py); изолированный
import в тесте может не иметь настоящего PromptServer и не доказывает доступность
HTTP endpoint. Получение node definitions/комбо должно учитывать текущие форматы
ComfyUI V1/V3, а не только старое list-combo представление.

User text вставляют через textContent/DOM API, не через blind innerHTML.
Слушатели документа, observers, object URLs, raf/таймеры и fullscreen handles
очищают при удалении ноды; прежние onRemoved/onResize/onConfigure вызывают
в корректной цепочке. Иначе переход между workflow оставляет старый UI активным
или незаметно возвращает defaults.

## Тема и проверка

Подключение цветов/компонентов и локализация — [theme hub](../theme/README.md).
Tooltips нативных нод EN принадлежат Python-схеме, RU — locales/ru/nodeDefs.json;
DOM-текст — словарям JS. Результат API/backend не надо переводить догадками.
Набор проверки — [testing](../operations/testing.md): compile/syntax отдельно
от реального открытия ноды в браузере. Не обещать исправленное сохранение MP4,
проверив только появление preview: файл проверяется на сервере.
