<div align="center">
<img src="icon.png" alt="Timesaver" width="100" />

# Timesaver Nodes для ComfyUI

Ноды и расширения интерфейса для изображений, видео, аудио, промптов и управления workflow.

[English](README.md) · [Версия и требования](pyproject.toml) · [Изменения](CHANGELOG.md) · [MIT](LICENSE.txt)
</div>

## За минуту

Пак загружается самим ComfyUI из `custom_nodes`. Пользователю не нужна отдельная сборка frontend. После установки найдите ноды по `TS` в поиске канваса. Отдельные ML-функции скачивают модели или требуют optional-зависимости; требования указаны в справке конкретной ноды.

| Задача | Куда перейти |
|---|---|
| Установить или запустить | [Установка и запуск](doc/operations/development.md). |
| Найти ноду и её параметры | [Каталог RU/EN](doc/nodes/README.md), [скриншоты](doc/nodes/screenshots.md). |
| Разложить граф или пересоздать ноду | [Пользовательские команды](doc/user-guide/README.md). |
| Разобраться в проекте | [Архитектура](doc/architecture/README.md). |
| Продолжить разработку | [AGENTS.md](AGENTS.md), затем [правила реализации](doc/engineering/README.md). |
| Проверить правку | [Проверки и границы CI](doc/operations/testing.md). |
| Выпустить версию | [Процедура релиза](doc/operations/release.md). |
| Найти настройки и логи | [Окружение и данные](doc/operations/environment.md). |
| Диагностировать сбой | [Таблица симптомов](doc/operations/troubleshooting.md). |
| Узнать требования и причины | [Память проекта](project_memory/MEMORY.md). |

## Установка

ComfyUI Manager → поиск `Timesaver` → Install. Или из каталога ComfyUI:

```sh
cd custom_nodes
git clone https://github.com/AlexYez/comfyui-timesaver
cd comfyui-timesaver
python -m pip install -r requirements.txt
```

Используйте Python, который запускает ComfyUI. Затем перезапустите ComfyUI. Обновление установленного клона — `git pull --ff-only`, установка обновлённых зависимостей и согласованный перезапуск. Подробнее: [порядок работы](doc/operations/development.md).

## Карта репозитория

```text
__init__.py          загрузчик и ComfyUI V3 entrypoint
nodes/              Python-ноды и общие ML/media helpers
js/                 frontend extensions и встроенная справка js/docs
locales/ru/         переводы интерфейса
example_workflows/  готовые примеры графов
doc/                тематические документы и история
project_memory/     требования владельца, решения, опыт и история
pyproject.toml      источник версии, требований и метаданных Registry
```

## Что нельзя менять незаметно

Сохранённые workflow зависят от node_id, имён и порядка входов/выходов, значений по умолчанию и категорий. Совместимость проверяют по [контрактам](tests/contracts/node_contracts.json). Дословные требования владельца находятся в [памяти требований](project_memory/owner/README.md); они важнее краткого пересказа в инструкции.

Документация и память входят в репозиторий. Токены, модели, выходные файлы и локальная тестовая лаборатория — нет. Весь указатель документации: [doc/README.md](doc/README.md).

## Лицензии и поддержка

Код пака — [MIT](LICENSE.txt). У скачиваемых весов, SDK и внешних приложений собственные условия: смотрите справку ноды и лицензии компонентов. [Источники и благодарности](doc/CREDITS.md). Мейнтейнер: [AlexYez](https://github.com/AlexYez); [issues](https://github.com/AlexYez/comfyui-timesaver/issues).

Пак не хостит и не распространяет рантайм и не аффилирован с NVIDIA. Устанавливайте только то, на что у вас есть право.

<a id="-справочник-нод"></a>
<a id="image"></a><a id="video"></a><a id="hdr"></a><a id="audio"></a><a id="llm"></a><a id="text"></a><a id="ideogram"></a><a id="files"></a><a id="loaders"></a><a id="utils"></a><a id="conditioning"></a>
Прежние разделы справочника перенесены в [каталог нод](doc/nodes/README.md).
