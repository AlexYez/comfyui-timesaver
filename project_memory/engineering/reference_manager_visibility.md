---
name: reference_manager_visibility
description: "Built-in Manager button depends on launch flag and cached system stats; helper now enables it"
metadata:
  type: reference
---

Built-in Manager button depends on launch flag and cached system stats; helper now enables it

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

01.11.2026 пользователь сообщил о периодическом исчезновении Manager справа
вверху. Он уточнил: его Launcher добавляет `--enable-manager`.
Проверка текущего /system_stats: ComfyUI 0.43.2, frontend 1.57.4;
argv содержит --use-ck-attention, --enable-manager, --port 8188.
/features extension.manager supports_v4=true, supports_csrf_post=true.
Manager установлен через pip, версия 4.2.2, git-папки Manager в custom_nodes нет.
Путь /manager/version даёт 404 — это НЕ доказательство отключения v4 Manager.

Нативный AX Chrome показал текущую страницу без Extensions. В свежей скрытой
IAB-вкладке того же адреса кнопка Extensions появилась; TS Free memory и TS
Nodes Wizard тоже загружены. Не перезагружали страницу пользователя и сервер.
В установленном frontend useManagerState вычисляет доступность из
systemStats.system.argv.includes('--enable-manager') + feature flags.
systemStatsStore получает данные при инициализации, штатного refetch при
reconnect в найденных вызовах не было. Старое состояние вкладки после
перезапуска с другими аргументами — обоснованная гипотеза, внутреннее
хранилище живой Chrome-вкладки напрямую не читали.

В старом D:/AiApps/ComfyUI/comfyui/start.bat и tools/comfy_server.py отсутствовал
--enable-manager; прежний ts_server.log также не показывал запуск Manager.
В tools/comfy_server.py добавлен --enable-manager, синтаксис и список аргументов
проверены без запуска/перезапуска. start.bat НЕ изменяли; пользователь использует
Launcher. Это наш служебный скрипт, а не авто-запуск нод пака. Нельзя впредь
перезапускать без прямой команды пользователя или терять Manager flag.

Наш frontend добавляет Free memory через actionBarButtons, её CSS скрывает
только .ts-free-memory-button. TS Sidebar меняет левые панели, не кнопку Manager.
Прямой связи нашего пакета с исчезновением Manager не найдено.
Источник: Comfy-Org/ComfyUI_frontend src/workbench/extensions/manager/composables/useManagerState.ts.
