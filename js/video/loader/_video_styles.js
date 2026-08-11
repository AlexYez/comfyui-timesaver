// Стили видео-нод.
//
// ⚠️ ВНУТРИ ШАБЛОННОЙ СТРОКИ НЕЛЬЗЯ СТАВИТЬ ОБРАТНЫЕ КАВЫЧКИ — даже в
// комментарии. Строка закроется, файл перестанет разбираться, и расширение не
// загрузится ЦЕЛИКОМ (в консоли будет только невнятное «Unexpected identifier»).
// Ловит это tests/test_js_syntax.py; глаз и node --check — нет.
//
// Цвета берутся из токенов темы. Единственные исключения — то, что лежит поверх
// кадра пользователя и обязано читаться на любой картинке.

import { TS_UI_CLASS, ensureThemeStyles } from "../../_theme.js";

const STYLE_ID = "ts-video-media-styles";

export function ensureVideoStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-vid{display:flex;flex-direction:column;gap:4px;height:100%;box-sizing:border-box;
    padding:2px;overflow:hidden;font-size:var(--ts-fs-sm)}
/* Перенос обязателен: в ноде минимальной ширины десяток кнопок в одну строку
   не влезает, и часть из них оказывается за краем — нажать их нечем. */
.ts-vid__bar{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:4px;
    min-height:28px;padding:2px 4px}
.ts-vid__name{flex:1 1 60px;min-width:0;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;color:var(--ts-muted)}
.ts-vid__spacer{flex:1 1 auto}

/* ⚠️ Сцена — единственный эластичный ряд. Все полосы контролов фиксированы,
   поэтому при коротком слоте сожмётся картинка, а не уедут кнопки. */
/* Сцена и таймлайн ДЕЛЯТ свободную высоту (2:1). Фиксированная высота
   таймлайна клипала статус-строку: классический рендерер выдаёт виджету
   заметно меньше, чем Vue (замерено: 315 против 376 при одинаковой ноде). */
.ts-vid__stage{position:relative;flex:2 1 auto;min-height:80px;
    background:#000;border-radius:var(--ts-radius-sm);overflow:hidden}
/* ⚠️ Видео строго absolute: в потоке его собственная высота становится
   min-content контейнера, и в Nodes 2.0 нода растёт вниз бесконечно. */
.ts-vid__video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;
    display:block;background:#000}
.ts-vid__empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    text-align:center;padding:12px;color:var(--ts-muted);font-size:var(--ts-fs-sm);
    line-height:1.5;pointer-events:none;white-space:pre-line}
.ts-vid__badge{position:absolute;left:6px;bottom:6px;padding:2px 6px;border-radius:4px;
    font-size:var(--ts-fs-xs);font-variant-numeric:tabular-nums;pointer-events:none;
    /* Поверх кадра пользователя — подложка обязана читаться на любой картинке. */
    background:rgba(0,0,0,.55);color:#f0f0f2}

.ts-vid__transport{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:4px;
    min-height:28px;padding:0 2px}
.ts-vid__time{font-variant-numeric:tabular-nums;color:var(--ts-muted);white-space:nowrap}
/* Кнопки в полосах — компактные: при ширине ноды по минимуму их полтора
   десятка, и штатные 28 px не оставляют места под поля таймкода. */
.ts-vid__bar .ts-ui-btn--icon,.ts-vid__transport .ts-ui-btn--icon{width:24px;height:24px;
    font-size:11px;display:inline-flex;align-items:center;justify-content:center;padding:0}
/* Значки нарисованные: цвет берут из кнопки, поэтому активное состояние и
   наведение работают сами.
   ⚠️ fill и stroke задаются ЗДЕСЬ, а не атрибутами в разметке: атрибут — это
   представление, и любое чужое правило для svg внутри кнопки его перебивает,
   отчего контурные значки превращаются в сплошные пятна. */
.ts-vid .ts-ui-btn--icon svg{display:block;pointer-events:none;
    fill:none;stroke:currentColor}
.ts-vid .ts-ui-btn--icon svg *{fill:none}
.ts-vid .ts-ui-btn--icon svg .ts-ico-solid{fill:currentColor}
.ts-vid .ts-ui-btn--icon[disabled] svg{opacity:.45}
.ts-vid__bar .ts-ui-btn{min-height:24px;padding:2px 8px}
.ts-vid__tc{width:72px;text-align:center;font-variant-numeric:tabular-nums;
    background:var(--ts-sunken);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-sm);color:var(--ts-text);padding:2px 0;
    font-size:var(--ts-fs-xs);font-family:inherit}
.ts-vid__tc:focus{outline:1px solid var(--ts-accent)}

.ts-vid__timeline{position:relative;flex:1 1 auto;min-height:96px;max-height:260px;
    border-radius:var(--ts-radius-sm);overflow:hidden;background:var(--ts-sunken);
    touch-action:none}
.ts-vid__canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.ts-vid__canvas--overlay{pointer-events:none}

.ts-vid__scroll{flex:0 0 auto;height:12px;position:relative;border-radius:6px;
    background:var(--ts-sunken);cursor:pointer}
.ts-vid__thumb{position:absolute;top:2px;height:8px;min-width:28px;border-radius:4px;
    background:var(--ts-border-strong)}
.ts-vid__thumb:hover{background:var(--ts-accent)}

/* min-height:0 — чтобы пустая строка статуса схлопывалась, а не оставляла
   тёмную полосу под плеером (у сохранятеля она обычно пуста). */
.ts-vid__status{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;
    gap:8px;min-height:0;padding:0 4px;color:var(--ts-muted);font-size:var(--ts-fs-xs)}
.ts-vid__status.is-error{color:var(--ts-danger)}
.ts-vid__status span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ⚠️ Своя доля обязательна: у .ts-ui-input ширина 100%, и в гибкой полосе он
   забирает всю строку, выталкивая кнопки на второй и третий ряд. */
.ts-vid__path{flex:0 1 200px;width:auto;min-width:90px}
.ts-vid__hidden-input{position:fixed;left:-9999px;top:-9999px;opacity:0;pointer-events:none}

/* Строка результата — В ОДНУ строку с многоточием: перенос налезал на ряд
   кнопок под ней, и нажать их становилось нечем. */
.ts-vid__result{flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:2px 4px;
    color:var(--ts-muted);font-size:var(--ts-fs-xs);overflow:hidden}
.ts-vid__result span{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}
.ts-vid__result .ts-ui-btn{flex:0 0 auto;padding:2px 8px;min-height:22px}
.ts-vid__result b{color:var(--ts-text);font-weight:600}

/* ⚠️ Перемотка стоит ОТДЕЛЬНОЙ строкой во всю ширину под кадром. Рядом с
   громкостью получались два одинаковых коротких ползунка: непонятно, который из
   них что делает, и оба читались как поломка.
   ⚠️ Рамка обязательна: дорожка ползунка почти сливается с фоном ноды, и без
   неё от него остаётся один кружок бегунка. */
.ts-vid__seekrow{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:0 4px}
.ts-vid__seekrow .ts-vid__seek{flex:1 1 auto;width:auto;min-width:60px;height:16px;
    border:1px solid var(--ts-border);border-radius:8px;box-sizing:border-box}
.ts-vid__transport .ts-vid__volume{flex:0 0 72px;width:72px;height:16px;
    border:1px solid var(--ts-border);border-radius:8px;box-sizing:border-box}
`;
    document.head.appendChild(style);
}

export { TS_UI_CLASS };
