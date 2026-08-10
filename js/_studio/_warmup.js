// Чем занять человека, пока показывать ещё нечего.
//
// Между нажатием Run и первым превью проходит заметное время: грузится модель,
// читается промпт, готовится латент. Раньше рабочая область в этот момент была
// просто пустой, и единственным признаком жизни оставалась полоска в панели —
// человек не знает, идёт работа или всё повисло.
//
// КАРТИНКА ОСТАЁТСЯ НА ЭКРАНЕ. Показ ложится поверх неё: кадр уходит в
// размытие и притемняется, но никуда не девается — человек всё время видит, с
// чем идёт работа. Пустой сцене (генерация с нуля) размывать нечего, и там то
// же самое читается как затемнённый холст.
//
// ЧТО ИМЕННО ПОКАЗЫВАЕТСЯ. Кадр плавно затемняется, и поверх него встаёт
// круговой индикатор: кольцо хода, вращающаяся дуга и частицы, разлетающиеся
// от неё. Под кольцом — одна строка: чем занята машина прямо сейчас (загрузка
// модели, промпт, латент). Больше на экране нет ничего: список этапов и полоса
// хода живут в панели под кнопкой Run, а на весь кадр они читаются как мусор.
//
// Кольцо держится до ПЕРВОГО КАДРА превью и после него плавно гаснет: дальше
// говорит сам кадр — превью в генерации, раскладка кусков в апскейле.
//
// Этапы не выдумываются: они приходят от автомата прогона (`_progress.js`), тот
// берёт их из событий ComfyUI. Первое превью — и показ уходит.
//
// ДВА ВИДА:
//
//   full     показывать нечего (генерация с чистого листа) — сцена во всю
//            область;
//   compact  на экране уже есть картинка (исходник апскейла, холст инпэйнта,
//            кадр перед расширением). Закрывать её нельзя: человек смотрит
//            именно на неё. Тогда — карточка внизу области, сцена в миниатюре.
//
// ⚠️ Никакой JS-геометрии: всё двигается средствами CSS (CLAUDE.md §12.5.3), а
// значит одинаково работает на любом зуме и не ест кадры на перерисовках.
// Уважает `prefers-reduced-motion` — глобальное правило темы гасит анимации, и
// показ остаётся информативным без движения.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-warmup-styles";

/** Порядок, в котором этапы идут в жизни. Ключи — от `_progress.stageOf`. */
export const WARMUP_STAGES = ["load", "prompt", "encode", "sample"];

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Только раскладка и движение; каждый цвет — из токенов --ts-*.
    // NOTE: no backticks in this comment — the whole stylesheet is one template
    // literal, and one backtick would end it.
    style.textContent = `
/* Поверх кадра, а не вместо него: размытие + притемнение. Картинка видна всё
   время работы — это и есть ответ на «куда делось изображение». */
/* ⚠️ Показ НИКОГДА не уходит в display:none. Переход по прозрачности не
   запускается, если элемент в тот же кадр появляется из display:none:
   браузеру не с чего начинать, и вместо плавного затемнения человек видит
   рывок. Поэтому элемент всегда в раскладке, а прячется видимостью.
   Затемнение и размытие тоже едут переходом — кадр гаснет постепенно. */
.ts-warmup{position:absolute;inset:0;z-index:7;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:18px;padding:24px;
  background:transparent;
  backdrop-filter:blur(0) saturate(1);
  -webkit-backdrop-filter:blur(0) saturate(1);
  pointer-events:none;overflow:hidden;
  visibility:hidden;opacity:0;
  transition:opacity .55s ease,background .55s ease,
             backdrop-filter .55s ease,visibility 0s linear .55s}
.ts-warmup.is-open{visibility:visible;opacity:1;
  background:var(--ts-scrim-strong);
  backdrop-filter:blur(10px) saturate(.85);
  -webkit-backdrop-filter:blur(10px) saturate(.85);
  transition-delay:0s}
/* Компактный вид: карточка внизу области, картинка под ней видна целиком.
   ⚠️ Размытие снимается ЯВНО. Пелена наверху задаёт и фон, и размытие фона;
   компактный вид отменял только фон, и картинка всё равно уходила в муть —
   в инпэйнте это прямо мешает работать: человек смотрит, что происходит с
   его маской, а видит размазанный кадр. Правило стоит ниже пелены и той же
   силы, поэтому побеждает. */
.ts-warmup[data-look="compact"]{background:none;justify-content:flex-end;
  gap:0;padding:0 16px 16px;
  backdrop-filter:none;-webkit-backdrop-filter:none}
/* На весь кадр — ТОЛЬКО название этапа. Список этапов, полоса хода и
   пояснение — это чтение, а не показ: на всю картинку они выглядят мусором, а
   ход и без того виден по полосе под кнопкой Run. В компактной карточке
   (инпэйнт) они, наоборот, единственный источник подробностей. */
.ts-warmup:not([data-look="compact"]) .ts-warmup__steps,
.ts-warmup:not([data-look="compact"]) .ts-warmup__bar,
.ts-warmup:not([data-look="compact"]) .ts-warmup__note{display:none}
.ts-warmup[data-look="compact"] .ts-warmup__note{display:none}
.ts-warmup[data-look="compact"] .ts-warmup__steps{gap:4px}

/* ── сцена этапа ─────────────────────────────────────────────────────────── */
/* Сцена — на всю область: она и есть фон происходящего, а подписи лежат
   поверх неё. В компактном виде сжимается в значок рядом с названием этапа. */
/* Коробка кольца стоит В ПОТОКЕ панели, НАД строкой этапа. ⚠️ Не фоном во всю
   область: тогда надпись ложится ровно на кольцо, и обе картинки мешают друг
   другу — видно на снимке. */
.ts-warmup__stagebox{position:relative;width:min(34vh,168px);aspect-ratio:1;
  flex:0 0 auto;pointer-events:none}
/* Смена сцены — ПЕРЕКРЁСТНЫМ переходом: новая проявляется поверх уходящей, и
   между ними нет пустого кадра. Обе лежат друг на друге, поэтому и «дырки» тут
   быть не может. */
.ts-warmup__stagebox > *{position:absolute;inset:0;
  animation:ts-warmup-enter .8s ease both}
.ts-warmup__stagebox > *.is-leaving{animation:ts-warmup-leave .8s ease forwards}
@keyframes ts-warmup-enter{from{opacity:0}to{opacity:1}}
@keyframes ts-warmup-leave{from{opacity:1}to{opacity:0}}
.ts-warmup[data-look="compact"] .ts-warmup__stagebox{width:34px;height:34px}

/* Круговой индикатор. Дуга хода — conic-gradient по настоящей доле; поверх
   неё бежит своя дуга, чтобы движение было видно и когда доля стоит на месте
   (модель читается с диска и о ходе не сообщает). */
/* Вся картина слегка наклоняется за курсором: смотреть на неё становится
   занятием, а не ожиданием. Наклон маленький — это фон работы, а не игрушка,
   которая перетягивает внимание. */
.ts-warmup__scene{transform:perspective(620px)
  rotateX(calc(var(--ts-tilt-y,0) * -16deg))
  rotateY(calc(var(--ts-tilt-x,0) * 16deg))
  translate3d(calc(var(--ts-tilt-x,0) * 14px),
              calc(var(--ts-tilt-y,0) * 14px), 0);
  transition:transform .18s ease-out}
/* Слои уезжают на разную глубину — картина «оживает» под курсором. */
.ts-warmup__glow{transform:translate3d(calc(var(--ts-tilt-x,0) * -22px),
  calc(var(--ts-tilt-y,0) * -22px),0)}
.ts-warmup__orbit{transform:translate3d(calc(var(--ts-tilt-x,0) * 9px),
  calc(var(--ts-tilt-y,0) * 9px),0)
  rotate(calc(var(--ts-tilt-x,0) * 12deg))}
/* Волны расходятся от кольца: три круга с разной задержкой. Это и есть «волновая»
   часть картины — она даёт ритм и глубину, которых одному кольцу не хватает. */
.ts-warmup__wave{position:absolute;inset:0;border-radius:50%;
  border:1px solid var(--ts-accent-line);opacity:0;
  animation:ts-warmup-ripple 3.6s ease-out infinite;
  animation-delay:var(--ts-delay,0s)}
@keyframes ts-warmup-ripple{
  0%{transform:scale(.72);opacity:0}
  18%{opacity:.55}
  100%{transform:scale(2.1);opacity:0}}
/* Внешняя пунктирная орбита: медленно вращается в другую сторону. */
.ts-warmup__orbit{position:absolute;inset:-14%;border-radius:50%;
  border:1px dashed var(--ts-border-strong);opacity:.55;
  animation:ts-warmup-spin 26s linear infinite reverse}
/* Внутреннее кольцо-эхо: дышит в такт работе. */
.ts-warmup__echo{position:absolute;inset:16%;border-radius:50%;
  border:1px solid var(--ts-accent-line);
  animation:ts-warmup-breathe 3.4s ease-in-out infinite}
@keyframes ts-warmup-breathe{0%,100%{transform:scale(.94);opacity:.35}
  50%{transform:scale(1.04);opacity:.8}}
/* Мягкое свечение под кольцом — глубина без единого лишнего элемента. */
.ts-warmup__glow{position:absolute;inset:-30%;border-radius:50%;
  background:radial-gradient(circle,var(--ts-accent-soft) 0%,transparent 62%);
  animation:ts-warmup-glow 4.6s ease-in-out infinite}
@keyframes ts-warmup-glow{0%,100%{opacity:.35;transform:scale(.92)}
  50%{opacity:.7;transform:scale(1.06)}}
/* Кольцо: дуга хода по настоящей доле. Середина ВЫРЕЗАНА маской, а не закрыта
   диском — под кольцом виден сам кадр, и оно не выглядит тёмной нашлёпкой. */
.ts-warmup__ring{position:absolute;inset:0;border-radius:50%;
  background:conic-gradient(var(--ts-accent) calc(var(--ts-ring,0) * 1turn),
                            var(--ts-border-soft) 0);
  /* В маске важна только непрозрачность, а не цвет: currentColor берётся из
     темы и не заводит в файл ещё одну константу. */
  -webkit-mask:radial-gradient(circle,transparent 62%,currentColor 63%);
  mask:radial-gradient(circle,transparent 62%,currentColor 63%);
  transition:background .5s ease}
/* Бегущая дуга поверх кольца. */
/* Бегущая дуга — соседом кольца, вне его маски. */
.ts-warmup__spin{position:absolute;inset:-3%;border-radius:50%;
  border:3px solid transparent;border-top-color:var(--ts-accent);
  animation:ts-warmup-spin 1.6s linear infinite}
@keyframes ts-warmup-spin{to{transform:rotate(360deg)}}
/* Частицы разлетаются от кольца наружу. Каждая — своя орбита и своя задержка,
   поэтому поток выглядит живым, а не строем. */
.ts-warmup__spark{position:absolute;left:50%;top:50%;
  width:var(--ts-size,5px);height:var(--ts-size,5px);
  margin:calc(var(--ts-size,5px) / -2) 0 0 calc(var(--ts-size,5px) / -2);
  border-radius:999px;background:var(--ts-accent);
  transform:rotate(var(--ts-angle,0deg)) translateX(var(--ts-orbit,60px));
  animation:ts-warmup-spark var(--ts-life,3.4s) ease-out infinite;
  animation-delay:var(--ts-delay,0s);opacity:0}
@keyframes ts-warmup-spark{
  0%{opacity:0;transform:rotate(var(--ts-angle,0deg)) translateX(var(--ts-orbit,60px)) scale(.4)}
  25%{opacity:.9}
  100%{opacity:0;
       transform:rotate(calc(var(--ts-angle,0deg) + 40deg))
                 translateX(calc(var(--ts-orbit,60px) * 2.4)) scale(.2)}}
/* Сцена держит кольцо по центру кадра. */
.ts-warmup__scene{position:absolute;inset:0;display:flex;
  align-items:center;justify-content:center}

/* ── панель этапов ───────────────────────────────────────────────────────── */
/* Панель появляется ПОСЛЕ затемнения, а не вместе с ним: сначала кадр уходит в
   тень, и только потом на нём проступает кольцо. Одновременный вход читается
   как вспышка. */
.ts-warmup__panel{position:relative;display:flex;flex-direction:column;
  align-items:center;gap:18px;max-width:100%;opacity:0}
/* Анимация висит на ОТКРЫТОМ показе: класс добавляется каждый раз заново, и
   каждый прогон входит одинаково. На самой панели она отыграла бы один раз за
   жизнь страницы — при первой же загрузке, когда показ ещё скрыт. */
.ts-warmup.is-open .ts-warmup__panel{animation:ts-warmup-rise .75s ease .3s both}
@keyframes ts-warmup-rise{
  from{opacity:0;transform:translateY(14px) scale(.94)}
  to{opacity:1;transform:none}}
.ts-warmup[data-look="compact"] .ts-warmup__panel{gap:8px;padding:10px 16px 12px;
  border:1px solid var(--ts-border);border-radius:var(--ts-radius-lg);
  background:var(--ts-scrim-strong);box-shadow:var(--ts-shadow);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.ts-warmup__row{display:flex;flex-direction:column;align-items:center;gap:14px}
.ts-warmup[data-look="compact"] .ts-warmup__row{flex-direction:row;gap:12px}
/* Крупная надпись по центру: она называет этап словами и читается с двух
   метров. Тень — чтобы не потеряться на светлом кадре под ней. */
/* Надпись — лёгкая и просторная: она подписывает картину, а не спорит с ней.
   Жирное начертание на весь кадр читалось как баннер. */
.ts-warmup__title{position:relative;font-size:clamp(19px,2.6vw,34px);font-weight:300;
  color:var(--ts-on-media);letter-spacing:.14em;text-transform:uppercase;
  text-align:center;text-shadow:0 2px 20px var(--ts-scrim-strong)}
.ts-warmup[data-look="compact"] .ts-warmup__title{font-size:var(--ts-fs);
  font-weight:600;text-shadow:none}
.ts-warmup__steps{display:flex;align-items:center;gap:6px;flex-wrap:wrap;
  justify-content:center;max-width:100%}
.ts-warmup__step{display:flex;align-items:center;gap:6px;padding:4px 10px;
  border:1px solid var(--ts-border-soft);border-radius:999px;
  background:var(--ts-surface);color:var(--ts-muted);font-size:var(--ts-fs-sm);
  white-space:nowrap;transition:color .2s ease,border-color .2s ease,background .2s ease}
.ts-warmup[data-look="compact"] .ts-warmup__step{background:var(--ts-scrim);
  border-color:var(--ts-border-soft);padding:3px 8px;font-size:var(--ts-fs-xs)}
.ts-warmup__dot{width:7px;height:7px;border-radius:999px;background:var(--ts-border-strong);
  flex:0 0 auto;transition:background .2s ease}
/* Пройденный этап — спокойный, текущий — акцентный, будущий — тусклый. */
.ts-warmup__step.is-done{color:var(--ts-text);border-color:var(--ts-border)}
.ts-warmup__step.is-done .ts-warmup__dot{background:var(--ts-success)}
.ts-warmup__step.is-now{color:var(--ts-accent);border-color:var(--ts-accent-line);
  background:var(--ts-accent-soft)}
.ts-warmup__step.is-now .ts-warmup__dot{background:var(--ts-accent)}
/* Полоса хода: пока доля неизвестна — бегущий свет, дальше честное заполнение.
   Врать заполнением там, где счёта ещё нет, нельзя. */
.ts-warmup__bar{position:relative;width:min(340px,100%);height:3px;border-radius:999px;
  background:var(--ts-border-soft);overflow:hidden}
.ts-warmup[data-look="compact"] .ts-warmup__bar{width:210px}
.ts-warmup__fill{position:absolute;inset:0 auto 0 0;width:0;border-radius:999px;
  background:var(--ts-accent);transition:width .35s ease}
.ts-warmup__bar.is-idle .ts-warmup__fill{width:38%;
  animation:ts-warmup-slide 1.6s ease-in-out infinite}
@keyframes ts-warmup-slide{0%{transform:translateX(-110%)}
  100%{transform:translateX(300%)}}
.ts-warmup__note{font-size:var(--ts-fs-sm);color:var(--ts-muted);text-align:center;
  max-width:60ch}
`;
    document.head.appendChild(style);
}

/**
 * Сцены этапов: каждая — маленькая самостоятельная картинка про то, что
 * происходит именно сейчас. Общего состояния между ними нет, поэтому добавить
 * пятую — значит просто дописать сюда функцию.
 */
/** Сколько частиц кружит вокруг кольца. Больше — каша, меньше — пусто. */
const SPARKS = 26;

/**
 * Круговой индикатор: кольцо хода, бегущая дуга и частицы.
 *
 * Одна картина на все стадии до первого превью — меняется только строка под
 * ней. Разные сцены на каждую стадию мелькали бы на секундных этапах и читались
 * как сбой, а не как ход работы.
 */
function ringScene() {
    const scene = document.createElement("div");
    scene.className = "ts-warmup__scene";
    // Слои идут от дальнего к ближнему: свечение, пунктирная орбита, кольцо
    // хода, эхо внутри, бегущая дуга — и поверх всего частицы.
    for (const index of [0, 1, 2]) {
        const wave = document.createElement("div");
        wave.className = "ts-warmup__wave";
        wave.style.setProperty("--ts-delay", `${index * 1.2}s`);
        scene.appendChild(wave);
    }
    for (const layer of ["glow", "orbit", "ring", "echo", "spin"]) {
        const element = document.createElement("div");
        element.className = `ts-warmup__${layer}`;
        scene.appendChild(element);
    }
    for (let index = 0; index < SPARKS; index += 1) {
        const spark = document.createElement("div");
        spark.className = "ts-warmup__spark";
        spark.style.setProperty("--ts-angle", `${(360 / SPARKS) * index}deg`);
        spark.style.setProperty("--ts-orbit", `${44 + (index % 6) * 13}px`);
        spark.style.setProperty("--ts-size", `${4 + (index % 4) * 2}px`);
        spark.style.setProperty("--ts-life", `${2.6 + (index % 6) * 0.45}s`);
        spark.style.setProperty("--ts-delay", `${index * 0.19}s`);
        scene.appendChild(spark);
    }
    return scene;
}

/**
 * Показ ожидания для рабочей области студии.
 *
 * @param {object} [options]
 * @param {Record<string, string>} [options.labels] Подписи этапов по ключам
 *   `_progress.stageOf` — те же, что показывает панель, чтобы человек читал одно
 *   и то же в двух местах.
 * @param {string} [options.title] Строка, когда этап ещё не назван.
 * @param {string} [options.note] Пояснение под этапами (компактный вид).
 * @returns {{element: HTMLElement, show: Function, setStage: Function,
 *            setProgress: Function, setRatio: Function, look: Function,
 *            hide: Function, isOpen: Function, stage: Function}}
 */
export function createWarmup({ labels = {}, title = "", note = "" } = {}) {
    ensureStyles();

    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-warmup`;

    const stagebox = document.createElement("div");
    stagebox.className = "ts-warmup__stagebox";

    const panel = document.createElement("div");
    panel.className = "ts-warmup__panel";

    const titleText = document.createElement("div");
    titleText.className = "ts-warmup__title";
    titleText.textContent = title;

    const steps = document.createElement("div");
    steps.className = "ts-warmup__steps";
    const chips = new Map();
    for (const stage of WARMUP_STAGES) {
        const chip = document.createElement("div");
        chip.className = "ts-warmup__step";
        chip.dataset.stage = stage;
        const dot = document.createElement("span");
        dot.className = "ts-warmup__dot";
        const text = document.createElement("span");
        text.textContent = labels[stage] || stage;
        chip.append(dot, text);
        steps.appendChild(chip);
        chips.set(stage, chip);
    }

    const bar = document.createElement("div");
    bar.className = "ts-warmup__bar is-idle";
    const fill = document.createElement("div");
    fill.className = "ts-warmup__fill";
    bar.appendChild(fill);

    const noteText = document.createElement("div");
    noteText.className = "ts-warmup__note";
    noteText.textContent = note;

    // В компактном виде сцена встаёт РЯДОМ с названием этапа, а не над ним:
    // карточка обязана остаться низкой, иначе закроет то, что бережёт.
    const row = document.createElement("div");
    row.className = "ts-warmup__row";
    row.append(stagebox, titleText);

    // В полном виде на экране ТОЛЬКО название этапа поверх сцены: этапы
    // списком, полоса хода и пояснение — это чтение, а не показ, и на весь
    // кадр они выглядят мусором. Всё это остаётся в компактной карточке
    // (инпэйнт) и в панели под кнопкой Run, где ему и место.
    panel.append(row, steps, bar, noteText);
    element.append(panel);

    let current = "";
    /** Надпись раздела, если он назвал происходящее по-своему. */
    let fixedLabel = "";
    /** Таймер плавного ухода: новый показ обязан его отменить. */
    let closing = 0;
    /** Кадр анимации наклона: движение мыши приходит чаще, чем экран рисует. */
    let tilt = 0;

    /**
     * Лёгкий наклон картины за курсором.
     *
     * Слушаем на документе, а не на самом показе: у него `pointer-events:none`,
     * иначе он перехватывал бы клики по рабочей области. Считаем от центра
     * ЭКРАНА, а не элемента — замер элемента здесь ничего не уточнит, а работы
     * добавит.
     */
    function onPointer(event) {
        if (tilt) return;
        tilt = requestAnimationFrame(() => {
            tilt = 0;
            if (!element.classList.contains("is-open")) return;
            // От центра САМОГО показа: курсор у края кадра должен давать
            // полный наклон, а не малую долю от размера экрана.
            const box = element.getBoundingClientRect();
            if (!box.width || !box.height) return;
            const x = Math.max(-1, Math.min(1,
                (event.clientX - box.left) / box.width * 2 - 1));
            const y = Math.max(-1, Math.min(1,
                (event.clientY - box.top) / box.height * 2 - 1));
            element.style.setProperty("--ts-tilt-x", x.toFixed(3));
            element.style.setProperty("--ts-tilt-y", y.toFixed(3));
        });
    }
    document.addEventListener("pointermove", onPointer, { passive: true });

    function paintScene() {
        if (stagebox.children.length) return;      // одна картина на весь показ
        const next = ringScene();
        // Уходящая сцена не выбрасывается сразу: она тает под новой. Иначе
        // между этапами мелькает пустой кадр.
        for (const old of [...stagebox.children]) {
            old.classList.add("is-leaving");
            setTimeout(() => old.remove(), 850);
        }
        stagebox.appendChild(next);
    }

    /**
     * Доля выполнения, 0..1, или null — «счёта ещё нет».
     *
     * Ноль и «неизвестно» — разные вещи: в первом случае честно показываем
     * пустую полосу, во втором пускаем по ней свет. Тем же числом наполняется
     * сосуд: два разных хода на экране читались бы как две разные работы.
     */
    function setProgress(fraction) {
        const known = typeof fraction === "number" && Number.isFinite(fraction);
        bar.classList.toggle("is-idle", !known);
        fill.style.width = known
            ? `${Math.max(0, Math.min(1, fraction)) * 100}%` : "";
        // Кольцо показывает настоящую долю; пока её нет — тонкая дуга, а
        // движение даёт бегущая дуга поверх.
        element.style.setProperty("--ts-ring",
            known ? String(Math.max(0.02, Math.min(1, fraction))) : "0.04");
    }

    function setStage(stage) {
        // Этап, которого нет в списке (crop, save, other), не сбивает показ:
        // он либо мгновенный, либо уже после превью, и мигать им незачем.
        const index = WARMUP_STAGES.indexOf(stage);
        if (index < 0) return;
        paintScene();
        current = stage;
        titleText.textContent = fixedLabel || labels[stage] || title;
        WARMUP_STAGES.forEach((key, position) => {
            const chip = chips.get(key);
            chip.classList.toggle("is-done", position < index);
            chip.classList.toggle("is-now", position === index);
        });
    }

    return {
        element,
        /**
         * Показать ожидание.
         *
         * @param {object} [state]
         * @param {string} [state.stage] С какого этапа начать.
         * @param {number} [state.ratio] Пропорция считаемого кадра (ш/в).
         * @param {boolean} [state.compact] Карточкой поверх картинки.
         */
        show({ stage = WARMUP_STAGES[0], ratio = 0, compact = false,
                label = "" } = {}) {
            // Раздел может назвать происходящее по-своему: в апскейле человек
            // ждёт не «загрузку модели вообще», а начало апскейла.
            fixedLabel = label;
            if (ratio > 0) element.style.setProperty("--ts-warmup-ratio", String(ratio));
            // Вид решается ЗДЕСЬ и на всё время показа: переключение посреди
            // работы читалось бы как ещё одно событие, которого не было.
            element.dataset.look = compact ? "compact" : "full";
            // Показ мог гаснуть — отменяем уборку, иначе она оборвёт новый прогон.
            clearTimeout(closing);
            current = "";
            setStage(stage);
            setProgress(null);
            element.classList.add("is-open");
        },
        setStage,
        setProgress,
        /** Пропорция кадра, который сейчас считается. */
        setRatio(ratio) {
            if (ratio > 0) element.style.setProperty("--ts-warmup-ratio", String(ratio));
        },
        /** Каким видом показано сейчас — для тестов и для отладки. */
        look: () => element.dataset.look || "full",
        hide() {
            if (!element.classList.contains("is-open")) return;
            // Уход — тот же переход в обратную сторону: снятие класса гасит
            // прозрачность, размытие и затемнение разом, а видимость уходит уже
            // после него.
            element.classList.remove("is-open");
            clearTimeout(closing);
            closing = setTimeout(() => {
                stagebox.replaceChildren();
                current = "";
            }, 600);
        },
        isOpen: () => element.classList.contains("is-open"),
        /** Снять слушатель наклона — показ живёт столько же, сколько студия. */
        teardown() {
            document.removeEventListener("pointermove", onPointer);
            cancelAnimationFrame(tilt);
        },
        /** Что показано сейчас — для тестов и чтобы не перерисовывать зря. */
        stage: () => current,
    };
}
