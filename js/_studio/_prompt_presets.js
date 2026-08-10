// TS Studio kit — библиотека готовых промптов под режим (ui-kit layer).
//
// Данные, а не поведение: список знает только «какой текст под какую задачу».
// Кто его покажет (панель инструментов промпта) и когда — решают выше.
//
// ── почему тексты именно такие ────────────────────────────────────────────── //
//
// Пресеты написаны под Refine — доводку того, что уже нарисовано, на низком
// denoise (0.1–0.45 по слайдеру силы). Это другой жанр промпта, чем при
// генерации с нуля:
//
//   • Описываем ТО, ЧТО ЕСТЬ, и добавляем слова про фактуру. На малом denoise
//     модель не сочиняет объект заново, а переписывает пиксели — просьба
//     «beautiful woman in a red dress» здесь просто нечего делать.
//   • Просим сохранить личность, позу и свет. Без этого доводка лица уводит
//     черты, и человек на картинке перестаёт быть собой.
//   • Никаких «masterpiece, 8k, trending on artstation». Мусорные усилители
//     ничего не добавляют к куску размером с ладонь и только разбавляют
//     конкретные слова про фактуру.
//   • Анатомические перечисления («five fingers», «one thumb») оставлены там,
//     где ошибка стоит дорого: руки и ноги — самая частая причина, по которой
//     кадр вообще открывают в инпэйнте.
//
// Тексты — по-английски: их читает модель, а не человек. Переводятся названия.

/** Готовые промпты для доводки существующих объектов (режим Refine). */
export const INPAINT_REFINE_PRESETS = [
    {
        id: "people",
        label: { en: "People", ru: "Люди" },
        items: [
            {
                id: "woman-face",
                label: { en: "Detailed Woman Face", ru: "Детальное женское лицо" },
                prompt: "a detailed woman's face, natural skin texture with fine pores, "
                    + "soft realistic shading, sharp clear eyes with detailed iris, "
                    + "natural eyebrows and eyelashes, subtle lip texture, "
                    + "keep the same identity, expression and lighting, photographic detail",
            },
            {
                id: "man-face",
                label: { en: "Detailed Man Face", ru: "Детальное мужское лицо" },
                prompt: "a detailed man's face, natural skin texture with fine pores and "
                    + "light stubble, defined jawline, sharp clear eyes with detailed iris, "
                    + "natural eyebrows and eyelashes, keep the same identity, expression "
                    + "and lighting, photographic detail",
            },
            {
                id: "child-face",
                label: { en: "Detailed Child Face", ru: "Детальное детское лицо" },
                prompt: "a detailed child's face, smooth soft skin with natural fine texture, "
                    + "round facial proportions, bright clear eyes with detailed iris, "
                    + "soft eyelashes, keep the same identity, expression and lighting, "
                    + "photographic detail",
            },
            {
                id: "eyes",
                label: { en: "Detailed Eyes", ru: "Детальные глаза" },
                prompt: "sharply detailed eyes, clear iris pattern with fine fibers, "
                    + "round well-defined pupils, natural catchlight reflection, "
                    + "crisp separated eyelashes, moist eye surface, "
                    + "both eyes looking in the same direction, keep the same gaze and lighting",
            },
            {
                id: "lips-teeth",
                label: { en: "Detailed Lips & Teeth", ru: "Детальные губы и зубы" },
                prompt: "natural lip texture with fine vertical lines and subtle moisture, "
                    + "clean evenly shaped teeth with natural separation and a soft gum line, "
                    + "keep the same expression, mouth shape and lighting",
            },
            {
                id: "skin",
                label: { en: "Natural Skin Texture", ru: "Естественная кожа" },
                prompt: "realistic skin texture with visible pores and fine vellus hair, "
                    + "subtle natural imperfections, soft subsurface glow, "
                    + "no plastic smoothing, no waxy sheen, "
                    + "keep the same skin tone, shape and lighting",
            },
            {
                id: "hair",
                label: { en: "Detailed Hair", ru: "Детальные волосы" },
                prompt: "individual hair strands with natural flyaways, soft specular "
                    + "highlights along the strands, realistic hair roots and parting, "
                    + "keep the same hairstyle, color and lighting direction",
            },
            {
                id: "hand",
                label: { en: "Detailed Hand", ru: "Детальная рука" },
                prompt: "a natural human hand, exactly five fingers with one thumb, "
                    + "anatomically correct proportions, defined knuckles and finger joints, "
                    + "natural fingernails, realistic skin texture, "
                    + "keep the same pose, gesture and lighting",
            },
            {
                id: "feet",
                label: { en: "Detailed Feet", ru: "Детальные ступни" },
                prompt: "a natural human foot, exactly five toes, anatomically correct "
                    + "proportions, defined ankle and arch, natural toenails, "
                    + "realistic skin texture, keep the same pose and lighting",
            },
            {
                id: "body",
                label: { en: "Detailed Body & Pose", ru: "Детальное тело и поза" },
                prompt: "anatomically correct body proportions, natural muscle and bone "
                    + "structure under the skin, realistic skin texture, "
                    + "keep the same pose, silhouette and lighting",
            },
        ],
    },
    {
        id: "materials",
        label: { en: "Materials", ru: "Материалы" },
        items: [
            {
                id: "fabric",
                label: { en: "Fabric & Clothing", ru: "Ткань и одежда" },
                prompt: "detailed fabric weave, natural folds and wrinkles with soft shadows "
                    + "inside the folds, visible stitching and seams, realistic material sheen, "
                    + "keep the same garment shape, color and lighting",
            },
            {
                id: "metal",
                label: { en: "Metal & Jewelry", ru: "Металл и украшения" },
                prompt: "polished metal with crisp specular highlights, fine engraving and "
                    + "surface detail, realistic reflections of the surroundings, "
                    + "clean gemstone facets, keep the same shape, material and lighting",
            },
            {
                id: "eyewear",
                label: { en: "Glasses & Eyewear", ru: "Очки" },
                prompt: "clean eyeglass frame with even thickness, clear lenses with subtle "
                    + "reflections, correct hinge and temple detail, eyes visible through the "
                    + "lenses, keep the same frame shape, position and lighting",
            },
            {
                id: "text",
                label: { en: "Sharp Text & Logo", ru: "Чёткий текст и логотип" },
                prompt: "crisp legible lettering with clean sharp edges, even stroke weight "
                    + "and consistent letter spacing, correct alignment on the surface, "
                    + "keep the same wording, font style, color and perspective",
            },
            {
                id: "food",
                label: { en: "Food & Drink", ru: "Еда и напитки" },
                prompt: "appetizing food texture with natural moisture and highlights, "
                    + "crisp surface detail, realistic crumbs and steam, fresh natural color, "
                    + "keep the same dish, arrangement and lighting",
            },
            {
                id: "product",
                label: { en: "Product Surface", ru: "Поверхность предмета" },
                prompt: "clean product surface with sharp edges and accurate material finish, "
                    + "readable label detail, realistic soft studio reflections, "
                    + "no dust or scratches, keep the same shape, branding and lighting",
            },
        ],
    },
    {
        id: "scene",
        label: { en: "Scene", ru: "Сцена" },
        items: [
            {
                id: "foliage",
                label: { en: "Foliage & Nature", ru: "Листва и природа" },
                prompt: "individual leaves and fine branches, natural depth between layers, "
                    + "realistic bark and grass texture, dappled natural light, "
                    + "keep the same plant shapes, season and lighting",
            },
            {
                id: "architecture",
                label: { en: "Architecture Detail", ru: "Детали архитектуры" },
                prompt: "clean straight architectural edges, fine brick, tile and window "
                    + "detail, correct perspective and vanishing lines, realistic material "
                    + "weathering, keep the same structure, proportions and lighting",
            },
            {
                id: "sky-water",
                label: { en: "Sky & Water", ru: "Небо и вода" },
                prompt: "natural cloud structure with soft gradients, realistic water surface "
                    + "with fine ripples and reflections, smooth tonal transitions without "
                    + "banding, keep the same time of day, color and lighting",
            },
            {
                id: "background",
                label: { en: "Clean Background", ru: "Чистый фон" },
                prompt: "clean even background with natural gradient and grain, smooth "
                    + "out-of-focus falloff, no seams or repeating patterns, "
                    + "keep the same color, depth of field and lighting",
            },
        ],
    },
    {
        id: "repair",
        label: { en: "Repair", ru: "Починка" },
        items: [
            {
                id: "sharpen",
                label: { en: "Recover Sharp Detail", ru: "Вернуть резкость" },
                prompt: "restore fine detail and crisp micro-contrast, remove blur and "
                    + "compression artifacts, clean edges without halos, "
                    + "keep the same shapes, colors and lighting",
            },
            {
                id: "denoise",
                label: { en: "Clean Noise & Artifacts", ru: "Убрать шум и артефакты" },
                prompt: "remove digital noise, color blotches and compression blocks, "
                    + "smooth even tonal transitions while preserving real texture, "
                    + "keep the same shapes, colors and lighting",
            },
            {
                id: "blemish",
                label: { en: "Remove Blemishes", ru: "Убрать дефекты кожи" },
                prompt: "clear even skin with blemishes, spots and stray hairs removed, "
                    + "natural pores and texture preserved, no plastic smoothing, "
                    + "keep the same skin tone, features and lighting",
            },
            {
                id: "seam",
                label: { en: "Blend the Seam", ru: "Спрятать стык" },
                prompt: "seamless blend with the surrounding area, matching grain, color "
                    + "temperature and sharpness, no visible edge or halo, "
                    + "continue the existing texture and lighting",
            },
            {
                id: "light",
                label: { en: "Match Light & Color", ru: "Свести свет и цвет" },
                prompt: "lighting and color that match the rest of the image, consistent "
                    + "shadow direction and softness, matching white balance and contrast, "
                    + "keep the same content and texture",
            },
        ],
    },
];

/**
 * Готовые промпты для режима, или пустой список.
 *
 * Пустой список — это «кнопки не будет»: панель инструментов не должна
 * показывать пустое окно там, где библиотеки для режима ещё нет.
 *
 * @param {string} mode Режим бэкенда из манифеста ("inpaint", "t2i", …).
 * @returns {Array<{id:string,label:object,items:Array<object>}>}
 */
export function promptPresetsFor(mode) {
    return mode === "inpaint" ? INPAINT_REFINE_PRESETS : [];
}

/** Название пресета или группы на языке интерфейса. */
export function presetLabel(entry, locale) {
    const label = entry?.label;
    if (!label) return entry?.id || "";
    if (typeof label === "string") return label;
    return label[locale] || label.en || entry.id || "";
}
