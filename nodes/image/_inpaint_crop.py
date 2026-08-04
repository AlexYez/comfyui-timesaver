"""Перерисовка по вырезу: геометрия, проверенная в TS Smart Inpaint.

Одна и та же мысль лежит в основе любого приличного инпэйнта: не гонять через
модель весь кадр, а вырезать область маски с запасом контекста, привести её к
разрешению, на котором модель действительно рисует детали, перерисовать и
аккуратно вернуть на место. Крупный кадр так не разоряет память, а мелкая
правка получает столько пикселей, сколько ей нужно.

В `ts_smart_inpaint.py` это было сплавлено с собственным сэмплером. Здесь та
же технология живёт отдельно от того, чем именно рисуют, поэтому её может
использовать и студия — там перерисовывает LanPaint, а вырез, масштаб и
возврат остаются этими.

Что тут есть:

* хелперы, общие со Smart Inpaint (проценты→пиксели, множитель под бюджет
  мегапикселей, растушёвка, sRGB↔линейный свет, цветокоррекция шва);
* `plan_and_crop` / `paste_back` — пиксельная пара «вырезать» и «вернуть», не
  знающая ни про латенты, ни про модель.

Почему пиксели, а не латенты: у Smart Inpaint геометрия живёт в латентах, так
как он сам кодирует и декодирует. Разделённой паре VAE недоступен — между
вырезом и возвратом стоит чужой граф, — поэтому вся геометрия считается в
пикселях, а стороны выреза кратны `SNAP_PX`, чтобы любой VAE получил целые
размеры.

Инварианты: ни одна функция не мутирует вход; батч изображения сохраняется;
маска приводится к одному кадру (как в Smart Inpaint) — маска у перерисовки
одна на кадр.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import comfy.utils
import torch

RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]

# Цветокоррекция снимает сдвиг с СОХРАНЁННОГО кольца вокруг маски. Чтобы кольцо
# существовало даже при нулевом контексте, вырез растят минимум на столько.
CC_ANALYSIS_MARGIN_PX = 32

# Растушёвка и контекст задаются в процентах от собственного размера маски.
# Полученные пиксели зажимаются, чтобы края не сходили с ума: у крошечной маски
# всё равно есть слабый переход, у огромной он не разрастается без предела.
FEATHER_FLOOR_PX = 2
FEATHER_CEIL_PX = 256
CONTEXT_CEIL_PX = 1024

# Контекст в процентах от маски — правильная метрика для крупных выделений и
# бесполезная для мелких: 8% от мазка в 40 пикселей это три пикселя окружения,
# по которым модель не поймёт ни масштаба, ни освещения. Поэтому у отступа есть
# абсолютный пол: не меньше этого и не меньше доли короткой стороны кадра.
# (Практика Inpaint Crop & Stitch: относительный фактор плюс абсолютный запас;
# у автора в рабочих схемах контекст-фактор 2.0, а не дефолтные 1.2.)
CONTEXT_FLOOR_PX = 64
CONTEXT_FLOOR_IMAGE_PCT = 3.0

# Растушёвка считается от размера МАСКИ — так она остаётся соразмерной правке.
# Процент от стороны выреза был ошибкой: на вырезе в тысячу пикселей он давал
# перо под полсотни пикселей, а вместе с раздутием маски сэмплера — полосу под
# полтораста. Живьём это выглядело как размытое пятно, вылезшее далеко за
# нарисованное. Поэтому у пера есть пол (иначе на мелкой правке шов виден) и
# жёсткий потолок относительно самого выреза.
FEATHER_FLOOR_ABS_PX = 6
FEATHER_CEIL_ABS_PX = 48
FEATHER_CEIL_OF_CROP = 0.04

# На сколько расширить жёсткую маску сэмплера, чтобы полоса растушёвки легла
# внутрь перерисованного. Полторы сигмы покрывают видимую часть перехода; три
# сигмы, стоявшие раньше, заставляли модель перерисовывать втрое больше нужного.
FEATHER_GROW_FACTOR = 1.5

# Потолок увеличения мелкого выреза: без него маска в пару десятков пикселей
# раздувалась бы восьмикратно.
MAX_LINEAR = 3.0

# Кратность сторон обрабатываемого выреза. Это НЕ мелочь: у VAE свой делитель
# (замерено — `ae` и `qwen_image_vae` делят на 8, оба `flux2-vae` на 16), а
# трансформер поверх него режет латент на патчи, обычно 2x2. Промах даёт
# внутренний паддинг и лишнюю пересборку размера на возврате.
#
# Здесь лежит только значение по умолчанию для `TS Smart Inpaint`, у которого
# своя история и свои умолчания. Студия задаёт кратность на семейство: 64 как
# надёжная общая (кратна и 8, и 16, и 32), 56 у Qwen — его арифметика завязана
# на собственный энкодер и не сводится к степени двойки.
SNAP_PX = 16

# Известная область обязана занимать не меньше этой доли выреза. Правило автора
# LanPaint (scraed/LanPaint#83: «keep the known region no less than 50% of the
# picture»), и оно же лечит главную беду тугого выреза: когда маска занимает
# почти весь кадр, модели не от чего оттолкнуться, и она рисует внутри маски
# самостоятельную сцену в своём натуральном масштабе вместо продолжения того,
# что вокруг. Проверено живьём: при контексте 8% маска занимала три четверти
# выреза, и «зелёное яблоко» получалось целой картинкой внутри пятна.
KNOWN_AREA_MIN = 0.5

# При полной замене (denoise близко к единице) у модели нет пиксельной опоры
# внутри маски вообще: и масштаб, и свет она берёт только из видимого known-
# региона. Поэтому там требования жёстче — известного нужно три четверти, а
# увеличивать вырез сильнее чем вдвое вредно: чем крупнее холст, тем крупнее
# «натуральный масштаб» того, что модель на нём рисует. (Impact Pack держит
# crop_factor 3.0 — это 89% известного; ориентир по апскейлу — не более 2–3x.)
REPLACE_DENOISE = 0.6
KNOWN_AREA_MIN_REPLACE = 0.75

# При замене вырез НЕ увеличивается вовсе. Это не осторожность, а причина
# главной претензии к результату: «новый объект сильно больше и не садится в
# кадр». Модель рисует объект в масштабе, натуральном для холста, который ей
# дали. Растянув вырез вдвое, мы показываем ей вдвое больший холст — и голова
# выходит крупнее той, что была. При доработке этого не происходит: там опорой
# служат сами пиксели под маской, и масштаб задаётся ими.
#
# Уменьшать при этом по-прежнему можно: большой вырез ужимается до бюджета,
# иначе крупная маска в 8K съест всю память. Ровно так устроен и эталонный
# рабочий граф Krea 2 — вырез с фиксированным отступом, без пересчёта масштаба.
MAX_LINEAR_REPLACE = 1.0


def pad_for_known_area(mask_w: float, mask_h: float, known_min: float = KNOWN_AREA_MIN) -> float:
    """Насколько отступить от рамки маски, чтобы она заняла меньше `known_min`
    площади выреза.

    Решается точно: (w+2p)(h+2p) >= w*h / (1-known_min).
    """
    if mask_w <= 0 or mask_h <= 0:
        return 0.0
    target = mask_w * mask_h / max(1e-6, 1.0 - known_min)
    # 4p² + 2p(w+h) + wh - target = 0
    b = 2.0 * (mask_w + mask_h)
    c = mask_w * mask_h - target
    disc = b * b - 16.0 * c
    if disc <= 0:
        return 0.0
    return max(0.0, (-b + math.sqrt(disc)) / 8.0)


def pct_to_px(pct: float, base_px: float, floor_px: float, ceil_px: float) -> float:
    """Проценты от размера маски — в пиксели, с зажимом в [floor_px, ceil_px].

    `base_px` — короткая сторона маски. Один и тот же процент масштабируется
    вместе с выделением, тогда как фиксированное число пикселей было бы огромным
    для маленькой маски и незаметным для большой. Поскольку вырез всё равно
    приводится к бюджету мегапикселей, «процент от маски» — это заодно и
    «процент от обрабатываемого выреза».
    """
    return float(min(ceil_px, max(floor_px, pct / 100.0 * base_px)))


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Плотный прямоугольник ненулевой маски: (y0, y1, x0, x1) или None.

    Порог 0.01 включает растушёванный край — рамка охватывает и мягкую кромку,
    а не только двоичную середину.
    """
    m = mask[0] if mask.dim() == 3 else mask
    nz = m > 0.01
    if not nz.any():
        return None
    rows = nz.any(dim=1)
    cols = nz.any(dim=0)
    ridx = rows.nonzero(as_tuple=False).squeeze(-1)
    cidx = cols.nonzero(as_tuple=False).squeeze(-1)
    return (
        int(ridx[0].item()),
        int(ridx[-1].item()) + 1,
        int(cidx[0].item()),
        int(cidx[-1].item()) + 1,
    )


def fine_upscale_factor(
    bbox_w: float, bbox_h: float, target_mp: float, max_linear: float = MAX_LINEAR,
) -> float:
    """Во сколько раз масштабировать вырез, чтобы попасть в бюджет мегапикселей.

    Бюджет — настоящая цель, а не только нижняя граница:

    * вырез МЕНЬШЕ бюджета увеличивается к нему, но не сильнее `max_linear`;
    * вырез БОЛЬШЕ бюджета уменьшается до него, поэтому работа модели
      ограничена при любом размере маски — большое выделение в 8K больше не
      вешает машину.

    Обратно на место патч возвращается в родном размере рамки, так что
    уменьшение съедает только обрабатываемую детализацию, а не разрешение
    итогового кадра.
    """
    if bbox_w <= 0 or bbox_h <= 0:
        return 1.0
    current_mp = bbox_w * bbox_h / 1_000_000.0
    if current_mp <= 0:
        return 1.0
    return min(math.sqrt(target_mp / current_mp), max_linear)


def resize_spatial(t: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    """Сменить пространственный размер тензора одним из методов ComfyUI.

    Принимает [C,H,W], [B,C,H,W] или [1,H,W] (маска), возвращает тот же ранг.
    Идёт через `comfy.utils.common_upscale`, чтобы работали bislerp и lanczos.
    """
    method = method if method in RESIZE_METHODS else "nearest-exact"
    if t.dim() == 3:
        out = comfy.utils.common_upscale(t.unsqueeze(0), target_w, target_h, method, "disabled")
        return out.squeeze(0)
    if t.dim() == 4:
        return comfy.utils.common_upscale(t, target_w, target_h, method, "disabled")
    raise ValueError(f"resize_spatial: неожиданный ndim {t.dim()}")


def downscale_to_megapixels(
    pixels: torch.Tensor, target_mp: float, method: str
) -> torch.Tensor:
    """Уменьшить кадр (B, H, W, C) до площади в `target_mp` мегапикселей.

    Никогда не увеличивает: картинка меньше бюджета проходит нетронутой —
    растягивание не добавляет сведений. Стороны кратны 8 ради VAE.
    """
    if pixels is None or pixels.dim() != 4:
        return pixels
    h = int(pixels.shape[1])
    w = int(pixels.shape[2])
    if h <= 0 or w <= 0:
        return pixels
    budget = max(1, int(float(target_mp) * 1_000_000))
    if h * w <= budget:
        return pixels
    s = math.sqrt(budget / float(h * w))
    new_w = max(8, (int(round(w * s)) // 8) * 8)
    new_h = max(8, (int(round(h * s)) // 8) * 8)
    method = method if method in RESIZE_METHODS else "lanczos"
    chw = pixels.permute(0, 3, 1, 2)
    chw = comfy.utils.common_upscale(chw, new_w, new_h, method, "disabled")
    return chw.permute(0, 2, 3, 1).contiguous()


def grow_mask(mask: torch.Tensor, grow_px: float, blur_px: float = 0.0) -> torch.Tensor:
    """Расширить маску на `grow_px` и мягко размыть край на `blur_px`.

    Отличается от «просто размыть» тем, где оказывается переход. Размытие на
    месте съедает край закрашенного: половина рампы уходит внутрь мазка, и
    композит подмешивает туда исходные пиксели. Здесь маска сперва растёт, и
    внутри мазка альфа остаётся строго единицей — весь переход выносится в
    кольцо снаружи.

    Дилатация — максимум по окну (`max_pool2d`): дёшево и без зависимостей.
    """
    m = mask if mask.dim() == 4 else mask.unsqueeze(1) if mask.dim() == 3 else mask[None, None]
    grown = m
    radius = int(math.ceil(grow_px))
    if radius > 0:
        grown = torch.nn.functional.max_pool2d(
            grown, kernel_size=2 * radius + 1, stride=1, padding=radius)
    if blur_px > 0:
        grown = gaussian_blur_2d(grown, blur_px)
    grown = grown.clamp(0.0, 1.0)
    if mask.dim() == 3:
        return grown.squeeze(1)
    if mask.dim() == 2:
        return grown[0, 0]
    return grown


def gaussian_blur_2d(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Разделимое гауссово размытие маски [H,W], [B,H,W] или [B,C,H,W]."""
    if sigma <= 0:
        return mask
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    half = ksize // 2
    x = torch.arange(ksize, device=mask.device, dtype=torch.float32) - half
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()

    orig_ndim = mask.dim()
    if orig_ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        m = mask.unsqueeze(1)
    elif orig_ndim == 4:
        m = mask
    else:
        raise ValueError(f"gaussian_blur_2d: неожиданный ndim {orig_ndim}")

    kh = k1d.view(1, 1, 1, ksize)
    kv = k1d.view(1, 1, ksize, 1)
    m = torch.nn.functional.pad(m, (half, half, 0, 0), mode="replicate")
    m = torch.nn.functional.conv2d(m, kh)
    m = torch.nn.functional.pad(m, (0, 0, half, half), mode="replicate")
    m = torch.nn.functional.conv2d(m, kv)

    if orig_ndim == 2:
        return m.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return m.squeeze(1)
    return m


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] → линейный свет. Нужен, чтобы смешивание по растушёвке не
    темнило переход: альфа-смешивание прямо в гамме оставляет тёмную линию по
    краю маски."""
    c = c.float().clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    """Линейный свет [0,1] → sRGB, точный обратный переход. При альфе 0 и 1
    возвращает вход без изменений, поэтому меняется только мягкая полоса."""
    c = c.float().clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def color_correct_patch(
    refined: torch.Tensor,      # [B, h, w, C] — перерисованный патч
    original: torch.Tensor,     # [B, h, w, C] — тот же вырез до перерисовки
    alpha: torch.Tensor,        # [B, h, w, 1] — альфа композита, 1 = маска
    *,
    alpha_eps: float = 0.02,
    min_samples: int = 256,
    strength: float = 1.0,
) -> torch.Tensor:
    """Снять систематический сдвиг цвета, который вносит круговой рейс VAE.

    Оценка берётся ТОЛЬКО по сохранённому кольцу — пикселям, где альфа около
    нуля, то есть внутри выреза, но вне перерисованной области. Там содержимое
    одно и то же, значит разница по каналам — это ровно сдвиг VAE, а не разница
    сюжетов. Поэтому красный предмет на синем фоне остаётся красным: мы
    отменяем добавленный оттенок, а не подгоняем новое содержимое под окружение.

    По каналу считается устойчивая пара «усиление + сдвиг» (МНК по срезанным
    через MAD образцам кольца) и применяется ко всему патчу. Если кольца мало —
    возвращается вход без изменений. Результат всегда новый тензор.
    """
    if refined.shape != original.shape or refined.dim() != 4:
        return refined
    if alpha.dim() != 4 or alpha.shape[1:3] != refined.shape[1:3]:
        return refined
    if alpha.shape[0] == 1 and refined.shape[0] > 1:
        alpha = alpha.expand(refined.shape[0], -1, -1, -1)
    elif alpha.shape[0] != refined.shape[0]:
        return refined
    channels = int(refined.shape[-1])
    preserved = (alpha[..., 0] < alpha_eps).reshape(-1)
    n = int(preserved.sum().item())
    if n < min_samples:
        return refined

    ref_flat = refined.reshape(-1, channels).float()
    org_flat = original.reshape(-1, channels).float()
    ref_ring = ref_flat[preserved]
    org_ring = org_flat[preserved]

    gain = torch.ones(channels, dtype=torch.float32, device=refined.device)
    offset = torch.zeros(channels, dtype=torch.float32, device=refined.device)
    for c in range(channels):
        x = ref_ring[:, c]
        y = org_ring[:, c]
        diff = y - x
        med = diff.median()
        mad = (diff - med).abs().median()
        if mad > 0:
            keep = (diff - med).abs() <= (3.0 * 1.4826 * mad)
            x = x[keep]
            y = y[keep]
        if x.numel() < 16:
            continue
        mx = x.mean()
        my = y.mean()
        var_x = ((x - mx) ** 2).mean()
        if var_x > 1e-6:
            a = (((x - mx) * (y - my)).mean() / var_x).clamp(0.5, 2.0)
        else:
            a = torch.ones((), dtype=torch.float32, device=refined.device)
        b = (my - a * mx).clamp(-0.25, 0.25)
        gain[c] = a
        offset[c] = b

    corrected = (refined.float() * gain + offset).clamp(0.0, 1.0)
    if strength < 1.0:
        corrected = refined.float() * (1.0 - strength) + corrected * strength
    return corrected.to(refined.dtype)


# ── пиксельная пара «вырезать» и «вернуть» ───────────────────────────────── #

@dataclass(frozen=True)
class CropPlan:
    """Всё, что нужно, чтобы вернуть перерисованный вырез на место.

    Хранит рамку в пикселях исходного кадра, размер обработки и уже готовую
    растушёванную альфу этого выреза: пересчитывать её при возврате — значит
    рисковать разойтись с тем, что видела модель.
    """

    y0: int
    y1: int
    x0: int
    x1: int
    out_w: int
    out_h: int
    alpha: torch.Tensor          # [1, h, w, 1] в родном размере рамки
    feather_px: float
    context_px: float
    color_correct: bool = True

    @property
    def crop_w(self) -> int:
        return self.x1 - self.x0

    @property
    def crop_h(self) -> int:
        return self.y1 - self.y0


def normalise_mask(mask: torch.Tensor) -> torch.Tensor:
    """Маску любого принятого вида — к [1, H, W]. Перерисовка работает по одной
    маске на кадр, как и Smart Inpaint."""
    if mask.dim() == 2:
        return mask.unsqueeze(0)
    if mask.dim() == 4:
        return mask[:1, 0]
    return mask[:1]


def plan_and_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    megapixels: float,
    context_pct: float,
    feather_pct: float,
    resize_method: str = "lanczos",
    snap_px: int | None = None,
    color_correct: bool = True,
    max_linear: float | None = None,
    denoise: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CropPlan] | None:
    """Вырезать область маски с контекстом и привести её к бюджету.

    Возвращает (вырез, жёсткая маска, мягкая маска, план) либо None, если маска
    пуста — тогда перерисовывать нечего и вызывающий код обязан вернуть кадр как
    есть.

    Масок две не по забывчивости. Сэмплеру без Differential Diffusion мягкий
    край бесполезен: LanPaint бинаризует маску по 0.5 у себя внутри. Но если
    модель пропущена через Differential Diffusion, тот же мягкий край становится
    расписанием денойза по пикселям — кромка правится меньше шагов и потому
    сходится с окружением. Так собран и эталонный рабочий граф Krea 2.

    Порядок важен: сперва растушёвка, потом рамка. Рамка считается по уже
    растушёванной маске (порог 0.01), поэтому мягкая кромка целиком попадает
    внутрь выреза; иначе на возврате она обрезалась бы и давала жёсткий край.
    """
    if image.dim() != 4:
        raise ValueError(f"plan_and_crop: изображение должно быть [B,H,W,C], получено {tuple(image.shape)}")
    m = normalise_mask(mask).float().clamp(0.0, 1.0)
    img_h = int(image.shape[1])
    img_w = int(image.shape[2])
    if int(m.shape[-2]) != img_h or int(m.shape[-1]) != img_w:
        m = resize_spatial(m, img_h, img_w, "bilinear").clamp(0.0, 1.0)

    tight = mask_bbox(m)
    if tight is None:
        return None
    ty0, ty1, tx0, tx1 = tight
    mask_min_side = float(min(ty1 - ty0, tx1 - tx0))

    # Контекст: проценты от маски, но не меньше абсолютного пола и не меньше
    # того, при котором маска займёт меньше половины выреза.
    context_px = pct_to_px(float(context_pct), mask_min_side, 0.0, CONTEXT_CEIL_PX)
    floor_px = max(
        float(CONTEXT_FLOOR_PX),
        CONTEXT_FLOOR_IMAGE_PCT / 100.0 * float(min(img_h, img_w)),
    )
    pad_px = max(context_px, floor_px)
    if color_correct:
        # Кольцу для цветокоррекции нужен сохранённый запас вокруг маски.
        pad_px = max(pad_px, float(CC_ANALYSIS_MARGIN_PX))
    replacing = float(denoise) >= REPLACE_DENOISE
    known_min = KNOWN_AREA_MIN_REPLACE if replacing else KNOWN_AREA_MIN
    pad_px = max(pad_px, pad_for_known_area(float(tx1 - tx0), float(ty1 - ty0), known_min))

    pad = int(round(pad_px))
    y0 = max(0, ty0 - pad)
    y1 = min(img_h, ty1 + pad)
    x0 = max(0, tx0 - pad)
    x1 = min(img_w, tx1 + pad)
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return None

    # Растушёвка — процент от размера МАСКИ, зажатый абсолютным полом и
    # потолком, а сверху ещё ограниченный долей выреза: полоса перехода обязана
    # оставаться узкой относительно того, что перерисовывается, иначе результат
    # размазывается по окрестности вместо того, чтобы в неё вписаться.
    # Ровный ноль — это «совсем без перехода», а не «минимальный переход»: пол
    # действует только на положительные проценты. Нужно, чтобы можно было честно
    # сравнить композит с растушёвкой и без неё.
    if float(feather_pct) <= 0.0:
        feather_px = 0.0
    else:
        feather_px = pct_to_px(
            float(feather_pct), mask_min_side, FEATHER_FLOOR_ABS_PX, FEATHER_CEIL_ABS_PX,
        )
        feather_px = min(feather_px, FEATHER_CEIL_OF_CROP * float(min(y1 - y0, x1 - x0)))

    # Растушёвка идёт НАРУЖУ от мазка, а не по его краю.
    #
    # Размывая маску на месте, мы получаем рампу, половина которой лежит внутри
    # закрашенного: там композит подмешивает к перерисованному исходные пиксели
    # — и на стыке появляется мутная кайма, ровно та, на которую жаловались.
    # Поэтому маска сперва растёт на ширину пера, и только потом размывается
    # вдвое слабее: внутри мазка альфа строго единица, а весь переход лежит в
    # кольце снаружи — там, где модель под Differential Diffusion получила мало
    # шагов и воспроизвела окружение почти без изменений, так что смешивать
    # нечего и шов не виден.
    #
    # Так устроены и проверенные чужие решения: рабочий граф Krea 2 (расширение
    # 50, размытие 9.1) и Inpaint Crop & Stitch (дилатация N, размытие N/2).
    # Взятие максимума с исходным мазком — не перестраховка: у гауссианы хвост
    # бесконечный, и у самой кромки закрашенного она даёт 0.977, а не единицу.
    # Эти два процента исходных пикселей, подмешанные вплотную к краю правки,
    # и читаются как мутная кайма. С максимумом обещание выполняется буквально:
    # внутри мазка альфа ровно единица, весь переход лежит снаружи.
    soft = (torch.maximum(grow_mask(m, feather_px, blur_px=feather_px * 0.5), m)
            if feather_px > 0 else m)

    crop = image[:, y0:y1, x0:x1, :].contiguous()
    alpha = soft[:1, y0:y1, x0:x1].unsqueeze(-1).contiguous()   # [1, h, w, 1]

    cap = max_linear if max_linear is not None else (
        MAX_LINEAR_REPLACE if replacing else MAX_LINEAR)
    scale = fine_upscale_factor(float(x1 - x0), float(y1 - y0), float(megapixels), cap)
    snap = int(snap_px) if snap_px else SNAP_PX
    out_h = max(snap, int(math.ceil((y1 - y0) * scale / snap) * snap))
    out_w = max(snap, int(math.ceil((x1 - x0) * scale / snap) * snap))

    crop_chw = crop.movedim(-1, 1)
    crop_up = comfy.utils.common_upscale(
        crop_chw, out_w, out_h, resize_method if resize_method in RESIZE_METHODS else "lanczos",
        "disabled",
    ).movedim(1, -1).contiguous()

    # Маску масштабируем билинейно независимо от выбранного метода: у lanczos
    # одноканальная ветка возвращает транспонированный тензор, а bislerp —
    # сферическая интерполяция векторов, для альфы бессмысленная.
    #
    # Сэмплеру отдаём ЖЁСТКУЮ маску, а не растушёванную. LanPaint всё равно
    # бинаризует её по порогу 0.5 у себя внутри (`KSamplerX0Inpaint`), поэтому
    # мягкий край не смягчает ничего — он лишь сдвигает фактическую границу на
    # половину пера, и она перестаёт совпадать с альфой, по которой мы потом
    # вклеиваем. Мягкость нужна ровно в одном месте — в композите на возврате.
    #
    # Заодно жёсткую маску растим на ширину растушёвки: иначе внешняя половина
    # мягкой полосы подмешивает пиксели, которых модель не касалась, — а они
    # прошли круговой рейс VAE и несут его цветовой сдвиг. Вся рампа обязана
    # лежать внутри честно перерисованного.
    hard = (m[:1, y0:y1, x0:x1] >= 0.5).to(m.dtype)
    grow = int(math.ceil(FEATHER_GROW_FACTOR * feather_px))
    if grow > 0:
        # Дилатация max-пулом: дёшево и без зависимостей.
        k = 2 * grow + 1
        hard = torch.nn.functional.max_pool2d(
            hard.unsqueeze(0), kernel_size=k, stride=1, padding=grow,
        ).squeeze(0)
    mask_up = (resize_spatial(hard, out_h, out_w, "bilinear") >= 0.5).to(m.dtype)

    mask_soft = resize_spatial(
        soft[:1, y0:y1, x0:x1], out_h, out_w, "bilinear").clamp(0.0, 1.0)

    plan = CropPlan(
        y0=y0, y1=y1, x0=x0, x1=x1, out_w=out_w, out_h=out_h,
        alpha=alpha, feather_px=feather_px, context_px=context_px,
        color_correct=color_correct,
    )
    return crop_up, mask_up, mask_soft, plan


def paste_back(
    image: torch.Tensor,
    patch: torch.Tensor,
    plan: CropPlan,
    *,
    resize_method: str = "lanczos",
) -> torch.Tensor:
    """Вернуть перерисованный вырез в кадр: масштаб обратно, цвет, растушёвка.

    Вход не мутируется — композит уходит в копию кадра.
    """
    if image.dim() != 4 or patch.dim() != 4:
        raise ValueError("paste_back: и кадр, и патч должны быть [B,H,W,C]")

    patch_native = patch
    if int(patch.shape[1]) != plan.crop_h or int(patch.shape[2]) != plan.crop_w:
        patch_native = comfy.utils.common_upscale(
            patch.movedim(-1, 1), plan.crop_w, plan.crop_h,
            resize_method if resize_method in RESIZE_METHODS else "lanczos", "disabled",
        ).movedim(1, -1)

    original_crop = image[:, plan.y0:plan.y1, plan.x0:plan.x1, :]
    alpha = plan.alpha.to(device=patch_native.device, dtype=patch_native.dtype)
    if float(alpha.max()) <= 0.0:
        # Маска пуста — смешивать нечего. Возвращаем кадр как есть: прогон через
        # sRGB↔линейный обратим не побитово, и «ничего не менял» превратилось бы
        # в едва заметный сдвиг всего кадра.
        return image.clone()
    if patch_native.shape[0] != original_crop.shape[0]:
        # Сэмплер вправе вернуть один кадр на батч — тогда он ложится во все.
        patch_native = patch_native[:1].expand(original_crop.shape[0], -1, -1, -1)

    if plan.color_correct:
        patch_native = color_correct_patch(patch_native, original_crop, alpha)

    composited = linear_to_srgb(
        srgb_to_linear(patch_native) * alpha + srgb_to_linear(original_crop) * (1.0 - alpha)
    ).to(image.dtype)
    # Там, где альфа ровно ноль, оставляем исходные пиксели байт в байт. Формально
    # смешивание при alpha=0 и так возвращает оригинал, но круг sRGB↔линейный
    # обратим лишь до последнего разряда — и сохранённое кольцо вокруг маски
    # каждый проход уезжало бы на единицы младшего бита.
    composited = torch.where(alpha > 0, composited, original_crop)

    out = image.clone()
    out[:, plan.y0:plan.y1, plan.x0:plan.x1, :] = composited
    return out
