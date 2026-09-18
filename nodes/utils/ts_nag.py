"""TS NAG — negative prompt inside cross-attention, for models that run at CFG 1.

Normalized Attention Guidance (`ChenDarYen/Normalized-Attention-Guidance
<https://github.com/ChenDarYen/Normalized-Attention-Guidance>`_) applies the
negative prompt **inside every cross-attention block** instead of at the CFG
level. With the same query, attention is computed twice — once against the
positive context, once against a negative one — and the two are combined:

```
guidance = x_positive * scale - x_negative * (scale - 1)     # extrapolate
r        = ||guidance||_1 / ||x_positive||_1                 # per token
if r > tau:  guidance *= (||x_positive||_1 * tau) / ||guidance||_1
out      = guidance * alpha + x_positive * (1 - alpha)
```

The clamp by ``tau`` is what makes it usable at all: an extrapolation with a
scale of 11 would otherwise tear the activations apart. Hence *normalized*.

⚠️ **Why this exists at all.** The extra cost is one more cross-attention per
block — query from the picture against a short text context — and **not** a
second pass of the model. So a negative prompt keeps working on a distilled
model sampled at ``cfg = 1``, where ComfyUI does not compute the negative branch
at all and the negative wire is decoration.

⚠️ **Only models with a real cross-attention module can do this.** Checked
against ComfyUI's own code:

* Wan — ``blocks[i].cross_attn`` (separate T2V and I2V classes) — supported.
* LTX — ``transformer_blocks[i].attn2`` (``CrossAttention`` with a context dim)
  — supported.
* Krea 2 — ``combined = cat((context, img))`` **once, before the stack**, and
  the text tokens then evolve with the picture. A negative variant would have to
  be carried through every block, which is a second full forward — exactly what
  CFG already does. Refused with that explanation.
* MiniMax H3 — one attention over a single concatenated sequence. Same story.

⚠️ **Where the two outputs are combined differs per family, on purpose.** Wan's
cross-attention body is five lines, so it is reproduced faithfully and NAG lands
**before** the output projection, matching the upstream KJNodes node number for
number — settings people share for Wan transfer as they are. LTX's body carries
branches this node has no business duplicating (RoPE, guide masks, per-head
gating), so there the original forward is called twice and NAG lands **after**
the output projection. The projection is linear, so the extrapolation itself is
unchanged; only the norm clamp happens in output space. Values tuned for Wan
would not have transferred to a different model anyway.
"""

from __future__ import annotations

import logging
import types

import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_nag")
LOG_PREFIX = "[TS NAG]"

AUTO = "auto"

#: Семейства, у которых cross-attention нет по устройству. Отказ с причиной —
#: единственный честный ответ: CFG сделает то же самое за те же деньги.
UNSUPPORTED = {
    "krea2": "Krea 2",
    "minimax": "MiniMax H3",
}


# ────────────────────────────── математика NAG ───────────────────────────────

def apply_nag(x_positive, x_negative, *, scale: float, alpha: float, tau: float):
    """Свести позитивное и негативное внимание по формуле NAG.

    Ничего не мутирует: обе входные величины нужны дальше по формуле, и
    экономия на копии здесь уже стоила бы правильности.
    """
    guidance = x_positive * scale - x_negative * (scale - 1.0)

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(guidance, p=1, dim=-1, keepdim=True)

    # ⚠️ 0/0 на пустом токене даёт NaN. Верхнее значение выбрано заведомо больше
    # любого разумного tau, то есть такой токен обрезается, а не разносит кадр.
    ratio = torch.nan_to_num(norm_guidance / norm_positive, nan=10.0)
    adjustment = (norm_positive * tau) / (norm_guidance + 1e-7)
    guidance = guidance * torch.where(ratio > tau, adjustment, torch.ones_like(adjustment))

    return guidance * alpha + x_positive * (1.0 - alpha)


def conditional_rows(batch: int, transformer_options: dict) -> slice | None:
    """Какие строки батча — позитивные.

    ⚠️ Спрашиваем у ЯДРА, а не гадаем по форме. ComfyUI кладёт в
    ``transformer_options["cond_or_uncond"]`` по числу на кусок батча: 0 —
    позитив, 1 — негатив (``comfy/samplers.py``). Без этого пачка из двух картинок
    при cfg = 1 неотличима от пары «позитив + негатив», и половина кадров
    осталась бы без направляющей.

    Возвращает срез позитивных строк, или ``None``, когда позитив — весь батч.
    """
    layout = transformer_options.get("cond_or_uncond")
    if not layout or all(entry == 0 for entry in layout):
        return None
    chunk = batch // len(layout)
    starts = [index for index, entry in enumerate(layout) if entry == 0]
    if len(starts) != 1 or chunk <= 0:
        # Несколько разрозненных позитивных кусков одним срезом не описать.
        # Такой раскладки ядро не делает, но угадывать мы не станем.
        return None
    first = starts[0]
    return slice(first * chunk, (first + 1) * chunk)


def _to_module(tensor, module):
    """Привести тензор к устройству и типу модуля, который его сейчас считает."""
    parameter = next(module.parameters(), None) if hasattr(module, "parameters") else None
    if parameter is None:
        return tensor
    return tensor.to(parameter.device, parameter.dtype)


class _NagContext:
    """Негативный контекст, приготовленный под модель и закэшированный.

    Проекция текста дешёвая, но блоков десятки, а шагов ещё десятки — считать её
    заново в каждом вызове значит платить тысячи раз за одно и то же.
    """

    def __init__(self, raw, project):
        self._raw = raw
        self._project = project
        self._cache: dict[tuple, torch.Tensor] = {}

    def get(self, reference: torch.Tensor) -> torch.Tensor:
        key = (reference.device, reference.dtype, reference.shape[-1])
        cached = self._cache.get(key)
        if cached is None:
            cached = self._project(self._raw.to(reference.device, reference.dtype), reference)
            self._cache[key] = cached
        if cached.shape[0] != reference.shape[0]:
            cached = cached.expand(reference.shape[0], -1, -1)
        return cached


# ──────────────────────────────── Wan ────────────────────────────────────────

def _wan_attention(module, query, context, transformer_options):
    """Одно кросс-внимание Wan: ключи и значения из контекста, запрос готов."""
    from comfy.ldm.modules.attention import optimized_attention

    key = module.norm_k(module.k(context))
    value = module.v(context)
    return optimized_attention(query, key, value, heads=module.num_heads,
                               transformer_options=transformer_options)


def _wan_forward(module, x, context, nag, params, transformer_options, context_img_len=None):
    """Точный порт: NAG считается ДО выходной проекции ``o``."""
    from comfy.ldm.modules.attention import optimized_attention

    image_out = None
    if context_img_len is not None and hasattr(module, "k_img"):
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]
        query_img = module.norm_q(module.q(x))
        key_img = module.norm_k_img(module.k_img(context_img))
        value_img = module.v_img(context_img)
        # ⚠️ low_precision_attention=False — как в ядре: sage на этом крошечном
        # внимании даёт NaN, а выигрыша не даёт.
        image_out = optimized_attention(query_img, key_img, value_img,
                                        heads=module.num_heads,
                                        transformer_options=transformer_options,
                                        low_precision_attention=False)

    query = module.norm_q(module.q(x))
    rows = conditional_rows(x.shape[0], transformer_options)

    if rows is None:
        out = apply_nag(
            _wan_attention(module, query, context, transformer_options),
            _wan_attention(module, query, nag.get(context), transformer_options),
            **params,
        )
    else:
        out = _wan_attention(module, query, context, transformer_options)
        positive = out[rows]
        out = out.clone()
        out[rows] = apply_nag(
            positive,
            _wan_attention(module, query[rows], nag.get(context[rows]), transformer_options),
            **params,
        )

    if image_out is not None:
        out = out + image_out
    return module.o(out)


def _wan_blocks(diffusion_model):
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        return None
    if not blocks or not hasattr(blocks[0], "cross_attn"):
        return None
    return [(f"diffusion_model.blocks.{index}.cross_attn.forward", block.cross_attn)
            for index, block in enumerate(blocks)]


def _wan_project(raw, reference):
    return raw


# ──────────────────────────────── LTX ────────────────────────────────────────

def _ltx_forward(module, x, context, nag, params, transformer_options, **kwargs):
    """Оригинальный forward зовётся дважды; NAG ложится ПОСЛЕ проекции.

    ⚠️ ``type(module).forward`` — нетронутый метод класса: патч живёт атрибутом
    экземпляра и сюда не достаёт. Так тело LTX не дублируется, и ветки внутри
    него (RoPE, маска-гид, пер-головное гейтирование) остаются авторскими.
    """
    original = type(module).forward
    rows = conditional_rows(x.shape[0], transformer_options)

    if rows is None:
        positive = original(module, x, context=context,
                            transformer_options=transformer_options, **kwargs)
        negative = original(module, x, context=nag.get(context),
                            transformer_options=transformer_options, **kwargs)
        return apply_nag(positive, negative, **params)

    out = original(module, x, context=context,
                   transformer_options=transformer_options, **kwargs)
    negative = original(module, x[rows], context=nag.get(context[rows]),
                        transformer_options=transformer_options, **kwargs)
    out = out.clone()
    out[rows] = apply_nag(out[rows], negative, **params)
    return out


def _ltx_blocks(diffusion_model):
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        return None
    if not blocks or not hasattr(blocks[0], "attn2"):
        return None
    return [(f"diffusion_model.transformer_blocks.{index}.attn2.forward", block.attn2)
            for index, block in enumerate(blocks)]


def _ltx_project(raw, reference):
    """Повторяет ``LTXBaseModel._prepare_context``: проекция и сборка в 3D."""
    return raw.view(raw.shape[0], -1, reference.shape[-1])


# ─────────────────────────────── реестр ──────────────────────────────────────

ADAPTERS = {
    "ltx": {"blocks": _ltx_blocks, "forward": _ltx_forward, "project": _ltx_project,
            "caption": "caption_projection"},
    "wan": {"blocks": _wan_blocks, "forward": _wan_forward, "project": _wan_project,
            "caption": "text_embedding"},
}
MODEL_TYPES = [AUTO, *ADAPTERS]


def detect(diffusion_model) -> str | None:
    """Какое семейство перед нами — по структуре блоков, а не по имени файла."""
    for name, adapter in ADAPTERS.items():
        if adapter["blocks"](diffusion_model) is not None:
            return name
    return None


def unsupported_family(diffusion_model) -> str | None:
    """Узнать однопоточную архитектуру, чтобы отказать по имени."""
    module = type(diffusion_model).__module__
    for marker, title in UNSUPPORTED.items():
        if marker in module:
            return title
    return None


class _Patch:
    """Связывает патч с модулем так, как этого ждёт ComfyUI.

    ⚠️ Объектный патч кладётся АТРИБУТОМ ЭКЗЕМПЛЯРА, то есть дескриптором не
    становится и ``self`` сам не получает. Поэтому метод связывается здесь
    вручную — иначе первым позиционным аргументом приедет ``x``.
    """

    def __init__(self, forward, nag, params):
        self._forward = forward
        self._nag = nag
        self._params = params

    def bind(self, module):
        forward, nag, params = self._forward, self._nag, self._params

        def wrapped(self_module, x, context=None, transformer_options={}, **kwargs):
            if context is None:
                raise RuntimeError(f"{LOG_PREFIX} cross-attention was called without a context")
            return forward(self_module, x, context, nag, params,
                           transformer_options, **kwargs)

        return types.MethodType(wrapped, module)


class TS_NAG(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_NAG",
            display_name="TS NAG",
            category="TS/Utils",
            description=(
                "Normalized Attention Guidance: a negative prompt applied inside every "
                "cross-attention block instead of through CFG. Costs one extra "
                "cross-attention per block, not a second pass of the model, so the "
                "negative prompt keeps working on distilled models sampled at cfg 1. "
                "Wan and LTX only — models that fuse text and picture into one stream "
                "(Krea 2, MiniMax H3) are refused, because there the same thing is "
                "exactly what CFG already does."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="A Wan or LTX model."),
                IO.Conditioning.Input(
                    "negative",
                    tooltip=(
                        "What the picture should move away from. This is a real negative "
                        "prompt, not the zeroed-out one a cfg-1 sampler ignores."
                    ),
                ),
                IO.Float.Input(
                    "nag_scale",
                    default=11.0, min=0.0, max=100.0, step=0.1,
                    tooltip=(
                        "How far to push away from the negative. 0 switches the node off "
                        "entirely. The clamp below is what keeps a number this large usable."
                    ),
                ),
                IO.Float.Input(
                    "nag_alpha",
                    default=0.25, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the guided attention reaches the result; the rest "
                        "stays the plain positive attention."
                    ),
                ),
                IO.Float.Input(
                    "nag_tau",
                    default=2.5, min=0.0, max=10.0, step=0.05,
                    tooltip=(
                        "Ceiling on how far the guided attention may stray from the "
                        "positive one, measured as a ratio of L1 norms. Lower is safer "
                        "and weaker."
                    ),
                ),
                IO.Combo.Input(
                    "model_type",
                    options=MODEL_TYPES,
                    default=AUTO,
                    advanced=True,
                    tooltip=(
                        "Which family to patch. 'auto' recognises it by the structure of "
                        "the blocks, which is what you want; name one only to check a "
                        "model the detection does not know yet."
                    ),
                ),
            ],
            outputs=[
                IO.Model.Output(
                    display_name="model",
                    tooltip="The same model with guided cross-attention. The original is untouched.",
                ),
            ],
            search_aliases=["nag", "normalized attention guidance", "negative prompt",
                            "cfg 1", "distilled negative"],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, negative, nag_scale: float, nag_alpha: float,
                nag_tau: float, model_type: str = AUTO) -> IO.NodeOutput:
        patcher = model.clone()

        # 0 — это «ничего не делать», и патч в таком виде только жёг бы время.
        if float(nag_scale) == 0.0:
            logger.debug("%s nag_scale is 0 — nothing to guide", LOG_PREFIX)
            return IO.NodeOutput(patcher)

        diffusion_model = patcher.get_model_object("diffusion_model")

        family = model_type if model_type != AUTO else detect(diffusion_model)
        if family is None:
            refused = unsupported_family(diffusion_model)
            if refused:
                raise RuntimeError(
                    f"{LOG_PREFIX} {refused} joins text and picture into a single stream, "
                    "so it has no cross-attention to guide. A negative variant there costs "
                    "a full second pass of the model — which is exactly what CFG does, and "
                    "ComfyUI already has it. Raise cfg on the sampler instead."
                )
            raise RuntimeError(
                f"{LOG_PREFIX} Could not find cross-attention blocks in this model. "
                f"Supported families: {', '.join(ADAPTERS)}."
            )

        adapter = ADAPTERS.get(family)
        if adapter is None:
            raise RuntimeError(f"{LOG_PREFIX} Unknown model_type '{family}'.")

        targets = adapter["blocks"](diffusion_model)
        if not targets:
            raise RuntimeError(
                f"{LOG_PREFIX} This model does not look like {family}: its blocks carry no "
                "cross-attention. Leave model_type on 'auto' unless you know otherwise."
            )

        # Текстовая проекция модели (у Wan это text_embedding, у LTX —
        # caption_projection). Её у некоторых сборок нет вовсе — тогда негативное
        # кондиционирование уже в нужном виде.
        caption = getattr(diffusion_model, adapter["caption"], None)
        raw = negative[0][0]
        if callable(caption):
            raw = caption(_to_module(raw, caption))

        nag = _NagContext(raw, adapter["project"])
        params = {"scale": float(nag_scale), "alpha": float(nag_alpha), "tau": float(nag_tau)}
        patch = _Patch(adapter["forward"], nag, params)

        for path, module in targets:
            patcher.add_object_patch(path, patch.bind(module))

        logger.info("%s %s: guided %d cross-attention blocks (scale %.2f, alpha %.2f, tau %.2f)",
                    LOG_PREFIX, family, len(targets), params["scale"], params["alpha"],
                    params["tau"])
        return IO.NodeOutput(patcher)


NODE_CLASS_MAPPINGS = {"TS_NAG": TS_NAG}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_NAG": "TS NAG"}
