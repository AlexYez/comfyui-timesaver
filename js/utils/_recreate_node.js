// Чистая часть «Пересоздать ноду»: снимок связей и подбор гнёзд у новой ноды.
//
// Без `app` — её проверяет настоящий Node в тестах. Связь с графом живёт в
// `ts-recreate-node.js`.

/**
 * Связи ноды до пересоздания. Снимок, а не живые ссылки: после удаления
 * старой ноды её слоты уже пусты.
 *
 * @returns {{incoming: Array, outgoing: Array}}
 */
export function snapshotLinks(node, getLink) {
    const incoming = [];
    (node.inputs || []).forEach((input, slot) => {
        if (input?.link == null) return;
        const link = getLink(input.link);
        if (!link) return;
        incoming.push({
            name: input.name, type: input.type, slot,
            originId: link.origin_id, originSlot: link.origin_slot,
            parentId: link.parentId ?? null,
        });
    });
    const outgoing = [];
    (node.outputs || []).forEach((output, slot) => {
        for (const id of output?.links || []) {
            const link = getLink(id);
            if (!link) continue;
            outgoing.push({
                name: output.name, type: output.type, slot,
                targetId: link.target_id, targetSlot: link.target_slot,
                parentId: link.parentId ?? null,
            });
        }
    });
    return { incoming, outgoing };
}

function sameType(a, b) {
    return String(a ?? "") === String(b ?? "");
}

/**
 * Слот новой ноды для прежней связи: сначала по имени, затем — если вход
 * переименовали в новой версии ноды — тот же номер с тем же типом.
 *
 * @param {Array} slots входы или выходы новой ноды
 * @param {{name: string, type: *, slot: number}} wanted
 * @param {(index: number) => boolean} [isFree] занят ли уже слот (вход
 *        держит только одну связь)
 * @returns {number} индекс или -1
 */
export function matchSlot(slots, wanted, isFree = () => true) {
    const list = slots || [];
    const byName = list.findIndex((slot) => slot?.name === wanted.name);
    if (byName >= 0) return isFree(byName) ? byName : -1;
    const same = list[wanted.slot];
    if (same && sameType(same.type, wanted.type) && isFree(wanted.slot)) return wanted.slot;
    return -1;
}

/** Можно ли пересоздать: тип известен ComfyUI и это не сабграф. */
export function canRecreate(node, registeredTypes) {
    if (!node || !node.type) return false;
    if (typeof node.isSubgraphNode === "function" && node.isSubgraphNode()) return false;
    return Boolean(registeredTypes?.[node.type]);
}
