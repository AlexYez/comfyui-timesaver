// JSON state shared by the compact editor and its fullscreen tile layout.
export function parseConfig(raw) {
    if (typeof raw !== "string" || !raw.trim()) return null;
    try {
        const data = JSON.parse(raw);
        const list = Array.isArray(data) ? data : (data?.blocks ?? data?.order);
        if (!Array.isArray(list)) return null;
        return list.flatMap((entry, index) => {
            if (entry?.type === "custom") {
                return [{ type: "custom", name: String(entry.id || `custom-${index}`),
                    text: String(entry.text ?? ""), enabled: entry.enabled !== false }];
            }
            const name = typeof entry === "string" ? entry : (entry?.file || entry?.name);
            return name ? [{ name: String(name), enabled: entry?.enabled !== false }] : [];
        });
    } catch { return null; }
}

export function buildItems(blocks, files, saved) {
    const available = new Set([
        ...(Array.isArray(blocks) ? blocks.map(e => e?.file || e?.name) : []),
        ...(Array.isArray(files) ? files : []),
    ].filter(Boolean).map(String));
    const hasSaved = Array.isArray(saved);
    const source = hasSaved ? saved : parseConfig(JSON.stringify(blocks || [])) || [];
    const result = [], seen = new Set();
    for (const entry of source) {
        const custom = entry.type === "custom";
        const key = `${custom ? "custom" : "file"}:${entry.name}`;
        if (seen.has(key) || (!custom && !available.has(entry.name))) continue;
        seen.add(key);
        result.push({ ...entry, label: custom ? "" : entry.name.replace(/\.txt$/i, "") });
    }
    for (const name of available) {
        if (seen.has(`file:${name}`)) continue;
        result.push({ name, label: name.replace(/\.txt$/i, ""), enabled: !hasSaved });
    }
    return result;
}

export function serializeItems(items, { filesOnly = false } = {}) {
    return JSON.stringify(items.filter(item => !filesOnly || item.type !== "custom").map(item =>
        item.type === "custom"
            ? { type: "custom", id: item.name, text: item.text, enabled: Boolean(item.enabled) }
            : { file: item.name, enabled: Boolean(item.enabled) }));
}

export function moveItem(items, from, to) {
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0
        || from >= items.length || to >= items.length || from === to) return items;
    const next = items.slice();
    next.splice(to, 0, next.splice(from, 1)[0]);
    return next;
}
