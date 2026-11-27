// Edit the original combo widgets so wired inputs, filename tokens and
// positional workflow serialization still use the same values.
export function addAudioExportSettings(node, topbar, L) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn ts-ui-btn--icon ts-audio-saver__settings-button";
    button.setAttribute("aria-label", L.exportSettings);
    button.title = L.exportSettings;
    button.textContent = "⚙︎";
    const panel = document.createElement("div");
    panel.className = "ts-audio-loader__palette ts-audio-saver__settings";
    panel.popover = "auto";
    button.popoverTargetElement = panel;
    const controls = [];
    const refresh = () => {
        for (const { widget, select, label } of controls) {
            select.value = widget.value;
            select.disabled = node.inputs?.some(input => input.name === widget.name && input.link != null) ?? false;
            label.hidden = widget.name === "mp3_bitrate" && controls[0].widget.value !== "MP3" && !controls[0].select.disabled;
        }
    };
    for (const [name, title] of [["format", L.fileFormat], ["mp3_bitrate", L.mp3Bitrate]]) {
        const widget = node.widgets?.find(item => item.name === name);
        if (!widget) continue;
        widget.hidden = true;
        const label = document.createElement("label");
        const select = document.createElement("select");
        for (const value of widget.options.values) select.add(new Option(value, value));
        select.addEventListener("change", () => {
            widget.value = select.value;
            widget.callback?.(select.value);
            node.graph?.change?.();
            refresh();
        });
        label.append(title, select);
        panel.append(label);
        controls.push({ widget, select, label });
    }
    panel.addEventListener("beforetoggle", event => {
        if (event.newState !== "open") return;
        refresh();
        const rect = button.getBoundingClientRect();
        panel.style.left = Math.max(4, Math.min(rect.left, window.innerWidth - 224)) + "px";
        panel.style.top = (rect.bottom + 96 < window.innerHeight ? rect.bottom + 4 : Math.max(4, rect.top - 96)) + "px";
    });
    topbar.append(button, panel);
}
