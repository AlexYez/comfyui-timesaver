import { app } from "/scripts/app.js";

const EXTENSION_ID = "ts.bookmark";
const NODE_TYPE = "TS_Bookmark";
const NODE_TITLE = "TS Bookmark 🔖";
const NODE_CATEGORY = "TS/Interface Tools";
const BOOKMARK_ICON = "🔖";

app.registerExtension({
    name: EXTENSION_ID,
    registerCustomNodes() {
        const liteGraph = globalThis.LiteGraph;
        const LGraphNode = liteGraph?.LGraphNode;
        if (!liteGraph || !LGraphNode) {
            return;
        }

        class TsBookmarkNode extends LGraphNode {
            type = NODE_TYPE;
            title = BOOKMARK_ICON;

            slot_start_y = -20;

            ___collapsed_width = 0;

            get _collapsed_width() {
                return this.___collapsed_width;
            }

            set _collapsed_width(width) {
                const canvas = app.canvas;
                const ctx = canvas?.canvas?.getContext("2d");
                if (ctx) {
                    const oldFont = ctx.font;
                    ctx.font = canvas.title_text_font;
                    this.___collapsed_width = 40 + ctx.measureText(this.title).width;
                    ctx.font = oldFont;
                }
            }

            isVirtualNode = true;
            serialize_widgets = true;
            keypressBound = null;

            constructor() {
                super(BOOKMARK_ICON);
                this.comfyClass = NODE_TYPE;
                this.addWidget(
                    "text",
                    "shortcut_key",
                    "1",
                    (value) => {
                        value = value.trim()[0] || "1";
                        if (value !== "") {
                            this.title = `${BOOKMARK_ICON} ${value}`;
                        }
                    },
                    {
                        y: 8,
                    }
                );
                this.addWidget(
                    "number",
                    "zoom",
                    1,
                    () => {},
                    {
                        y: 8 + liteGraph.NODE_WIDGET_HEIGHT + 4,
                        max: 2,
                        min: 0.5,
                        precision: 2,
                    }
                );
                this.keypressBound = this.onKeypress.bind(this);
            }

            onAdded() {
                setTimeout(() => {
                    const value = this.widgets[0].value;
                    if (value) {
                        this.title = `${BOOKMARK_ICON} ${value}`;
                    }
                }, 1);
                window.addEventListener("keydown", this.keypressBound);
            }

            onRemoved() {
                window.removeEventListener("keydown", this.keypressBound);
            }

            onKeypress(event) {
                const target = event.target;
                if (target && ["input", "textarea"].includes(target.localName)) {
                    return;
                }
                if (this.widgets[0] && event.key.toLocaleLowerCase() === this.widgets[0].value.toLocaleLowerCase()) {
                    this.canvasToBookmark();
                }
            }

            canvasToBookmark() {
                const canvas = app.canvas;
                // ComfyUI seemed to break this before; keep the original guards.
                if (canvas?.ds?.offset) {
                    canvas.ds.offset[0] = -this.pos[0] + 16;
                    canvas.ds.offset[1] = -this.pos[1] + 40;
                }
                if (canvas?.ds?.scale != null) {
                    canvas.ds.scale = Number(this.widgets[1].value || 1);
                }
                canvas?.setDirty(true, true);
            }
        }

        liteGraph.registerNodeType(
            NODE_TYPE,
            Object.assign(TsBookmarkNode, {
                title: NODE_TITLE,
            })
        );
        TsBookmarkNode.category = NODE_CATEGORY;
    },
});
