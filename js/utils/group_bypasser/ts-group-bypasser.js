// TS Group Bypasser — switch whole groups of the workflow on and off.
//
// This file is the wiring: it holds the node's settings, asks _groups_watch.js
// what the graph looks like, asks _groups_model.js what a click means, and
// hands the answer to _groups_view.js to draw. Each of those three can be read
// on its own, and only this one knows about all of them.

import { app } from "/scripts/app.js";

import { addResizableDomWidget, getWidget } from "../../_dom_widget.js";
import {
    RESTRICT_ALWAYS_ONE,
    RESTRICT_MAX_ONE,
    RESTRICT_NONE,
    SORT_COLOR,
    SORT_POSITION,
    SORT_TITLE,
    STATE_ON,
    compileTitleFilter,
    filterGroups,
    groupState,
    planBulk,
    planToggle,
    sortGroups,
} from "./_groups_model.js";
import {
    MIN_NODE_HEIGHT,
    WIDGET_CHROME_HEIGHT,
    createGroupsPanel,
    heightForRows,
} from "./_groups_view.js";
import {
    NODE_TYPE,
    applyDecisions,
    invalidateSignature,
    readGroups,
    resolveColourFilter,
    revealGroup,
    watchGroups,
} from "./_groups_watch.js";

const EXTENSION_ID = "ts.groupBypasser";
const WIDGET_NAME = "ts_group_bypasser";

const DEFAULT_NODE_WIDTH = 240;
const MIN_NODE_WIDTH = 180;
// Settings are node PROPERTIES, edited through the node's own Properties Panel
// (right-click the node). Two reasons, and neither is taste. They do not belong
// in the node's body: the body is the list of groups, and a filter box sitting
// above it is chrome you look at every day to use twice. And they must not be
// widgets: `widgets_values` is positional, so a widget added here would shift
// the stored values of every node in every workflow already saved.
//
// The panel reads the editor for each property from a static `@name` field on
// the node class — LiteGraph's own convention, the same one rgthree uses.
const PROPERTY_INFO = {
    filter_title: { type: "string" },
    filter_colors: { type: "string" },
    sort: { type: "combo", values: [SORT_POSITION, SORT_TITLE, SORT_COLOR] },
    restriction: { type: "combo", values: [RESTRICT_NONE, RESTRICT_MAX_ONE, RESTRICT_ALWAYS_ONE] },
};

const DEFAULTS = {
    filter_title: "",
    // Comma-separated, and colour NAMES are accepted alongside hex — "blue,
    // green" is how a person thinks about the colours they gave their groups,
    // and "none" is the bucket for the ones they left plain.
    filter_colors: "",
    sort: SORT_POSITION,
    restriction: RESTRICT_NONE,
};

function readSetting(node, name) {
    const value = node?.properties?.[name];
    if (value === undefined || value === null || value === "") return DEFAULTS[name];
    return value;
}

function setupNode(node) {
    if (node.__tsGroupBypasser) return;
    // A node can be handed back after its widget already exists — switching
    // workflow tabs is the usual way. Adding a second DOM widget there is the
    // mistake that doubles the top of a node in Nodes 2.0, because Vue and
    // LiteGraph each keep their own registration (CLAUDE.md 12.5.12).
    if (getWidget(node, WIDGET_NAME)) return;

    node.properties ||= {};
    for (const [name, value] of Object.entries(DEFAULTS)) {
        if (node.properties[name] === undefined) node.properties[name] = value;
    }

    // The reading of the graph the panel is currently showing. Clicks are
    // planned against this list rather than against a fresh read, so what the
    // person sees is what they act on; the write re-checks membership anyway.
    let visible = [];
    let records = [];
    // The height this node last set for itself, whether the person has since
    // taken the matter into their own hands, and a latch so that our own write
    // is not mistaken for theirs.
    let autoHeight = null;
    let manualHeight = false;
    let applyingSize = false;

    const settings = () => ({
        filter_title: readSetting(node, "filter_title"),
        filter_colors: readSetting(node, "filter_colors"),
        sort: readSetting(node, "sort"),
        restriction: readSetting(node, "restriction"),
    });

    const panel = createGroupsPanel({
        onAction(action) {
            if (action.type === "reveal") {
                revealGroup(visible.find((row) => row.key === action.key));
                return;
            }
            const decisions = action.type === "bulk"
                ? planBulk(visible, action.action, settings().restriction)
                : planToggle(visible, action.key, settings().restriction);
            if (!decisions.length) return;
            applyDecisions(decisions, records);
            // Redraw at once instead of waiting out the poll: a checkbox that
            // answers a third of a second late feels broken.
            invalidateSignature();
            refresh();
        },
    });

    function draw(current) {
        records = current || [];
        const config = settings();
        const matcher = compileTitleFilter(config.filter_title);
        const shown = filterGroups(records, {
            title: config.filter_title,
            colours: resolveColourFilter(config.filter_colors),
        });
        visible = sortGroups(shown, config.sort);
        fitToRows(visible.length);
        panel.render({
            rows: visible.map((row) => ({
                key: row.key,
                title: row.title,
                color: row.color,
                state: groupState(row.modes),
            })),
            filterValid: matcher.valid,
            totalGroups: records.length,
        });
    }

    function refresh() {
        draw(readGroups());
    }

    function fitToRows(count) {
        if (manualHeight) return;
        const wanted = heightForRows(count);
        autoHeight = wanted;
        if (Math.round(node.size?.[1] || 0) === wanted) return;
        const width = Math.max(MIN_NODE_WIDTH, Math.round(node.size?.[0] || DEFAULT_NODE_WIDTH));
        applyingSize = true;
        try {
            if (typeof node.setSize === "function") node.setSize([width, wanted]);
            else node.size = [width, wanted];
        } finally {
            applyingSize = false;
        }
        node.graph?.setDirtyCanvas?.(true, true);
    }

    // Nothing is done to the wrapper the helper returns: the panel's own
    // `height:100%` is what fills it, exactly as in js/image/ts-resolution-selector.js,
    // and reaching into the renderer's element is how sizing goes wrong in Vue.
    addResizableDomWidget(node, panel.element, {
        name: WIDGET_NAME,
        minWidth: MIN_NODE_WIDTH,
        minHeight: MIN_NODE_HEIGHT,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: heightForRows(0),
        chromeHeight: WIDGET_CHROME_HEIGHT,
        minWidgetHeight: MIN_NODE_HEIGHT - WIDGET_CHROME_HEIGHT,
    });

    // Once the person drags the node's edge, that height is theirs and the
    // fitting stops — until the list of groups changes again, or the workflow
    // is reopened, because the height is derived from the content and there is
    // nowhere to keep an override that would not show up as a fifth property.
    // A couple of pixels of slack: the shared helper clamps sizes on its way
    // through, and that rounding is not a person reaching for the edge.
    const previousResize = node.onResize;
    node.onResize = function onResize() {
        const result = previousResize?.apply(this, arguments);
        const height = Math.round(node.size?.[1] || 0);
        if (!applyingSize && autoHeight !== null && Math.abs(height - autoHeight) > 2) {
            manualHeight = true;
        }
        return result;
    };

    let unwatch = watchGroups(draw);
    // The panel survives removal; only the polling stops. A node taken off the
    // canvas and put back — a workflow tab switched away from and returned to —
    // keeps the DOM it already has and simply starts listening again.
    node.__tsGroupBypasser = {
        refresh,
        strings: panel.strings,
        bulk: (action) => {
            const decisions = planBulk(visible, action, settings().restriction);
            if (!decisions.length) return;
            applyDecisions(decisions, records);
            invalidateSignature();
            refresh();
        },
        tally: () => ({
            on: visible.filter((row) => groupState(row.modes) === STATE_ON).length,
            total: visible.length,
        }),
        resume() {
            if (!unwatch) unwatch = watchGroups(draw);
            else refresh();
        },
    };

    // Editing a property in the panel must show up in the list at once, not
    // whenever the next poll happens to notice.
    const previousPropertyChanged = node.onPropertyChanged;
    node.onPropertyChanged = function onPropertyChanged(name) {
        const result = previousPropertyChanged?.apply(this, arguments);
        if (name in DEFAULTS) refresh();
        return result;
    };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function onRemoved() {
        unwatch?.();
        unwatch = null;
        return previousRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        // Read by LGraphNode.getPropertyInfo, which is what gives the
        // Properties Panel a dropdown instead of a free-text box.
        for (const [name, info] of Object.entries(PROPERTY_INFO)) {
            nodeType[`@${name}`] = info;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function created() {
            const result = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return result;
        };

        // Acting on many groups at once belongs in the menu, not in the body:
        // it is used rarely and would otherwise be three buttons under the list
        // for the whole life of the node.
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function extraMenu(canvas, options) {
            const result = getExtraMenuOptions?.apply(this, arguments);
            const control = this.__tsGroupBypasser;
            if (control && Array.isArray(options)) {
                const t = control.strings;
                const counts = control.tally();
                options.unshift(
                    { content: t.menuTally(counts.on, counts.total), disabled: true },
                    { content: t.menuEnableAll, callback: () => control.bulk("enable") },
                    { content: t.menuDisableAll, callback: () => control.bulk("disable") },
                    { content: t.menuInvert, callback: () => control.bulk("invert") },
                    null,
                );
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_TYPE) return;
        // Settings are restored from properties by the time this runs, and the
        // group list is read from the graph rather than stored, so there is
        // nothing to rehydrate — a redraw with the values now in place, and
        // never a rebuilt widget.
        if (!node.__tsGroupBypasser) setupNode(node);
        else node.__tsGroupBypasser.resume();
    },
});
