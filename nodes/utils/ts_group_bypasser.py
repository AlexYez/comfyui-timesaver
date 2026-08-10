"""TS Group Bypasser — a control panel for the groups of the open workflow.

node_id: TS_GroupBypasser

The work happens entirely on the canvas, in js/utils/group_bypasser/: the node
lists the workflow's groups with a checkbox each, and unchecking one puts every
node inside that group into bypass. There is nothing for the backend to do, and
this class is deliberately empty of behaviour.

So why is there a Python file at all? Because it is what makes this a node of
the pack rather than a stowaway: it appears in the node library and in search,
it carries a description and an embedded help page, and it is covered by the
contract snapshot like the other sixty-one. The alternative — registering a
purely frontend node through LiteGraph, as ts-bookmark.js does — is kept in
this pack for compatibility only (CLAUDE.md 12).

With no inputs and no outputs, and not being an output node, nothing in a graph
can depend on it, so the executor never reaches it. Its settings (the filters,
the sort order, the restriction rule) live in ``node.properties`` on the canvas
side and travel with the workflow from there. Deliberately NOT widgets:
``widgets_values`` is positional, and a widget added here would shift the values
of every node already saved in someone's workflow.

The state of the groups themselves is not stored anywhere either — it IS the
modes of the nodes inside them, which the workflow already saves. Open a file
with a group bypassed and it comes back bypassed, without a line of code here.
"""

from comfy_api.v0_0_2 import IO


class TS_GroupBypasser(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_GroupBypasser",
            display_name="TS Group Bypasser",
            category="TS/Utils",
            description=(
                "Switch whole groups of the open workflow on and off from one panel: "
                "unchecking a group puts every node inside it into bypass. Filter the "
                "list by group title or colour, and optionally keep only one group "
                "enabled at a time."
            ),
            inputs=[],
            outputs=[],
            # Findable by what it does, in either language and by either name —
            # people coming from other packs look for "muter" as often as for
            # "bypasser".
            search_aliases=[
                "group", "groups", "bypass", "bypasser", "mute", "muter",
                "группы", "группа", "байпас", "выключить группу",
            ],
        )

    @classmethod
    def execute(cls) -> IO.NodeOutput:
        """Never called: a node with no outputs is not reachable from any output.

        Present because the schema demands an entrypoint, and honest about the
        fact that reaching it would mean something upstream is wrong.
        """
        return IO.NodeOutput()


NODE_CLASS_MAPPINGS = {"TS_GroupBypasser": TS_GroupBypasser}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_GroupBypasser": "TS Group Bypasser"}
