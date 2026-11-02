# Tidy up — one command, no node

Frontend command; no separate node. Source: [tidy/ts-tidy-layout.js](../../js/utils/tidy/ts-tidy-layout.js). Measurements below are historical examples from the former README, not guarantees for every graph.

Right-click on the canvas (or on any node) → **Tidy up** → **Tidy layout**. Whatever is selected — or the whole graph, when nothing is — gets arranged: every node shrinks to the size its own content asks for, and the lot is laid out in columns that follow the wiring, left to right, snapped to the grid. Both entries are in the command palette too, so you can bind keys to them.

The column a node lands in is its distance from the start of the flow, so a loader is always left of the sampler that reads it, and a node with one input lines up with what feeds it. Inside a column the order is chosen to keep the links from crossing, starting from the order you already had — the command tidies your arrangement rather than replacing it with someone else's. The whole thing stays where the schema was: the top-left corner does not move, so you are not left hunting for your graph afterwards.

**Groups are laid out from the inside out.** The nodes of a group are arranged within it, the frame is then fitted to them, and the group takes part in the outer layout as a single block — an organised workflow stays organised. **Pinned nodes are never moved**: pinning is how you say "this one stays", and the command respects it.

**Wires that cut through nodes can be routed around them.** The second entry, **Tidy layout + route the wires**, lays the graph out and then, for every wire whose straight line would cross somebody else's node, drops link dots that take it into the corridor between columns, along a free lane, and back — one lane for the whole detour, and any dot that earns nothing is dropped again. Wires that already have dots are left to their owner, and wires running against the flow are left alone. Run it twice and nothing changes.

Each wire is routed knowing about the ones already routed, and the lane it takes is chosen by what it costs: **lying on top of another wire is all but forbidden** — two lines reading as one is what makes a schema unreadable, and you cannot even tell how many wires are there. Crossing costs far less, a longer detour least of all. That is a deliberate trade: a crossing is visible and understandable, an overlap is not. Wires leaving the same socket get no exemption either. A wire is rerouted not only when it cuts a node but also when it comes to rest on another wire, and the whole set is then routed a second time, each wire lifted and laid again now that it knows where all the others ended up.

Measured on a real 32-node workflow: the layout alone left 68 places where a wire runs through a node and 94 wire crossings; routing brings that to **0 cuts, 0 overlaps and 88 crossings**. The same graph as its author had arranged it by hand: 40 cuts, 84 crossings. The crossings that remain come from the order of the nodes in their columns rather than from the wires — laying out and routing together, so that the order is chosen with the wires in mind, is the next step and is not done yet.

**Or pack it as tiles.** The third entry, **Pack as tiles**, reserves an area on the left for model loaders and preparation: checkpoints, UNET, VAE, CLIP, ControlNet, image and latent upscale models. Loaders align in one column; LoRAs and model patches follow in dependency columns. Output socket types identify model nodes even when their titles have been renamed. Model nodes inside mixed processing groups move into the model area; groups containing only models stay together. The remaining nodes fill compact columns on the right in processing order, with equal tile widths within each column. Column heights account for the whole layout, including models, aiming for **16:9**. Selection and pinned-node rules still apply. **Wires and their link dots are preserved.**

Measured on a real 32-node workflow: 17 columns and 5540×1308 become **5 columns and 1760×1128** — the area actually filled by nodes goes from 17% to 76%, and the shape from 4.2:1 to 1.6:1, which fits on a screen.

**The dots on your links are straightened too.** ComfyUI's link reroutes — the small round points you drop onto a wire — are spread evenly along the straight line between the socket they leave and the socket they enter, so a wire that ran as a dogleg through a point you tossed somewhere becomes a straight run. A point shared by several links settles between them. A wire whose straight line would cross a node is left bent — a detour that exists for a reason is not undone. Nothing is created and nothing is deleted: the graph you get back is the one you had. **Align link dots only** is the second entry in the submenu, for when the nodes are already where you want them.

Works in both node renderers. That is not a given: measured on the same graph, Nodes 2.0 gives a node a different size than Nodes 1.0 (`SaveImage` 58 px against 70) and refuses to shrink some nodes at all while `node.size` claims otherwise — so the command asks the canvas what it actually drew before deciding where anything goes.

---

[Навигация / index](README.md).
