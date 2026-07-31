# TS Smart Switch

Type-aware boolean switch between two `ANY` inputs. Pick a `data_type` (images / video / audio / mask / string / int / float) so the node validates that the inputs match it. **Auto-failover**: if the selected input is missing, falls back to the other one — great for optional branches.

**Use when:** branching a workflow on a flag, or making one input optional with a sensible fallback.


<a id="conditioning"></a>
### 🎨 Conditioning (1 node)

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
