"""Access to the pack-level :class:`TSDependencyManager` from node modules.

Node modules import it from here (``from .._deps import TSDependencyManager``)
rather than reaching up with ``from ...ts_dependency_manager import ...``.

Why: ComfyUI loads the pack as a package, so the node modules are
``<pack>.nodes.<category>.<module>`` and a three-dot import resolves. The test
suite imports the very same files as ``nodes.<category>.<module>`` (pack root on
``sys.path``), where ``nodes`` IS the top-level package and the third dot walks
off the top — ``ImportError: attempted relative import beyond top-level
package``. Keeping that fallback in one place means node modules use a plain
two-dot import that resolves identically under both loaders.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

try:  # ComfyUI: the pack root is this module's grandparent package.
    from ..ts_dependency_manager import TSDependencyManager
except ImportError:  # Tests: `nodes` is top-level and the pack root is on sys.path.
    from ts_dependency_manager import TSDependencyManager

__all__ = ["TSDependencyManager"]
