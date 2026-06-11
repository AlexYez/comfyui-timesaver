"""TS Math Int — basic integer arithmetic (add/sub/mul/div/mod).

node_id: TS_Math_Int
"""

import math

from comfy_api.v0_0_2 import IO

from .._shared import TS_Logger


_OPERATIONS = [
    "add (+)",
    "subtract (-)",
    "multiply (*)",
    "divide (/)",
    "floor_divide (//)",
    "modulo (%)",
    "power (**)",
    "min",
    "max",
]

# Cap the bit length of a ``power`` result. A naive ``a ** b`` with the int32
# inputs this node accepts (e.g. ``2 ** 2_000_000_000``) would allocate
# gigabytes and freeze the worker thread — and a hang/OOM-kill is not catchable
# by the try/except in execute(). ~1024 bits (≈308 digits) is far beyond any
# sane integer use yet computes instantly.
_POW_MAX_RESULT_BITS = 1024


class TS_Math_Int(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Math_Int",
            display_name="TS Math Int",
            category="TS/Utils",
            description="Integer math operations",
            inputs=[
                IO.Int.Input("a", default=0, min=-2147483648, max=2147483647, step=1),
                IO.Int.Input("b", default=0, min=-2147483648, max=2147483647, step=1),
                IO.Combo.Input("operation", options=_OPERATIONS),
            ],
            outputs=[IO.Int.Output(display_name="result")],
        )

    @classmethod
    def _safe_pow(cls, a: int, b: int) -> int:
        """``a ** b`` guarded against unbounded big-integer blow-up.

        Negative exponents yield a fraction that ``int()`` truncates, so only
        positive exponents with ``|a| >= 2`` can explode. Those are rejected by
        estimating the result's bit length *before* computing it.
        """
        if b < 0:
            if a == 0:
                raise ValueError("TS Math Int: 0 cannot be raised to a negative power.")
            return int(a ** b)
        if b == 0:
            return 1
        base = abs(a)
        if base > 1 and b * math.log2(base) > _POW_MAX_RESULT_BITS:
            raise ValueError(
                f"TS Math Int: power result too large to compute safely ({a} ** {b}); "
                f"the result would exceed {_POW_MAX_RESULT_BITS} bits."
            )
        return pow(a, b)

    @classmethod
    def execute(cls, a: int, b: int, operation: str) -> IO.NodeOutput:
        # No silent sentinels: a 0 emitted on division-by-zero used to flow
        # into the graph as a "valid" size/seed/count and break things far
        # from this node. Errors propagate to the pack's runtime guard.
        if operation == "add (+)":
            result = a + b
        elif operation == "subtract (-)":
            result = a - b
        elif operation == "multiply (*)":
            result = a * b
        elif operation == "divide (/)":
            if b == 0:
                raise ZeroDivisionError("TS Math Int: division by zero.")
            result = int(a / b)
        elif operation == "floor_divide (//)":
            if b == 0:
                raise ZeroDivisionError("TS Math Int: floor division by zero.")
            result = a // b
        elif operation == "modulo (%)":
            if b == 0:
                raise ZeroDivisionError("TS Math Int: modulo by zero.")
            result = a % b
        elif operation == "power (**)":
            result = cls._safe_pow(a, b)
        elif operation == "min":
            result = min(a, b)
        elif operation == "max":
            result = max(a, b)
        else:
            raise ValueError(f"TS Math Int: unknown operation '{operation}'.")

        TS_Logger.log("MathInt", f"{a} {operation} {b} = {result}")
        return IO.NodeOutput(int(result))


NODE_CLASS_MAPPINGS = {"TS_Math_Int": TS_Math_Int}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Math_Int": "TS Math Int"}
