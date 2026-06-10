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
                TS_Logger.error("MathInt", "0 cannot be raised to a negative power")
                return 0
            return int(a ** b)
        if b == 0:
            return 1
        base = abs(a)
        if base > 1 and b * math.log2(base) > _POW_MAX_RESULT_BITS:
            TS_Logger.error(
                "MathInt",
                f"power result too large to compute safely ({a} ** {b}); returning 0",
            )
            return 0
        return pow(a, b)

    @classmethod
    def execute(cls, a: int, b: int, operation: str) -> IO.NodeOutput:
        try:
            if operation == "add (+)":
                result = a + b
            elif operation == "subtract (-)":
                result = a - b
            elif operation == "multiply (*)":
                result = a * b
            elif operation == "divide (/)":
                if b == 0:
                    TS_Logger.error("MathInt", "Division by zero")
                    result = 0
                else:
                    result = int(a / b)
            elif operation == "floor_divide (//)":
                if b == 0:
                    TS_Logger.error("MathInt", "Division by zero")
                    result = 0
                else:
                    result = a // b
            elif operation == "modulo (%)":
                if b == 0:
                    TS_Logger.error("MathInt", "Modulo by zero")
                    result = 0
                else:
                    result = a % b
            elif operation == "power (**)":
                result = cls._safe_pow(a, b)
            elif operation == "min":
                result = min(a, b)
            elif operation == "max":
                result = max(a, b)
            else:
                TS_Logger.error("MathInt", f"Unknown operation: {operation}")
                result = 0

            TS_Logger.log("MathInt", f"{a} {operation} {b} = {result}")
            return IO.NodeOutput(int(result))
        except Exception as e:
            TS_Logger.error("MathInt", f"Error: {str(e)}")
            return IO.NodeOutput(0)


NODE_CLASS_MAPPINGS = {"TS_Math_Int": TS_Math_Int}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Math_Int": "TS Math Int"}
