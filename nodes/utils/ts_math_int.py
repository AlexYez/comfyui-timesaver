"""TS Math Int — basic integer arithmetic (add/sub/mul/div/mod).

node_id: TS_Math_Int
"""

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
                result = int(pow(a, b))
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
