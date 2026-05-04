"""TS Math Int — basic integer arithmetic (add/sub/mul/div/mod).

node_id: TS_Math_Int
"""

from .._shared import TS_Logger


class TS_Math_Int:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "b": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "operation": ([
                    "add (+)",
                    "subtract (-)",
                    "multiply (*)",
                    "divide (/)",
                    "floor_divide (//)",
                    "modulo (%)",
                    "power (**)",
                    "min",
                    "max"
                ],),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "calculate"
    CATEGORY = "TS/Math"
    DESCRIPTION = "Integer math operations"

    def calculate(self, a, b, operation):
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
            return (int(result),)
        except Exception as e:
            TS_Logger.error("MathInt", f"Error: {str(e)}")
            return (0,)

# ==============================================================================
# Node 5: TS Animation Preview


NODE_CLASS_MAPPINGS = {"TS_Math_Int": TS_Math_Int}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Math_Int": "TS Math Int"}
