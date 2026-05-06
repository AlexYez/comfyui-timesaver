import re

from comfy_api.latest import IO


class TS_BatchPromptLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_BatchPromptLoader",
            display_name="TS Batch Prompt Loader",
            category="TS/Text",
            inputs=[
                IO.String.Input(
                    "text",
                    default="Prompt 1: cat\n\nPrompt 2: dog\n\nPrompt 3: bird",
                    multiline=True,
                    dynamic_prompts=False,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="prompt", is_output_list=True),
                IO.Int.Output(display_name="prompts_count"),
            ],
        )

    @classmethod
    def execute(cls, text) -> IO.NodeOutput:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_prompts = re.split(r"\n\s*\n", text)

        valid_prompts = []
        for p in raw_prompts:
            cleaned_p = p.strip()
            if cleaned_p:
                valid_prompts.append(cleaned_p)

        if not valid_prompts:
            valid_prompts = [""]

        count = len(valid_prompts)

        return IO.NodeOutput(valid_prompts, count)


NODE_CLASS_MAPPINGS = {
    "TS_BatchPromptLoader": TS_BatchPromptLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_BatchPromptLoader": "TS Batch Prompt Loader",
}
