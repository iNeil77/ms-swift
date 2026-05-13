# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OctoLong agent template.

Subclasses HermesAgentTemplate so that the rendered chat-template output is
byte-equivalent to Qwen3's Hugging Face `chat_template.jinja` for the Swift
backend.

Two layout fixes vs. plain `hermes`:

  Q1. SYSTEM_EXTRA_NL — the upstream hermes `_format_tools` always prepends
      `f"{system}\\n\\n# Tools..."`. When `system` is empty, this yields
      `\\n\\n# Tools` and the wrapper produces `<|im_start|>system\\n\\n\\n# Tools`.
      Qwen3's HF template emits `<|im_start|>system\\n# Tools` in that case.
      We fix this by only emitting the leading `{system}\\n\\n` when `system`
      is non-empty.

  Q2. PRETEXT_TOOLCALL_BOUNDARY — see swift/template/templates/octolong.py
      (handled at the Template level rather than here, because the agent
      template doesn't see the surrounding messages).
"""
import json
from typing import List, Optional, Union

from .hermes import HermesAgentTemplate


class OctoLongAgentTemplate(HermesAgentTemplate):

    def _format_tools(self,
                      tools: List[Union[str, dict]],
                      system: Optional[str] = None,
                      user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(t), ensure_ascii=False) for t in tools]
        # Q1: only prepend "{system}\n\n" when system is non-empty.
        prefix = f'{system}\n\n' if system else ''
        return (
            prefix
            + '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
            + 'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
            + '\n'.join(tool_descs)
            + '\n</tools>\n\n'
            + 'For each function call, return a json object with function name and arguments within '
            + '<tool_call></tool_call> XML tags:\n<tool_call>\n'
            + '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
