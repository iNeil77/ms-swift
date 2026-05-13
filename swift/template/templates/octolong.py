# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OctoLong template.

A chatml-shaped chat template (same wire format as `qwen3_nothinking`) plus a
Template subclass that fixes the v4.2.0 hermes-path divergences from Qwen3's
official HF chat_template:

  Q2. PRETEXT_TOOLCALL_NEWLINE
      For the canonical OpenAI "pre-text + tool_calls" pattern (one assistant
      turn carrying both `content` and `tool_calls`), Qwen3's HF chat_template
      renders it as:
          <|im_start|>assistant\\n{pre_text}\\n<tool_call>...</tool_call><|im_end|>
      v4.2.0's `_swift_prepare_inputs` merges consecutive assistants by
      converting `content` to a List[str] (so per-segment loss_scale can be
      tracked) and the downstream concat drops the '\\n' between the pre_text
      and '<tool_call>'. We override the merge to keep `content` as a plain
      string and insert exactly one '\\n' separator between the two
      consecutive assistant contents.

  Q3. The downstream `endswith_stop_words` check in base.py:1248-1254 mistakes
      a list-content response for token_ids and crashes (`TypeError: argument
      'ids': 'str' object cannot be interpreted as an integer`). With Q2 we
      keep contents as strings, so this path no longer triggers.

The fixes here are activated by selecting `--template octolong` and
`--agent_template octolong` in the SFT command. Plain `hermes` / `qwen3_nothinking`
behavior is unchanged.
"""
from dataclasses import dataclass
from typing import Optional

from ..base import Template
from ..constant import LLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from .qwen import QwenTemplateMeta


class OctoLongTemplate(Template):

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        """Same as the base Template._swift_prepare_inputs, but the
        consecutive-assistant merge:
          - inserts the chat_sep + system_prefix-equivalent boundary
            ('<|im_end|>\\n<|im_start|>assistant\\n') between the two contents
            so the rendered text matches HF's chat_template;
          - keeps `content` as a plain str so the downstream final-round
            endswith_stop_words check doesn't try to decode it as token_ids.
        """
        messages = inputs.messages
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool' and self.template_backend == 'swift':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif (pre_role == 'assistant' and role == 'assistant'
                  or pre_role == 'user' and role == 'user'):
                if (self.template_backend == 'swift'
                        and pre_role == 'assistant'
                        and isinstance(pre_content, str)
                        and isinstance(content, str)):
                    # Q2 fix: keep `content` as a plain string and insert
                    # exactly one '\n' between the pre-text and the
                    # synthesized tool_calls block. Qwen3's HF chat_template
                    # always emits this '\n' (regardless of trailing
                    # whitespace in `pre_text`):
                    #   <|im_start|>assistant\n{pre_text}\n<tool_call>...</tool_call><|im_end|>
                    pre_message['content'] = pre_content + '\n' + content
                    # If either side has explicit loss/loss_scale, keep the
                    # first one (matches base Template behavior for binary
                    # loss; per-segment loss_scale on this combined content
                    # is not preserved here, but neither side should be using
                    # it for our SFT recipe).
                    for key in ('loss', 'loss_scale'):
                        if key in message and key not in pre_message:
                            pre_message[key] = message[key]
                else:
                    pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1


@dataclass
class OctoLongTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = None
    agent_template: str = 'octolong'


register_template(OctoLongTemplateMeta(LLMTemplateType.octolong, template_cls=OctoLongTemplate))
