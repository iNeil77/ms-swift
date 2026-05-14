#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Convert canonical OpenAI-format chat datasets into ms-swift's native message
format.

ms-swift's row preprocessor in ``swift/dataset/preprocessor/core.py`` keeps
only ``{role, content, loss, loss_scale}`` on each message and drops every
other field. This silently destroys the canonical OpenAI representation of
tool-calling traces, where an assistant turn carries both ``content`` (the
pre-tool-call reasoning) and ``tool_calls`` (the actual function-call
payload). To survive ms-swift's preprocessor, those messages must be
re-shaped into Swift's native sequence:

    {"role": "assistant",      "content": "<pre_text>"}                     # optional
    {"role": "tool_call",      "content": '{"name": ..., "arguments": ...}'} # one per call
    ...
    {"role": "tool_response",  "content": "<json string from the tool>"}    # one per response

ms-swift's swift backend then merges the consecutive ``tool_call`` rows back
into a single assistant turn at template-render time via the agent template
(e.g. hermes/octolong).

This script reads a HuggingFace dataset whose rows have the OpenAI shape:

    messages: list<struct<
        content: string,
        role: string,                   # "system" | "user" | "assistant" | "tool"
        tool_calls: string              # JSON-encoded list[OpenAI-style call] OR
                                        # the literal string "None" / null
    >>
    tools:    string                    # JSON-encoded list[tool spec] OR null
    ...                                 # any other columns are carried through

Each ``tool_calls`` entry is parsed and split into one ``role="tool_call"``
message per call; ``role="tool"`` is renamed to ``role="tool_response"`` (both
are accepted by ms-swift, ``tool_response`` is documented). Tool ``arguments``
strings are JSON-decoded inline so the resulting ``content`` carries an
already-structured payload, eliminating a layer of escape sequences.

Two input formats are accepted:

* ``load_from_disk`` directories (``state.json`` + ``data-*.arrow``). Use
  ``--src-format=disk``.
* HuggingFace parquet datasets (``data/train-*.parquet``). Use
  ``--src-format=hub`` (default).

The output is always written as parquet shards under ``<dst>/data/`` plus a
minimal ``README.md`` so the result is consumable directly via
``--dataset <dst>`` from any ms-swift CLI.

Example
-------

::

    python scripts/utils/openai_to_swift_dataset.py \
        --src     /path/to/Tool-SFT-Mix-Final \
        --src-format disk \
        --dst     /path/to/Tool-SFT-Mix-Swift \
        --num-shards 354 \
        --num-proc   160
"""
import argparse
import json
import os
import shutil
import sys
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_maybe(value: Any) -> Any:
    """Best-effort JSON-or-passthrough decode.

    ``tool_calls`` and ``tools`` columns in canonical OpenAI exports are
    typically stored as JSON strings, with empty values represented as the
    literal Python ``None`` or as the strings ``"None"``, ``"nan"``, ``"null"``.
    This helper returns:

    * ``None`` for any of the empty representations,
    * the original ``dict``/``list`` if already structured,
    * the JSON-decoded value if it parses,
    * ``None`` if the string is non-empty but not valid JSON.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "nan", "null"):
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def openai_to_swift_messages(messages: List[dict]) -> List[dict]:
    """Convert one row's messages from OpenAI format to ms-swift native form.

    Behavior:

    * ``role="assistant"`` carrying a non-empty ``tool_calls`` field becomes:
        - an optional ``role="assistant"`` row containing the original
          ``content`` (only if non-whitespace),
        - one ``role="tool_call"`` row per call, with ``content`` set to
          ``json.dumps({"name": <fn>, "arguments": <args>})`` where
          ``<args>`` is the parsed argument structure (a dict in nearly all
          cases) rather than its JSON string form. This avoids double-quoting
          when the agent template re-serializes for the wire format.
    * ``role="tool"`` is renamed to ``role="tool_response"``. Both are
      accepted by ms-swift; ``tool_response`` is the documented canonical
      name.
    * All other roles (``system``, ``user``, plain ``assistant`` without
      tool_calls) are passed through with only ``role`` and ``content``
      preserved. The ``tool_calls`` field, even if present and empty, is
      dropped to match the structure ms-swift's preprocessor would produce.

    Args:
        messages: list of OpenAI-style messages.

    Returns:
        List of ms-swift native messages (each having only ``role`` and
        ``content`` keys).
    """
    out: List[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        tc = _parse_json_maybe(m.get("tool_calls"))

        if role == "assistant" and tc:
            # Normalize a single dict to a list to handle non-conforming sources.
            if isinstance(tc, dict):
                tc = [tc]

            # Preserve any pre-tool-call reasoning as its own assistant turn.
            if isinstance(content, str) and content.strip():
                out.append({"role": "assistant", "content": content})

            for call in tc:
                if not isinstance(call, dict):
                    continue
                # Accept both nested OpenAI form ({"function": {...}, ...})
                # and flat form ({"name": ..., "arguments": ...}).
                fn = (
                    call.get("function")
                    if isinstance(call.get("function"), dict)
                    else call
                )
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    parsed = _parse_json_maybe(args)
                    if parsed is not None:
                        args = parsed
                if args is None:
                    args = {}
                out.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(
                            {"name": fn.get("name"), "arguments": args},
                            ensure_ascii=False,
                        ),
                    }
                )
        elif role == "tool":
            out.append({"role": "tool_response", "content": content})
        else:
            out.append({"role": role, "content": content})
    return out


# ---------------------------------------------------------------------------
# .map worker
# ---------------------------------------------------------------------------


def _row_worker(batch):
    """``datasets.Dataset.map`` worker: re-shape ``messages`` in place."""
    new_messages = []
    for raw in batch["messages"]:
        # ``raw`` is a list of struct rows; convert to plain dicts.
        msgs = [
            {
                "role": m["role"],
                "content": m["content"],
                "tool_calls": m.get("tool_calls"),
            }
            for m in raw
        ]
        new_messages.append(openai_to_swift_messages(msgs))
    return {"messages": new_messages}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_source(src: str, src_format: str):
    """Load the source dataset using the format-appropriate loader."""
    from datasets import load_dataset, load_from_disk

    if src_format == "disk":
        # `save_to_disk` arrow output (state.json + data-*.arrow).
        return load_from_disk(src)
    if src_format == "hub":
        # Either a local dir with `data/train-*.parquet` and a `README.md`
        # config block, or any string accepted by `datasets.load_dataset`
        # (HF hub id, ModelScope id, etc.). Local dirs are detected
        # automatically by load_dataset.
        return load_dataset(src, split="train")
    raise ValueError(f"unknown --src-format: {src_format!r}")


def _write_parquet_shards(ds, dst_dir: str, num_shards: int) -> None:
    """Re-shard ``ds`` into ``num_shards`` contiguous parquet files under
    ``<dst_dir>/data/`` and emit a minimal HF dataset card."""
    out_data = os.path.join(dst_dir, "data")
    if os.path.exists(out_data):
        shutil.rmtree(out_data)
    os.makedirs(out_data, exist_ok=True)

    # The shard count drives parallelism for downstream `load_dataset`. We
    # match the input shard count by default (passed from CLI).
    width = max(5, len(str(num_shards)))
    print(f"  writing {num_shards} parquet shards to {out_data} ...", flush=True)
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard.to_parquet(
            os.path.join(out_data, f"train-{i:0{width}d}-of-{num_shards:0{width}d}.parquet")
        )
        if i % 40 == 0 or i == num_shards - 1:
            print(f"    wrote shard {i + 1}/{num_shards}", flush=True)

    with open(os.path.join(dst_dir, "README.md"), "w") as f:
        f.write(
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: data/train-*\n"
            "---\n"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def convert(
    src: str,
    src_format: str,
    dst: str,
    num_shards: int,
    num_proc: int,
    batch_size: int,
) -> None:
    print(f"\n=== {src} -> {dst} ===", flush=True)
    ds = _load_source(src, src_format)
    print(
        f"  loaded {len(ds):,} rows; columns: {ds.column_names}",
        flush=True,
    )
    if "messages" not in ds.column_names:
        raise SystemExit(
            f"source dataset is missing the required `messages` column "
            f"(found: {ds.column_names!r})"
        )

    ds = ds.map(
        _row_worker,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="openai -> swift native",
        load_from_cache_file=False,
    )

    _write_parquet_shards(ds, dst, num_shards)
    print(f"  done: {len(ds):,} rows in {dst}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src",
        required=True,
        help=(
            "Source dataset. With --src-format=disk, this is a directory "
            "produced by `Dataset.save_to_disk`. With --src-format=hub, this "
            "is anything `datasets.load_dataset` can resolve (local dir with "
            "parquet shards, HF hub id, ModelScope id, etc.)."
        ),
    )
    ap.add_argument(
        "--src-format",
        choices=("disk", "hub"),
        default="hub",
        help=(
            "How to load --src: 'disk' for save_to_disk arrow directories, "
            "'hub' for everything else (default)."
        ),
    )
    ap.add_argument(
        "--dst",
        required=True,
        help="Output directory; will be created if missing. Existing data/ is replaced.",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=64,
        help="Number of parquet shards to write under <dst>/data/ (default: 64).",
    )
    ap.add_argument(
        "--num-proc",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel workers for the .map() pass (default: all CPUs).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per .map() batch (default: 1000).",
    )
    args = ap.parse_args(argv)

    convert(
        src=args.src,
        src_format=args.src_format,
        dst=args.dst,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
    )
    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
