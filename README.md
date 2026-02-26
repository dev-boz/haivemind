# hAIvemind

File-based multi-model consensus loop that runs LLMs through the [OpenCode](https://opencode.ai) CLI until their answers converge.

This is intentionally simple:
- No dedicated arbiter model — consensus emerges from convergence
- Shared memory is the on-disk run directory (prompts + per-round artifacts)
- Stops when enough model drafts are "close enough" (string similarity via `difflib.SequenceMatcher`)
- Zero external Python dependencies (stdlib only)

## Prerequisites

hAIvemind shells out to the `opencode` CLI to call models. You need:

1. **Python 3.10+**
2. **[OpenCode CLI](https://opencode.ai)** installed and configured with at least one provider/model

Verify OpenCode works:

```bash
opencode run -m google/gemini-3-flash-preview --format json "Reply with exactly: OK"
```

If your `opencode` binary is in a non-standard location, set `OPENCODE_BIN`:

```bash
export OPENCODE_BIN=~/.opencode/bin/opencode
```

## Quick start

```bash
git clone https://github.com/dev-boz/haivemind.git
cd haivemind

python haivemind.py \
  --models google/gemini-3-flash-preview,google/gemini-2.5-flash,qwen/qwen3-coder-plus \
  --rounds 4 \
  --min-agree 2 \
  --threshold 0.90 \
  --show-log \
  "Write a bash script that prints hello"
```

By default, model preflight (`--probe`) drops failing providers/models. If that leaves only one model, hAIvemind continues in single-model mode unless you pass `--no-allow-single`.

## Watch it live

Stream each model's output as it generates ("theatre mode") with `--live`.

Compact (prefix each fragment with the model id):

```bash
python haivemind.py \
  --live --live-compact \
  --models google/gemini-3-flash-preview,google/gemini-2.5-flash,qwen/qwen3-coder-plus \
  "Explain X in 5 bullets"
```

Verbose (per-model headers; fragments printed as-is):

```bash
python haivemind.py \
  --live \
  --models google/gemini-3-flash-preview,google/gemini-2.5-flash,qwen/qwen3-coder-plus \
  "Explain X in 5 bullets"
```

## Artifacts

Each run creates a timestamped directory under `tmp/haivemind/`:

```
tmp/haivemind/<timestamp>_<prompt-slug>/
  prompt.txt                        # Original user prompt
  room_rules.txt                    # System prompt given to all agents
  models.requested.txt              # Models you asked for
  models.active.txt                 # Models that actually ran (after probe)
  meta.json                         # Run configuration
  probe.json                        # Preflight results
  history.json                      # All rounds data
  final.md / final.txt              # Final consensus answer
  rounds/
    round_01/
      agent_prompt.txt              # Full prompt sent to agents this round
      <model-slug>.raw.txt          # Raw model output
      <model-slug>.json             # Parsed structured output
      <model-slug>.stderr.txt       # Stderr (if any)
    round_02/
      ...
```

## How it works

1. **Probe** — Each model gets a health check (`"Reply with exactly: OK"`). Failing models are dropped.
2. **Round loop** — All active models receive the user prompt + room rules + previous round's drafts. They respond with structured JSON containing a `draft`, `key_points`, `disagreements`, `changes_made`, and `confidence`.
3. **Consensus check** — Drafts are normalized and compared pairwise. If a group of `--min-agree` models are within `--threshold` similarity, the most central draft (medoid) is chosen as the final answer.
4. **Fallback** — If no consensus after `--rounds`, the most central draft from the last round is selected.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | 4 default models | Comma-separated `provider/model` list |
| `--rounds` | `5` | Maximum rounds |
| `--min-agree` | `3` | Minimum models that must agree |
| `--threshold` | `0.92` | Similarity threshold (0-1) |
| `--timeout` | `180` | Per-model timeout (seconds) |
| `--parallel` | `3` | Max concurrent model calls |
| `--outdir` | `tmp/haivemind` | Base output directory |
| `--opencode-bin` | `opencode` | Path to opencode binary |
| `--opencode-agent` | `compaction` | OpenCode agent name |
| `--probe / --no-probe` | `--probe` | Preflight health check |
| `--probe-timeout` | `60` | Probe timeout (seconds) |
| `--probe-retries` | `1` | Probe retries per model |
| `--live` | off | Stream output live to stderr |
| `--live-compact` | off | Prefix live fragments with model id |
| `--show-log` | off | Print artifact paths to stderr |
| `--allow-single / --no-allow-single` | `--allow-single` | Continue with 1 healthy model |

## Troubleshooting

**Only one model is talking?** Check the run's `history.json` and `*.stderr.txt` under `rounds/`.

Common causes:
- Provider/model id not available for your account
- Expired auth / missing credentials for that provider

Sanity check a model directly:

```bash
opencode run -m <provider/model> --format json "Reply with exactly: OK"
```

## Notes

- The agent prompt forces JSON output so we can extract a `draft` field reliably.
- Some models may not comply; those are kept as raw drafts and recorded under `errors`.
- Convergence is based on `difflib.SequenceMatcher` similarity; for code-heavy outputs you may want a lower threshold.

## License

[MIT](LICENSE)
