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

## Example: The car wash debate

Here's a real run showing 3 models debating a trick question across 5 rounds. Two models initially said "walk" but got convinced by the third model's logic — you need your car at the car wash.

```bash
python haivemind.py \
  --live --live-compact \
  --models opencode/big-pickle,modal/zai-org/GLM-5-FP8,opencode/minimax-m2.5-free \
  "The car wash is only 100m from my house, should I drive or walk?"
```

**Round 1** — Two models say walk, one says drive:
- `big-pickle`: "Walk. 100m is a 1-minute walk, driving wastes fuel..." (confidence: 0.95)
- `minimax-m2.5-free`: "Walk. Driving involves starting engine, braking, parking..." (confidence: 0.90)
- `GLM-5-FP8`: "Drive. The purpose of going to a car wash is to wash your car, which requires having your car present." (confidence: 0.95)

**Round 2** — GLM-5's argument lands. minimax flips:
- `minimax-m2.5-free`: "Drive. The critical point from GLM-5 changed my thinking — you need your car at the car wash." (changes_made: "Adopted the driving recommendation based on the logical necessity of having the car present")

**Round 3** — big-pickle concedes too:
- `big-pickle`: "Drive. Changed position from 'walk' to 'drive' based on the valid counterargument that walking leaves your car at home."

**Rounds 4-5** — All three models converge on the same answer. Consensus reached.

**Final output:**
> Drive. The car wash requires your car to be physically present — you cannot wash a vehicle that stays at home. Walking 100m means arriving without your car, then having to walk back and drive anyway, making the walk pointless. The distance is so short that fuel consumption is negligible, and driving is the only option that actually accomplishes the goal of getting your car washed.

This shows the core thesis: no single model is the arbiter. One model spotted the logical flaw, defended it with evidence, and the others independently verified and conceded.

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

## The idea

hAIvemind treats LLMs like people locked in a room who can't leave until they agree on the correct answer. There's no arbiter, no judge, no "smartest model in the room." Consensus emerges from the process:

- Models read each other's drafts and independently verify claims
- They adopt fluid roles — critic, builder, skeptic, synthesizer — based on what the conversation needs
- Sycophancy is countered by the room rules: "do not agree just to agree"
- Coverage beats individual intelligence: Model A catches bug X but misses Y; Model B catches Y but misses X; together they catch both

The core framework is intentionally minimal: a loop, a shared space, and a prompt that creates social dynamics between models. The intelligence emerges from interaction, not from any single model.

See [ORIGIN.md](ORIGIN.md) for more on the premise and theory behind this project.

## Notes

- The agent prompt forces JSON output so we can extract a `draft` field reliably.
- Some models may not comply; those are kept as raw drafts and recorded under `errors`.
- Convergence is based on `difflib.SequenceMatcher` similarity; for code-heavy outputs you may want a lower threshold.

## License

[MIT](LICENSE)
