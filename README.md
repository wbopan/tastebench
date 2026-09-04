# Taste-Bench

Taste-Bench measures the *taste* of an LLM agent: its ability to choose the better direction at a real decision fork in a long-horizon task.

Each question is a decision fork taken from a recorded agent trajectory. The evaluated model sees the task, the trajectory up to the fork, and two candidate next steps. The later part of the trajectory, which is hidden from the model, identifies the better candidate. The benchmark contains 502 questions mined from software engineering and machine-learning research trajectories without expert annotation.

- Paper: *The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks* (link to be added)
- Data: [`wbopan/tastebench`](https://huggingface.co/datasets/wbopan/tastebench) on Hugging Face
- This repository: the evaluation protocol, the scoring code, and the recorded results of frontier models

## Leaderboard

Accuracy in percent. A question counts as correct only when the model answers it correctly in both option orders, so random guessing scores 25 and a model that always picks the same position scores 0. **Average** is the 1:1 mean of the research and engineering subsets. The four cells cross the construction (Detour, Parallel) with the domain (Engineering, Research). **Unparsed** counts presentations whose final answer could not be read; they count as wrong.

| Model | Average | D-Eng | D-Res | P-Eng | P-Res | Unparsed |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 59.7 | 48.1 | 67.2 | 75.8 | 56.2 | 0 |
| GPT-5.5 | 59.5 | 47.4 | 68.8 | 73.4 | 56.2 | 0 |
| Claude Opus 5 | 55.5 | 35.3 | 64.1 | 71.0 | 64.6 | 0 |
| Grok 4.5 | 54.6 | 47.7 | 57.8 | 61.3 | 56.2 | 10 |
| GPT-5.6 Terra | 54.0 | 40.6 | 57.8 | 70.2 | 58.3 | 0 |
| GLM-5.2 | 53.9 | 40.2 | 59.4 | 64.5 | 60.4 | 17 |
| Claude Sonnet 5 | 51.6 | 36.1 | 60.9 | 62.1 | 56.2 | 0 |
| GPT-5.6 Luna | 49.0 | 32.3 | 57.8 | 67.7 | 50.0 | 0 |
| MiniMax M3 | 45.3 | 34.6 | 39.1 | 64.5 | 56.2 | 3 |
| DeepSeek V4 Flash | 43.3 | 29.3 | 46.9 | 58.1 | 50.0 | 2 |
| GPT-5.4 Mini | 40.1 | 25.2 | 54.7 | 26.6 | 54.2 | 112 |
| Mistral Medium 3.5 | 37.7 | 40.6 | 28.1 | 43.5 | 41.7 | 24 |
| GPT-5.4 Nano | 36.6 | 25.6 | 39.1 | 46.0 | 43.8 | 93 |
| Grok 4.20 Reasoning | 15.7 | 19.9 | 9.4 | 28.2 | 8.3 | 459 |

The per-model summaries behind this table are in `results/<model>/summary.json`. `tb leaderboard` regenerates the table from them.

## How the questions are built

A decision fork is a point where attempts at the same task diverge. We use two constructions.

- **Parallel forks.** Independent attempts at the same task diverge at the same point and end with different recorded outcomes. The shared part before the fork becomes the prefix, the two directions become the candidates, and the outcome of each attempt labels the better one.
- **Detour forks.** An agent takes a direction, abandons it after an observed failure, and recovers inside the same run. The fork is placed right before the abandoned direction. The abandoned direction and the later recovery become the candidates.

The engineering questions come from graded rollouts on SWE-bench and SWE-bench Pro tasks. The research questions come from MALT, the public transcript release of METR, on RE-Bench and HCAST research tasks. Candidate forks pass a rubric and two filters: a question is dropped as trivial when every judge answers it from the candidate wording alone, and it is dropped as undecidable when any judge disagrees with its label after reading the full record.

| Cell | Questions |
|---|---:|
| Detour, Engineering | 266 |
| Detour, Research | 64 |
| Parallel, Engineering | 124 |
| Parallel, Research | 48 |

The mining pipeline is not part of this repository. The released questions are the product.

## Evaluation protocol

The protocol is specified in [`protocol/paired_order_v1.yaml`](protocol/paired_order_v1.yaml) and explained in [`protocol/paired_order_v1.md`](protocol/paired_order_v1.md). In short:

1. **Input.** The model sees the task, the full trajectory prefix up to the fork, and the two candidates. The prefix is rebuilt from the released transcript and credential-shaped text is redacted before it is hashed or sent. When the request exceeds 65,536 tokens, the first 25 rendered lines and the longest possible tail are kept, with an explicit omission marker.
2. **Prompt.** The prompt states that exactly one candidate is better and asks for one line, `ANSWER: X`. It does not request visible chain of thought. Provider-native reasoning settings belong to the per-model config.
3. **Two orders.** Every question is asked twice, once in a seeded option order and once in the exact reverse. The letters are recomputed after each ordering.
4. **Scoring.** The denominator is every released question. Request errors and unparseable outputs count as wrong. The headline accuracy is the fraction of questions answered correctly in both orders. We also report the mean single-order accuracy, the four paired outcomes (CC, CW, WC, WW), and position-choice counts.

Only `query`, the rendered prefix, and the text of each choice reach the evaluated model. The rationale, the outcomes, and the provenance fields never do, and `tb validate` checks that boundary.

## Quick start

```bash
uv sync
uv run tb download            # fetch the release from Hugging Face into ./data
uv run tb validate            # check the 502 questions and the transcript hashes
```

Set your endpoint and key. The endpoint is any OpenAI-compatible `chat/completions` or `responses` URL.

```bash
export TASTEBENCH_API_KEY=...
uv run tb run --model gpt-5.5 --api https://api.openai.com/v1/chat/completions
uv run tb score runs/gpt-5.5/<date>
uv run tb leaderboard results/
```

`tb run` writes one JSON record per question and order under `runs/<model>/<date>/raw/`, plus `summary.json` and a `manifest.json` with the hashes of the config, the protocol, and the release. Runs are resumable, and `--limit N` runs a smoke test on the first N questions. Per-model request settings such as temperature, reasoning effort, and token limits live in [`configs/models.yaml`](configs/models.yaml).

## Data format

The release on Hugging Face has the same layout that the code reads locally:

```text
manifest.json                  item counts and SHA-256 of every file below
items/<cell>.jsonl             one question per line, four cells
transcripts.manifest.json      step counts and checksums of every transcript
transcripts/<traj_id>.json     the recorded trajectories the prefixes are rebuilt from
```

Each question has these fields.

| Field | Meaning |
|---|---|
| `id`, `method`, `dataset`, `task_id` | Identity: `method` is `parallel` or `detour`; `dataset` is the trajectory source |
| `query` | The task given to the agent |
| `reference_traj`, `breakpoint_step` | The transcript and the step at which it is frozen |
| `prefix_text` | The rendered prefix up to the fork |
| `choices[].text` | The two candidate next steps |
| `choices[].is_correct` | The label from the recorded outcome |
| `choices[].outcome` | The recorded outcome of the branch |
| `rationale` | Why the label holds, hidden from the model |
| `quality`, `provenance` | Filter votes and extraction metadata, hidden from the model |

## Citation

```bibtex
@inproceedings{tastebench2026,
  title     = {The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks},
  author    = {},
  booktitle = {},
  year      = {2026}
}
```

## Acknowledgements

The trajectories come from SWE-bench, SWE-bench Pro, and the MALT release of METR, which covers RE-Bench and HCAST tasks. The transcripts are technical agent records, not human-subject data, and credential strings are removed before any hashing or transmission.

## License

MIT. See [LICENSE](LICENSE).
