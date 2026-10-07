# Laya vs JEV

Hands-on comparison of two **decision models**, models that answer typed questions about an input with
calibrated probabilities instead of generating text:

- **[Laya](https://huggingface.co/convaiinnovations/laya)**: open weights (Apache-2.0), runs on your own CPU or GPU.
- **[JEV](https://openrouter.ai/typesafe/jev-1.13)** (`typesafe/jev-1.13`): hosted, called over HTTP through the
  [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

Both take the same request (`state` + `questions`), so they can be compared on identical inputs. The task throughout
is routing customer-support tickets:

| question     | type     | answers                                          |
| ------------ | -------- | ------------------------------------------------ |
| `department` | `choice` | `billing` / `technical` / `other`                |
| `urgency`    | `score`  | 0 = not urgent, 1 = soon, 2 = blocking           |
| `churn_risk` | `noul`   | probability that the customer threatens to leave |

## Results at a glance

Measured on an NVIDIA RTX 4070 Ti SUPER, against `typesafe/jev-1.13-20260917`.

**Speed (notebooks 01 and 02)**

| | one request | per request in a batch |
| --- | --- | --- |
| Laya on CPU | ~190 ms | ~150 ms |
| Laya on GPU | ~12 ms | ~4 ms |
| JEV (incl. network) | ~250 ms | ~22 ms (parallel requests) |

JEV costs about **$17 per million requests**; Laya costs nothing per request but needs the hardware.

**Accuracy (notebook 03)**: department / urgency / churn on 16 hand-labelled tickets that no model trained on

| model | accuracy |
| --- | --- |
| Laya, zero-shot | 69% / 56% / 94% |
| **Laya, fine-tuned** (~380 tickets, ~30 s of training) | **94% / 94% / 100%** |
| JEV, zero-shot | 94% / 69% / 94% |

Zero-shot Laya struggles with the catch-all `other` option, off-topic text and prompt-injection attempts. A small
fine-tune fixes those and catches implicit churn ("we're evaluating other vendors") that JEV misses. The test set
is small (one ticket ≈ 6 points), so treat these numbers as a first look, not a benchmark. See the notebooks'
_Takeaways_ sections for the full analysis.

## Notebooks

Run them in order; each one builds on the previous.

| notebook | what it does | needs |
| --- | --- | --- |
| [`01-laya_cpu_vs_gpu.ipynb`](01-laya_cpu_vs_gpu.ipynb) | Laya basics (the three question types, reading results), then latency and batch throughput on CPU vs GPU | CUDA GPU for the GPU half |
| [`02-laya_vs_jev.ipynb`](02-laya_vs_jev.ipynb) | Same questions to zero-shot Laya and JEV: accuracy on 16 labelled tickets incl. edge cases, disagreements, latency, cost | OpenRouter key |
| [`03-laya_finetune_vs_jev.ipynb`](03-laya_finetune_vs_jev.ipynb) | What data fine-tuning needs and how to split it, full fine-tune of Laya, temperature calibration, then base vs fine-tuned vs JEV on held-out tickets | CUDA GPU with ~8 GB free, OpenRouter key |

[`kickoff.py`](kickoff.py) is the minimal Laya quickstart, a single prediction in a few lines.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JinhoKim46/laya-vs-jev.git
cd laya-vs-jev
uv sync                      # installs laya, torch, jupyter, pandas, python-dotenv
cp .env.example .env         # then add your OPENROUTER_API_KEY
uv run jupyter lab
```

Select the project's `.venv` as the notebook kernel.

- The first Laya load downloads the `english` checkpoint (~800 MB) from Hugging Face into `~/.cache/huggingface`.
  Loading takes a few minutes on a slow disk; later runs reuse the cache.
- Notebook 03 writes the fine-tuned model to `checkpoints/laya-support-ft/` (~800 MB, gitignored).
- JEV calls cost a fraction of a cent per notebook run.

Use the fine-tuned model anywhere Laya is used:

```python
from laya import Router

router = Router(models={"english": "checkpoints/laya-support-ft"}, device="cuda")
router.predict("We were billed twice, refund it or we leave.", questions)
```

## Data

[`data/`](data/) holds the support tickets used in notebook 03, written with ChatGPT from two prompts (one chat
each). The prompts are in the notebook's appendix. See [`data/README.md`](data/README.md)
for the format and how the splits are built.

| split | source | purpose |
| --- | --- | --- |
| train | `chatgpt_trainval.jsonl`, ~80% of the label/style cells | updating the weights |
| validation | `chatgpt_trainval.jsonl`, held-out cells | best epoch, calibration temperatures |
| test A | `chatgpt_test.jsonl` (a separate chat) | final comparison |
| test B | 16 hand-written tickets (inside the notebooks) | final comparison on text no LLM wrote |

## Project layout

```text
.
├── 01-laya_cpu_vs_gpu.ipynb
├── 02-laya_vs_jev.ipynb
├── 03-laya_finetune_vs_jev.ipynb
├── kickoff.py               # minimal Laya example
├── data/                    # ticket dataset for notebook 03
├── checkpoints/             # fine-tuned model (created by notebook 03, gitignored)
├── pyproject.toml / uv.lock
└── .env.example             # copy to .env
```

## Limitations

- The training and test-A tickets are synthetic and come from the same kind of prompt, so test A favours the
  fine-tuned model. Test B is independent but only 16 tickets.
- Fine-tuning specialises Laya to these exact questions; changing a question's wording needs re-checking and
  probably re-training.
- Urgency labels are subjective, so read urgency accuracy as agreement with our labelling.
- Timings depend on hardware, and JEV latency also on the network.

## Acknowledgements

- [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache-2.0). The fine-tuning loop follows
  the settings of its official fine-tuning script.
- [JEV](https://openrouter.ai/typesafe/jev-1.13) by TypeSafe, accessed through [OpenRouter](https://openrouter.ai).
