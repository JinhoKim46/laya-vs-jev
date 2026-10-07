# Support-ticket dataset

Synthetic customer messages for a fictional B2B SaaS product ("Acme Cloud"), used to fine-tune and evaluate Laya in
[`03-laya_finetune_vs_jev.ipynb`](../03-laya_finetune_vs_jev.ipynb).

| file | tickets | written by | used as |
| --- | --- | --- | --- |
| `chatgpt_trainval.jsonl` | 470 | ChatGPT, chat 1 | train + validation |
| `chatgpt_test.jsonl` | 100 | ChatGPT, chat 2 (fresh chat) | test A |

## Format

JSON Lines, one ticket per line:

```json
{"ticket": "Could you send me a copy of our September invoice?", "department": "billing", "urgency": 0, "churn": false, "style": "plain"}
```

| field | values |
| --- | --- |
| `ticket` | the customer's message |
| `department` | `billing`, `technical`, `other` |
| `urgency` | `0` not urgent, `1` soon, `2` blocking |
| `churn` | `true` if the customer intends or threatens to cancel, leave or switch |
| `style` | `plain`, `terse`, `detailed`, `negation` (mentions leaving but isn't), `implicit` (leaving without saying so), `injection` (tries to instruct the classifier), `off-topic` |

## How it was made

Labels are set by the prompt: ChatGPT was asked for a fixed number of messages for every
department × urgency × churn × style combination, so each ticket's labels are known by construction. Generators
sometimes write a message that doesn't match its requested labels, so review the hard styles (`negation`,
`implicit`, `injection`) before reusing the data. The exact prompts are in the appendix of notebook 03.

## Splits

- **Validation** holds out whole label/style cells (one style per department × urgency × churn combination), not
  single rows, because tickets written for the same cell are near-paraphrases.
- **Test A** comes from a separate chat so it doesn't share the training data's phrasing.
- Any ticket text that appears in more than one split is removed from the lower-priority split
  (test B > test A > validation > train).
