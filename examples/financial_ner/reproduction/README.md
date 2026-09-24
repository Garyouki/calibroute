# Financial NER experiment sources

This directory supplies the source configuration, training/evaluation scripts,
and data acquisition record behind the CalibRoute Financial NER example.
CalibRoute itself still has no model or ML-library dependency.

## Start here

- To replay the existing predictions, follow [the case study](../README.md).
- To obtain the same raw and processed data, run the command below.
- To inspect model training, prompts, and scoring, read `reference/scripts/`.
  These are preserved research sources, not a Windows-supported training pipeline.

From the repository root, with Python 3.10 or newer:

`python examples/financial_ner/reproduction/prepare_data.py --output examples/financial_ner/reproduction/work/data`

This downloads seven small upstream files at fixed repository revisions. It
checks their SHA-256 hashes, reconstructs five raw and five processed JSONL
files, and requires every file to match the historical experiment's hash and
row count. It writes `verification.json` only after these checks pass. No
model, GPU, API key, PyTorch, PyYAML, or Hugging Face client is needed.

After the first successful download, append `--offline` to reuse the cache.
A checksum mismatch is an error; do not update expected hashes to suppress it.
Output under `work/` is ignored by Git and includes upstream dataset text.

## Source and experiment inventory

Eight files in `reference/` are byte-for-byte copies from the author-owned
Financial NER experiment repository at commit
`70df7dca0c5c55509d34bd115ae5740b410d7752`.
`source_manifest.json` records their source paths and hashes, upstream retrieval
URLs, data hashes, prediction-file hashes, and experiment configuration.

| Source | Purpose |
| --- | --- |
| `configs/config.yaml` | Dataset choices, labels, seeds, training and sampling settings |
| `scripts/01_download_data.py` | Original upstream-to-JSONL conversion |
| `scripts/02_preprocess.py` | Label harmonization, filtering, deterministic subsampling |
| `scripts/03_train_encoder.py` | BERT token-classification fine-tuning |
| `scripts/04_eval_encoder.py` | Encoder confidence signals and sentence correctness |
| `scripts/05_train_generative.py` | Qwen LoRA fine-tuning and exact system/user prompts |
| `scripts/06_eval_generative.py` | Greedy and sampled generation, scoring and self-consistency |
| `scripts/common.py` | Entity matching, original device selection and manifest helpers |

The data-only entry point reuses the recovered conversion and harmonization
functions. It reproduces the original shared-RNG sampling order and writes LF
newlines on every platform. It does not invoke the old scripts' ML-dependent
manifest writer.

`historical_manifests.json` contains six historical stage records. Only their
absolute input paths have been shortened to experiment-relative paths; the
hash of each original manifest is retained. Recorded dependency versions and
stage commits are historical observations, not a newly verified environment.

## Data and source terms

| Dataset | Role and final rows | Upstream source |
| --- | --- | --- |
| FIN | Training 1,014; validation 150; test 299 | [tner/fin](https://huggingface.co/datasets/tner/fin) |
| FiNER-ORD | Shifted-domain test, 300 | [gtfintechlab/finer-ord](https://huggingface.co/datasets/gtfintechlab/finer-ord) |
| TweetNER7 | Far-shift test, 300 | [tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7) |

The source manifest links the precise dataset-card revisions inspected during
packaging. FIN's card declares MIT, FiNER-ORD's card declares CC BY-NC 4.0,
and TweetNER7's card declares `other` (consult its upstream terms). CalibRoute's
MIT software license does not replace these dataset terms. Source sentences,
entity text, training checkpoints and model weights are not included here.

Historical downloads used `main` without recording upstream revision IDs.
The pinned retrieval revisions in this package were resolved later. Successful
historical hash checks establish that the reconstructed data matches the saved
experiment data; they do not establish which upstream commit was originally used.

Filtering preserves PER/ORG/LOC. FIN sentences with other entity types are
dropped. TweetNER7 retains only corporation/person/location types, mapped to
ORG/PER/LOC, and cleans its special tokens. The test cap is 300, sampling seed
1234. FIN has 299 rows after filtering; no additional row is fabricated.

## Models, prompts, and correctness

The encoder is `bert-base-cased`, trained for five epochs with learning rate
3e-5, batch size 16 and maximum length 256. The configuration lists seeds
13, 42 and 2026; CalibRoute's bundled CSV uses seed 42.

The generative model is `Qwen/Qwen2.5-0.5B-Instruct`, with LoRA rank 16,
alpha 32, dropout 0.05 on q/k/v/o projections. Its configuration uses three
epochs, learning rate 1e-4, batch size 2, gradient accumulation 8 and maximum
length 256. The case study uses seed 42. The exact prompt is `SYSTEM` plus
`make_messages()` in `05_train_generative.py`, passed through the tokenizer's
chat template. Evaluation uses one greedy decode and five samples, temperature
0.7, top-p 0.95 and at most 128 new tokens. Inspect the source for the exact
sampling-seed and batch behavior.

Encoder sentence confidence is the **minimum across predicted spans of each
span's mean first-subword maximum softmax probability**. It is 1.0 when there
are no predicted spans. Generative `conf_min_sc` is the minimum entity vote
share over five samples, also defaulting to 1.0 for no predictions. These
defaults can be confidently wrong and are retained as part of the experiment.

`correct` is `sent_error == 0`. Encoder errors compare typed token spans in
the retained tokenized window. Generative errors compare normalized text/type
multisets, ignoring span position. Its error flag is `fp + fn > 0`; invalid JSON
is tracked separately and is not automatically an error in that historical
label. These definitions are model-specific and are not entity F1 or a general
ranking of model quality.

## Boundaries of recovered training sources

The original run was on Apple MPS. `common.get_device()` selects MPS or CPU,
not CUDA; generative training hard-codes MPS autocast. Preserve these sources
when studying the original experiment. A Windows/CUDA port should be an
explicit follow-up change with a new run manifest.

The Git snapshot does not include the `requirements.lock` claimed by the old
README or the trained generative LoRA weight file. Base-model revisions were
not pinned. The saved adapter configuration records PEFT 0.20.0, while stage
manifests record Python 3.12.7, PyTorch 2.13.0 and Transformers 5.14.1; this is
not a complete or independently verified installation recipe. Copying scripts
does not reproduce the original stochastic training outputs.

Training/evaluation stages 03-06 are supplied for inspection and future porting;
they have not been re-executed as part of this recovery. Paper-specific plotting
and LaTeX generation are omitted. Use CalibRoute's existing example for reports.
Existing retrospective results remain retrospective; the original inspected
test batches are not new holdout evidence.
