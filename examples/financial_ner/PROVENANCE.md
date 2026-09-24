# Prediction provenance and scope

These CSVs are author-supplied derived sentence-level experiment outputs.
Their source is the local Financial NER experiment repository at commit
`70df7dca0c5c55509d34bd115ae5740b410d7752`:

- `experiment/results/encoder_sentences.jsonl`, filtered to seed 42.
- `experiment/results/generative_sentences_seed42_fin_valid.jsonl`.
- `experiment/results/generative_sentences_seed42_fin_test.jsonl`.
- `experiment/results/generative_sentences_seed42_finer_ord_test.jsonl`.
- `experiment/results/generative_sentences_seed42_tweetner7_test.jsonl`.

The encoder uses `sent_conf_msp` (minimum across predicted spans of each span's
mean first-subword maximum softmax probability); generative outputs use
`conf_min_sc` (minimum predicted-entity self-consistency vote share).
Both signals default to 1.0 when the model predicts no entities.
Correctness is exactly `sent_error == 0` from the source, not entity-level F1.
Conversion retains identifiers, confidence, correctness, domain, seed, and signal;
it does not include source sentences, entity strings, model weights, or credentials.
Each model has 150 FIN validation, 299 FIN test, 300 FiNER-ORD, and 300 TweetNER7
records. These are the available source counts; no missing 300th FIN test record
is invented. The original source has no configured public Git remote.

The checked-in predictions make **evaluation replay** reproducible.
[Recovered experiment sources](reproduction/README.md) now include the original
training/evaluation scripts, exact prompt code, configuration, historical
manifests, dataset-card links, and a portable data preparation command. The
command pins retrieval revisions and requires all ten normalized raw/processed
data files to match the historical Git snapshot's hashes and row counts.

These newly pinned retrieval revisions are not claimed to be the original
download revisions, which were not recorded. Model training and prediction
generation have not been independently reproduced: the recovered training
sources target Apple MPS, the original complete dependency lock is absent, base
model revisions were not pinned, and the trained generative LoRA weights are
not in the source Git snapshot. The software MIT license is not a grant of
rights to upstream datasets or model weights. No paper acceptance or external
adoption claim is inferred from these files.

All displayed results concern this supplied seed and these slices. Confidence
signals differ across models; results do not establish a general model ranking.
FIN test and the 0.02 shift floor were inspected during development, so this is
a retrospective example, not an untouched independent certification set.
