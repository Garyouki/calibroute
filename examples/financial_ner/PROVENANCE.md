# Prediction provenance and scope

These CSVs are author-supplied derived sentence-level experiment outputs.
Their source is the local Financial NER experiment repository at commit
`70df7dca0c5c55509d34bd115ae5740b410d7752`:

- `experiment/results/encoder_sentences.jsonl`, filtered to seed 42.
- `experiment/results/generative_sentences_seed42_fin_valid.jsonl`.
- `experiment/results/generative_sentences_seed42_fin_test.jsonl`.
- `experiment/results/generative_sentences_seed42_finer_ord_test.jsonl`.
- `experiment/results/generative_sentences_seed42_tweetner7_test.jsonl`.

The encoder uses `sent_conf_msp`; generative outputs use `conf_min_sc`.
Correctness is exactly `sent_error == 0` from the source, not entity-level F1.
Conversion retains identifiers, confidence, correctness, domain, seed, and signal;
it does not include source sentences, entity strings, model weights, or credentials.
Each model has 150 FIN validation, 299 FIN test, 300 FiNER-ORD, and 300 TweetNER7
records. These are the available source counts; no missing 300th FIN test record
is invented. The original source has no configured public Git remote.

The checked-in predictions make **evaluation replay** reproducible. They do not
yet make model training or prediction generation independently reproducible:
public source linkage, exact model/prompt configuration, dataset revisions,
acquisition instructions, and dataset-specific redistribution terms still need
an author-reviewed manifest. The software MIT license is not a grant of rights
to upstream datasets or model weights. No paper acceptance or external adoption
claim is inferred from these files.

All displayed results concern this supplied seed and these slices. Confidence
signals differ across models; results do not establish a general model ranking.
FIN test and the 0.02 shift floor were inspected during development, so this is
a retrospective example, not an untouched independent certification set.
