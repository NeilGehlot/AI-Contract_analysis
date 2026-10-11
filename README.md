# Contract Analyzer

This tool reads an English PDF contract, pulls out parties, dates, and amounts, and marks each clause HIGH, MEDIUM, or LOW risk. Three engines ship in this repo: keyword rules, sentence embeddings plus logistic regression, and LEGAL-BERT. It is a classroom and research helper, not legal advice.

## Features

```
PDF -> extract text -> clean -> entities -> contract type -> risk -> JSON / TXT / PDF
```

| Stage | What it does |
| --- | --- |
| Text extraction | pdfplumber, then PyPDF2 |
| Cleaning | Whitespace and length cleanup |
| Entities | Parties, dates, amounts, and terms |
| Contract type | Keyword scoring over common agreement types |
| Risk | One clause at a time, HIGH, MEDIUM, or LOW |
| Reports | JSON, text, and PDF |

## Risk engines

The default in `config.yaml` is `rules` for risk and `spacy` for entities. Neither needs trained weights.

| Engine | How to turn it on |
| --- | --- |
| Rules | `risk_detection.engine: "rules"` (default) |
| Sentence embeddings + logistic regression | `risk_detection.engine: "fast_ai"` after `scripts/train_risk_classifier.py` |
| LEGAL-BERT | After training, set `risk_detection.engine: "legal_bert"` and `legal_bert_model_path` to `data/models/risk_classifier_final`. For entities, set `entity_extraction.engine: "legal_bert"` after NER training. |

The results table also includes a TF-IDF plus logistic regression baseline. That baseline was evaluated separately and is not included as an engine.

## Quick start

```bash
python -m venv venv
```

Windows: `venv\Scripts\activate`

macOS or Linux: `source venv/bin/activate`

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_analyzer.py data/raw/sample_contract.pdf
```

Reports are written to `outputs/reports/`.

## Training

Raw CUAD and trained weights are not in this repo. See `data/README.md` for the CUAD download. Training writes checkpoints under `data/models/`.

```bash
python scripts/build_training_data_from_cuad.py
python scripts/train_risk_classifier.py
python scripts/train_legal_bert_risk.py
python scripts/build_ner_training_data_from_cuad.py
python scripts/train_legal_bert_ner.py
```

LEGAL-BERT risk training needs a CUDA GPU. On an RTX 3050 4 GB laptop GPU, three training runs took about 35 minutes.

## Evaluation

A held-out CUAD test split of 77 contracts (1,650 clauses), split by contract, mean over three seeds. These figures are recorded here as plain values. This repo does not include the prediction files or the weights, so the table cannot be regenerated from this copy alone.

| Engine | Accuracy | Weighted F1 | Macro F1 | HIGH F1 |
| --- | ---: | ---: | ---: | ---: |
| Rules | 0.421 | 0.283 | 0.240 | 0.098 |
| Sentence embeddings + logistic regression | 0.762 | 0.774 | 0.712 | 0.511 |
| TF-IDF + logistic regression | 0.835 | 0.838 | 0.796 | 0.665 |
| LEGAL-BERT | 0.903 | 0.901 | 0.867 | 0.758 |

## Limits

English clause text only. There is no OCR. Labels come from CUAD category names, not a fresh lawyer review of each sentence. There is no Indian-contract evaluation in these numbers. This is not legal advice.

## License

Code is MIT. See `LICENSE`.

CUAD is used under CC BY 4.0. Cite Hendrycks, Burns, Chen, and Ball, NeurIPS 2021.

## Author

Neil Gehlot, B.Tech Data Science Engineering, Manipal University Jaipur.
