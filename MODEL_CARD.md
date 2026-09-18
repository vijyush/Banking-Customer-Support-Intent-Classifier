# Model Card — Banking Support Intent Classifier

## Summary

This repository classifies short English banking-support messages into the 77 BANKING77 intents. It benchmarks two systems:

1. word/character TF-IDF with LinearSVC;
2. frozen MiniLM sentence embeddings with Logistic Regression.

The MiniLM system is the selected model because it achieved the highest validation macro-F1.

## Intended use

- Portfolio demonstration of multiclass NLP, transformer representations, evaluation, and error analysis
- Initial routing or agent-assist suggestions for synthetic/example banking queries
- Demonstration of confidence-based abstention and human handoff

## Out-of-scope use

- Autonomous financial decisions
- Fraud, credit, eligibility, or compliance decisions
- Processing private customer data without appropriate controls
- Production deployment without bank-specific retraining and validation
- Languages or intents outside the dataset's scope

## Data

- Dataset: BANKING77
- Official training rows: 10,003
- Official test rows: 3,080
- Labels: 77
- Minimum/maximum training examples per intent: 35/187
- Train duplicate rows: 4
- Test duplicate rows: 1
- Exact text overlap between official train and test files: 6
- Conflicting labels for identical text: 0

The official split is retained for benchmark comparability. A sensitivity evaluation excluding the six overlapping test texts changes MiniLM accuracy only from 92.76% to 92.75%.

## Model details

### TF-IDF baseline

- Word 1–2 grams, maximum 20,000 features
- Character-within-word 3–5 grams, maximum 25,000 features
- LinearSVC, `C=2.0`
- Handoff confidence: top-two decision-score margin

### Selected transformer-backed model

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Representation: normalized 384-dimensional sentence embeddings
- Classifier: Logistic Regression, `C=5.0`
- Handoff confidence: maximum class probability
- The transformer is frozen, not fine-tuned

## Evaluation protocol

1. Stratify the official training file into 85% development and 15% validation data.
2. Fit both candidates on development data.
3. Select by validation macro-F1.
4. Select a handoff threshold from validation data to reach at least 95% accepted-prediction accuracy while maximizing coverage.
5. Retrain each candidate on all official training examples.
6. Report one final evaluation on the official test split.

Random seed: 42.

## Results

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Test top-3 accuracy |
|---|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 90.46% | 90.91% | 90.91% | 96.82% |
| MiniLM + Logistic Regression | **92.08%** | **92.76%** | **92.74%** | **98.28%** |

MiniLM handoff threshold: 0.4298, selected on validation data.

| Handoff result | Validation | Official test |
|---|---:|---:|
| Automation coverage | 92.67% | 92.69% |
| Accuracy among accepted predictions | 95.83% | 95.90% |
| Messages sent to review | 110 | 225 |

The confidence threshold is a demonstration policy, not a universal operational setting. A real bank would optimize it against agent capacity, customer harm, latency, and the costs of different intent confusions.

## Observed failure modes

- Confusion between semantically close payment and transfer states
- Ambiguity between delivery progress and delivery-time questions
- Short messages without enough context
- Errors among virtual-card acquisition intents
- Confidence may not remain calibrated after domain or language shift

## Limitations

- BANKING77 is a public benchmark, not data from the author's employer or a deployed bank.
- English-only evaluation
- No customer-history or conversation-context features
- No fine-tuning of transformer parameters
- No dedicated probability calibration stage
- The handoff threshold is validated on one random split
- Accuracy does not measure downstream resolution quality or customer satisfaction

## Reproducibility

- Source data, seed, configuration, evaluation outputs, model bundles, and tests are included.
- The pretrained MiniLM encoder is downloaded from Hugging Face on first use and is not committed.
- Exact package ranges are in `requirements.txt`.

