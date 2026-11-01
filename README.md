RIO currently uses reproducible synthetic data for pipeline development. The
synthetic data is not evidence of real-world phishing or intent-model
performance, and generated rows are marked with `synthetic=true`.

## Phase 1: generate and inspect data

From the project root:

```powershell
python -m data_generation.generate_all
```

This writes Dataset A (2,000 phishing email records), Dataset B (1,000 intent
records), Dataset C (1,000 linked transition records), and one metadata JSON
file per dataset to `data/synthetic/`. The command prints class/intent
distributions, missing-value and duplicate checks, template-family diversity,
thread consistency, and representative samples.

Use a different seed only when intentionally creating a new synthetic
generation:

```powershell
python -m data_generation.generate_all --seed 20261031
```

Every record includes a template family and scenario identifier. Future model
training should split by those groups (not random rows) to reduce near-duplicate
leakage. The original generated files should be preserved. Real-world datasets
must be stored separately from `data/synthetic/` with their own provenance and
consent/licensing metadata.

Model training was intentionally kept separate from data generation. The next
phase is phishing classification, followed by intent extraction, transition
detection, and missing-context experiments.

## Phase 2: train the phishing classifier

After generating Dataset A, run:

```powershell
python -m ml.phishing.train
```

This trains Logistic Regression and Random Forest models using TF-IDF email
text plus selected numerical and categorical email features. The split is
grouped by `scenario_id` to prevent the same scenario group from appearing in
both training and testing. Models are saved under `models/phishing/`, and
metrics (including the confusion matrix and false-positive rate) are written
to `reports/phishing_metrics.json`.

The resulting metrics are synthetic-data development metrics only. They must
not be reported as real-world phishing-detection performance.

## Phase 3: train the intent extraction model

Run the reliable TF-IDF baseline:

```powershell
python -m ml.intent.train --skip-semantic
```

The pipeline extracts a requested-action phrase, reports prediction confidence,
and saves `models/intent/tfidf_logistic_regression.joblib` plus
`reports/intent_metrics.json`.

To run the Sentence Transformer comparison, install the optional dependency:

```powershell
python -m pip install sentence-transformers
python -m ml.intent.train
```

The default semantic encoder is `all-MiniLM-L6-v2`. Its model download is
separate from the synthetic dataset and may take time. If it is unavailable,
the pipeline still completes the TF-IDF baseline and records the semantic
comparison as unavailable. Synthetic intent scores do not demonstrate
real-world intent understanding.

## Phase 4: train the intent-shift detector

Run:

```powershell
python -m ml.transition.train
```

This compares a conservative rule-based baseline with Logistic Regression and
Random Forest. Features include whether intents/actions agree, TF-IDF cosine
similarity between the two emails, whether the current intent is sensitive, and
whether the current message contains urgency language. An intent change alone
is therefore not automatically treated as phishing. Results are evaluated
separately for normal continuation, legitimate intent change, and suspicious
intent shift.

Models are saved under `models/transition/` and the synthetic-only report is
written to `reports/transition_metrics.json`.

## Phase 5: missing-context fallback experiment

Run:

```powershell
python -m ml.fallback.train
```

This evaluates the same current email under full history, partial history, and
no previous messages. The model receives only the available text context; it
does not receive a feature indicating that history is missing. Results are
reported separately by transition label in `reports/fallback_metrics.json`,
with models saved under `models/fallback/`.

## Streamlit UI

After generating the datasets and training the models, start the UI:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser UI provides five tabs:

- **Phishing detection**: Real-time analysis with automated feature extraction from raw email text (URLs, domains, HTTPS, urgency cues, attachments, authentication headers) as well as manual parameter overrides. Includes Explainable AI (XAI) feature attribution breakdown.
- **Intent extraction**: Extracts email intent class and requested action with token saliency explanation.
- **Intent shift**: Analyzes intent progression between consecutive conversational turns and assesses security risk (normal continuation, legitimate change, or suspicious intent shift).
- **Missing context**: Fallback robustness comparison across full history, partial history, and isolated turns.
- **Conversation threads**: Explores real-world enterprise conversation threads from email_thread_details.csv, displaying chronological messages, dynamic intent trajectory, and turn-by-turn transition assessments.

## Evaluation & Benchmarking

To run the automated real-time latency and dataset benchmark:

`powershell
python -m evaluation.benchmark
`

## Running Tests

Run the full pytest suite (20 tests covering feature extraction, models, thread processing, and explainability):

`powershell
pytest
`
