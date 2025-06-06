# Dataset
The used dataset is a loan approval set collected in India, which contains demographic and financial information about loan applicants.
The dataset is used to predict whether an individual will receive a risk flag (a binary indicator of loan risk, where 1
represents a flagged risky applicant and 0 represents a non-risky applicant).
This application is highly relevant in banking and credit scoring, where loan risk assessments are part of routine decision-making.

# Task 1
## Scenario  
A new batch of data arrives from a branch located in a specific Indian state. Is this data suitable for model inference and operational use?
## Part 1: Checking Missing or Null Values
### Idea
As all features are considered essential in determining loan eligibility, no feature is expected to be missing or null. 
This test ensures that every row contains complete data.

### Reasoning
This test simulates a production-grade data segment where full records are required for model inference or business operations. 
It is assumed that this subset has been previously cleaned and validated. A strict threshold of 0% nulls is enforced to capture 
any unexpected degradation in data quality early in the pipeline.

### Implementation
The test is implemented using the Evidently library and structured as a standalone Python script. It:
- Runs on a filtered segment of the data (STATE == 'Uttar_Pradesh')
- Validates that no missing values exist in any column
- Fails with a clear error if any nulls are found

### Requirements
- ✅ Uses a testing framework (Evidently)  
- ✅ Executes as a standalone module (not notebook)  
- ✅ Applies to a data segment  
- ✅ Validates nulls in all columns  
- ✅ Defines an expectation for nulls (value = 0)  
- ✅ Handles failures with clear assertions  
- ✅ Is easy to read, adapt, and extend  

## Part 2: Test Distributions

### Income Distribution Test

**Expectation**:  
An acceptable income range is defined as [50,000 – 10,000,000] INR.
- The lower bound is based on the minimum annual income for working adults in India.
- The upper bound aligns with the 99.99th percentile of the dataset to exclude extreme outliers.
- All data is expected to fall within this range under clean conditions.

**Reasoning**:  
The range is derived from exploratory analysis and domain expertise. It captures realistic applicant incomes, while filtering anomalies.  
A 2% tolerance is accepted for edge cases.

---

### House_Ownership Distribution Test

**Expectation**:  
Allowed values: `"rented"`, `"owned"`, `"norent_noown"`.

**Reasoning**:  
These represent the only valid values. Any deviation signals possible corruption or inconsistencies in the data.

### Why `evidently` is not used for the distribution tests

The distribution tests for `Income` and `House_Ownership` are written using basic assertions, consistent with unit testing. 

The currently used Evidently version (`0.6.7`) does not support range or category checks out of the box.
Newer features (e.g., `custom_tests`) introduced in `0.7.x` are incompatible with this pipeline.

Thus, simple Python-based tests are used to ensure clear, assertive behavior with good readability.

## How to Run the Code

### With Docker
Build:
```bash
docker build -t mlops-tests .
```
and run:
```bash
docker run --rm \
  -v ${PWD}/Data:/app/Data \
  -v ${PWD}/Model:/app/Model \
  -v ${PWD}/logs:/app/logs \
  mlops-tests
```

### Notes on Docker
The `docker run` command mounts:
- `logs/` for storing test logs
- `Model/` to persist models and versioned artifacts
- `Data/` to ensure the dataset is accessible

Ensure these folders exist locally, or Docker will create them.

### Logging Test Results

Each run generates timestamped logs under `logs/`. Logs support reproducibility, comparison, and debugging throughout the MLOps cycle.

### Test Sets
Two test sets are provided, derived from the state "Uttar_Pradesh":
- `up_clean.csv`: the original subset
- `up_dirty.csv`: a manipulated version expected to fail

# Task 2
## Part 1 - Model
A `RandomForestClassifier` is used to predict `Risk_Flag`. It was selected for its robustness on tabular data and interpretability. 
The trained model achieved 89.85% accuracy on the test set.

Model versioning is performed using `.joblib` for artifacts and `.json` for metadata. 
Training also triggers the versioning logic via `run_all_tests.py`, ensuring freshness and traceability.

## Part 2 - Inference and Validation
Post-training inference is executed using `up_clean.csv` as a simulation of deployment conditions.
Outputs are saved as `predictions.csv`.

Post-inference checks ensure:
- Binary values only (0 or 1)
- No missing predictions

This validates reliability in deployment-like settings.

## Part 3 - Post Training / Inference Tests
Post-training evaluations validate:
- All predictions are binary
- No missing values
- Both classes are represented

These checks are part of `test_inference.py` and executed by `run_all_tests.py`.
All outputs are logged with timestamps for reproducibility and audit trails.

## Part 4 - Model and Metadata Versioning
Each model training operation saves:
- The trained model under `Model/model_rf.joblib`
- Metadata to `model_metadata.json`
- A full versioned snapshot in `Model/versions/v_<timestamp>`

The manifest file keeps track of the latest version. This setup supports reproducibility and compliance.

## Part 5 - Summary Report Generation
A model summary (`model_summary.md`) is generated automatically, containing:
- Accuracy
- Training date
- Feature details
- Paths to artifacts

This file enables non-technical stakeholders to interpret model characteristics easily.

## Part 6 - Robustness and Error Simulation
Robustness is evaluated using synthetic data errors:
- Missing values
- Unexpected categorical values
- Type mismatches

If the model encounters such cases, it fails safely with an informative message. These tests run in `robustness_check.py` and log outputs separately.

# Task 3 - Post-deployment Monitoring & Drift Detection

### Unified Docker Execution (Flow Orchestration)
The Dockerfile orchestrates:
- Data validation
- Drift monitoring
- Training and model versioning
- A/B model training
- A/B testing

This unified flow:
- Ensures consistency
- Enables CI/CD-style automation
- Preserves all logs

## Part 1 - Drift Detection using Jensen-Shannon Divergence
Drift is detected using `monitor_drift.py`, which compares `Income` distributions from `train_data.csv` and `unseen_segment.csv`.

JS divergence is chosen for its:
- Boundedness [0, 1]
- Symmetry
- Interpretability
- Applicability to unequal sample sizes

### Dataset Split
Two segments are created:
- `train_data.csv`: used during training
- `unseen_segment.csv`: used only post-training

Splitting is deterministic and based on a hash of the `Id` field.

### Logging
Each run creates a log `drift_log_<timestamp>.txt` under `logs/`, with emojis stripped from console output but kept in logs.

## Part 2 - Flow Versioning and Configuration Tracking

### Configuration Tracking
`model_trainer.py` accepts:
- `--n_estimators`
- `--max_depth`
- `--flow_version`

These parameters are stored in:
- `model_metadata.json`
- `manifest.json`
- `Model/versions/`

Runnable via:
```bash
python Model/model_trainer.py --n_estimators 150 --max_depth 10 --flow_version ab_test_round1
```

### Git-based Code Versioning
To preserve training code provenance, the Git commit hash is saved using:
```bash
git rev-parse HEAD
```
This ensures exact code reproducibility.

## Part 3 - Offline A/B Testing of Model Variants

### Strategy
Two models are trained:
- `model_a`: 100 trees, depth 10
- `model_b`: 150 trees, depth 15

### Dataset Split
`unseen_segment.csv` is used for testing.
A hash of `Id` ensures:
- Reproducibility
- Balanced split
- No leakage between groups

### Implementation
`ab_test_runner.py`:
- Loads each model and metadata
- Prepares features accordingly
- Predicts outcomes
- Logs results to `ab_test_log_<timestamp>.txt`

### Execution Flow: Forked Prediction Paths
The A/B test simulates a flow fork:
- Each model handles a separate group
- Predictions are made independently
- Accuracy is compared

### Managing Multiple A/B Tests
Flow versioning and timestamped logs allow separation of test runs.
For scalability:
- Each test could use a `test_id`
- Logs named `ab_test_<test_id>_<timestamp>.txt`
- A registry or experiment tracker can maintain mappings

This supports robust, concurrent test campaigns with full traceability.
