# Dataset
The used dataset is a loan approval set collected in India, which contains demographic and financial information about loan applicants.
The dataset will be used to predict if a person will get a risk flag or not (Binary indicator of loan risk, where 1
represents a flagged risky applicant and 0 represents a non-risky applicant).
This type of application is highly relevant in banking and credit scoring, where loan risk assessments are part of daily decision-making.

# Task 1
## Scenario  
A new batch of data arrives from a branch located in a specific Indian state. Is this data suitable for model inference and operational use?
## Part 1: Checking Missing or Null Values
### Idea
Since all features are essential in determining whether someone receives a loan, no feature should be missing or null. 
This test ensures that every row contains complete data.

### Reasoning
This test simulates a high-quality, production-ready data segment where full records are required for model inference
or business decision-making. The assumption is that this subset has been cleaned and validated previously. The threshold
of 0% nulls is intentionally strict to capture any unexpected worsening in data quality early in the ML pipeline.

### Implementation
The test is implemented using the Evidently library and structured as a standalone Python script.
It:
- Runs on a filtered segment of the data (STATE == 'Uttar_Pradesh')
- Validates that no missing values are present in any column
- Fails with an explicit error if any nulls are found

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
We define the acceptable income range as [50,000 – 10,000,000] INR.
- The lower bound of 50,000 is chosen based on minimum annual income for working adults in India (assumption: a person needs to be working to get a loan).
- The upper bound of 10 million corresponds to the 99.99th percentile of the dataset and filters out potential data entry errors or anomalies.
- 100% of the current data falls within this range, so this threshold also acts as a safeguard against future data drift or noise.


**Reasoning**:  
This range was chosen based on exploratory data analysis and domain knowledge.
It represents the realistic income range for most applicants. Values outside this range are likely outliers or data entry issues.  
The test allows a 2% margin for rare edge cases.

---

### House_Ownership Distribution Test

**Expectation**:  
Values must be one of: `"rented"`, `"owned"`, `"norent_noown"`.

**Reasoning**:  
These are the documented valid values. Any other value is unexpected and may indicate corrupted or inconsistent data entries.

### Why `evidently` is not used for the distribution tests

The distribution-based tests for `Income` and `House_Ownership` are written in assertion-based style, consistent with 
unit test practices. 

Although `evidently` was used for the null value tests, the version used (`0.6.7`) does not support value range or 
category validation tests out of the box. These features were only introduced in later versions (e.g., `0.7.x` via 
`custom_tests`), which are incompatible with the current pipeline.

Therefore, the tests are implemented as standalone Python functions with clear pass/fail behavior, 
readable output, and justifications provided inline.

## How to Run the Tests

### With Docker
Build:
```bash
docker build -t mlops-tests .
```
and run:
```bash
docker run --rm -v ${PWD}/logs:/app/logs -v ${PWD}/Model:/app/Model mlops-tests
```
### Notes on Docker
The `docker run` command includes volume mounts:
- `logs/` is mounted to preserve test logs from inside the container
- `Model/` is mounted to preserve the trained model, versioned artifacts, and metadata

Make sure these folders exist locally before running the container, or Docker will create them automatically.

### Logging Test Results

Each test run automatically generates timestamped log files under the `logs/` directory:

- `data_log_<timestamp>.txt` captures the output of all data validation tests, including null value checks and distribution checks.
- `inference_log_<timestamp>.txt` contains the results of all post-training inference validation tests.

This separation ensures clarity between data issues and model behavior. All logs preserve detailed test outcomes, which enables reproducibility, historical comparison, and easier debugging throughout the MLOps workflow.


### Test Sets
For the tests two different test sets were used. Both sets are subsets of the original master data where "Uttar_Pradesh" is the state.
This state was chosen since it is the biggest state in the set and therefore representative. In "up_clean.csv" the original subset is used,
in "up_dirty.csv" some values where manipulated so the test should fail.


# Task 2
## Part 1 - Model
We use a Random Forest Classifier to predict the `Risk_Flag`, a binary target indicating whether an applicant is considered risky 
for a loan. The Random Forest model was chosen for its robustness, interpretability, and strong performance on tabular data 
with mixed feature types (numerical and categorical). It handles feature importance estimation well and is less prone to 
overfitting than single decision trees. The trained model achieved an accuracy of 89.85% on the test set, making it a reliable 
baseline for deployment and evaluation.\
To support versioning and traceability, we store the serialized model artifact as a `.joblib` file and automatically generate 
a JSON metadata file upon training. This metadata includes model accuracy, training date, model path, and dataset characteristics 
(such as number of samples and features). This ensures transparency and reproducibility across development iterations. The training step is also automatically executed as part of the `run_all_tests.py` workflow, ensuring that model artifacts are always up to date with the most recent data validation results.


## Part 2 - Inference and Validation
After training the model, we implemented an inference pipeline that loads the saved model and applies it to new, unseen data. 
In our case, we used the cleaned subset `up_clean.csv` to simulate real-world prediction. The output is saved as `predictions.csv`. 
To ensure reliability and maintain data integrity, we implemented post-inference validation checks: we verify that all predictions 
are valid, binary values (0 or 1), and that no missing values occur. This ensures the model behaves as expected in deployment
scenarios and detects any anomalies caused by input formatting or inference logic.


## Part 3 - Post Training / Inference Tests
After model training, we validate the integrity and reliability of the prediction outputs. This includes verifying that all 
prediction values are binary (0 or 1), ensuring no missing predictions exist, and confirming that both target classes are 
represented. These tests are implemented in `test_inference.py` and run automatically via `run_all_tests.py`. This helps catch 
potential issues like model bias, data leakage, or output corruption before the model is deployed. 
Additionally, inference tests now support version-controlled test logging. Results from each test session are saved with a 
timestamp in a dedicated `logs/` folder to provide transparency, traceability, and to enable long-term tracking of model behavior.


## Part 4 - Model and Metadata Versioning
To support reproducibility and traceability, we implemented automated versioning of model artifacts. Each time a model 
is trained, both the model file and its associated metadata are saved into a timestamped version folder under `Model/versions/`. 
A `manifest.json` file is maintained to reference the latest model version. This versioning strategy ensures that we can always 
trace back to the model used in production or during evaluation, supporting effective debugging and auditing. Model versioning is also automatically triggered after each successful training run. Each version is saved with a timestamp in `Model/versions/`, and a `manifest.json` keeps track of the latest version. This ensures a complete audit trail of model evolution over time.


## Part 5 - Summary Report Generation
A human-readable model summary is automatically generated in markdown format (`model_summary.md`) using metadata captured 
after training. This summary includes the model path, feature set, test accuracy, sample size, and timestamp of generation. 
It provides a quick and standardized overview of the model, simplifying communication with stakeholders and helping maintain 
documentation for audits or reviews.

## Part 6 - Robustness and Error Simulation
To validate the model's resilience against poor-quality input data, we implemented a robustness check that simulates typical 
data errors. This includes injecting missing values, introducing unexpected categorical entries (e.g., `"unknown"`), and 
using incorrect data types (e.g., numeric values in place of categorical strings). During inference, the system now performs 
strict input validation by checking for:
- Missing values 
- Invalid or unexpected categorical entries
- Schema mismatches with the original training data

If any of these issues are found, the prediction step fails early and logs a detailed error message. This protects the 
model from silently producing unreliable outputs and helps surface potential data drift or pipeline misconfigurations 
 during deployment. These checks are executed via `robustness_check.py` and integrated into the testing workflow. Each robustness check is logged separately in `logs/robustness_log_<timestamp>.txt`. This isolates critical robustness test failures from standard data validation logs and supports detailed debugging when inference reliability is at risk.

# Task 3 - Post-deployment Monitoring & Drift Detection
## Part 1 - Drift Detection using Jensen-Shannon Divergence
To monitor for data drift after deployment, we implemented a dedicated monitoring flow (`monitor_drift.py`) that analyzes 
changes in the distribution of the Income feature between training and unseen data.

We chose Jensen-Shannon divergence (JS divergence) as our drift metric because it:
- Is symmetric and interpretable (bounded between 0 and 1)
- Does not require matching sample sizes
- Is more interpretable, since a value near 0 means “very similar distributions” and near 1 means “very different”
- Is designed to compare probability distributions, not raw counts

### Choosing the Unseen Segment
To simulate realistic monitoring and post-deployment evaluation, we split the original dataset into two disjoint sets:
- `train_data.csv`: Used for training and internal validation (model sees this data). 
- `unseen_segment.csv`: Held out completely during training. Used exclusively for post-deployment drift monitoring and A/B testing.

The split is random but deterministic, based on a hash of the Id column. This ensures:
- Fair distribution of examples between training and unseen segments.
- Reproducibility across runs.

This setup mimics a real production scenario, where incoming user data is unseen during training, and must be validated for drift and tested for performance consistency.

### Expected Behavior
A JS divergence value below 0.05 is considered stable, while higher values indicate possible drift.

### Logging
Each drift monitoring run automatically writes its results to a timestamped log file under logs/.
The log contains the divergence value and a success or warning message with emoji indicators for easier interpretation. Emojis are kept 
in the log file, and automatically filtered from console output to avoid encoding issues on Windows terminals.

## Part 2 - Flow Versioning and Configuration Tracking
To support experimentation and auditability, the training pipeline now includes full flow configuration tracking:
- The script `model_trainer.py` accepts arguments for `n_estimators`, `max_depth`, and a `flow_version` tag.
- These parameters are recorded in the `model_metadata.json` and also propagated into the `manifest.json`. 
- Each model version folder stores a snapshot of the trained model and its metadata.

Example command:
```bash
python Model/model_trainer.py --n_estimators 150 --max_depth 10 --flow_version ab_test_round1
```
This results in:
- A versioned model saved under `Model/versions/` 
- A metadata file capturing flow-specific hyperparameters and identifiers 
- A `manifest.json` with audit info for the latest model

This setup allows consistent tracking of model lineage and configuration changes — essential for reproducible A/B testing 
and diagnostics in real-world ML operations.

## Part 3 - Offline A/B Testing of Model Variants
To evaluate the performance of different model configurations before deployment, a deterministic and reproducible 
offline A/B test setup was implemented. This helps compare model behavior on the same target population using a clean and
controlled evaluation method.

### Strategy
We trained two versions of the model with different hyperparameters:
- `model_a`: `RandomForestClassifier(n_estimators=100, max_depth=10)`
- `model_b`: `RandomForestClassifier(n_estimators=150, max_depth=15)`

Both models were trained using the same `train_data.csv` file and saved along with their respective metadata 
(`model_a_metadata.json`, `model_b_metadata.json`).

### Dataset Split
The `unseen_segment.csv` file is used as the evaluation basis. It was not used during training or validation. To split it 
reproducibly:
- A hash of the `Id` column to split the dataset is used:
  - Records where `hash(Id) % 2 == 0` → assigned to Group A
  - Records where `hash(Id) % 2 == 1` → assigned to Group B

This ensures:
- Even and fair splitting
- Reproducibility across test runs 
- No overlap between groups

### Implementation
The script `ab_test_runner.py`:
- Loads both trained models and their metadata
- Prepares features based on each model's original feature set (from metadata)
- Applies predictions on each split 
- Computes accuracy for both model versions
- Logs the outcome to `logs/ab_test_log_<timestamp>.txt` (accuracy per model + final comparison message)

### Execution Flow: Forked Prediction Paths
The A/B test script (`ab_test_runner.py`) simulates a forked execution flow by:
- Loading two independently trained model variants (`model_a` and `model_b`)
- Splitting the unseen evaluation dataset into two groups (Group A and Group B) using a deterministic hash on the ID field 
- Applying each model to its corresponding group in parallel code branches

While this is implemented as a linear script, it effectively mimics a forked execution where two flows operate
independently on different inputs using different model configurations. This structure provides a clean 
separation between model paths and supports isolated evaluation for reliable A/B comparisons.

### Managing Multiple A/B Tests
To support multiple concurrent or sequential A/B tests, we rely on:
- Explicit flow version tags (e.g., `ab_test_round1`)
- Model metadata tracking (`model_metadata.json`) that includes the flow version and hyperparameters
- Timestamps in log filenames (e.g., `ab_test_log_20250527_141021.txt`)

In a realistic setup:
- A unique flow_version ID (e.g., `ab_test_depth10_vs_depth15`) could be used to label the test run
- Each test log file would be stored with that version ID and timestamp
- If using a tool like MLflow, test metadata (accuracy, parameters, timing) would be tagged and searchable

#### Hypothetical Solution for Scaling A/B Tests
If multiple A/B tests were to run concurrently, we would:
- Create a registry mapping flow versions → model IDs
- Include a `test_id` parameter in the A/B test runner script
- Log results to `ab_test_<test_id>_<timestamp>.log`
- Store evaluations in a centralized location or experiment tracker for comparison

This would allow for robust historical comparison and reproducibility across test campaigns.