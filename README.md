# ML Systems and Operations - Loan Risk Assessment

## Dataset Overview
This project uses a **loan approval dataset collected in India** containing demographic and financial information about loan applicants. The dataset includes **~25,000 tabular records** with a mix of numeric, categorical, and textual attributes, meeting the assignment requirements for real business process data.

**Prediction Target**: The model predicts `Risk_Flag` - a binary indicator where 1 represents a flagged risky applicant and 0 represents a non-risky applicant. This application is highly relevant in banking and credit scoring, where loan risk assessments are part of routine decision-making.

**Input Schema**: 
- **Numeric**: Income, Age, Experience, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS  
- **Categorical**: Married/Single, House_Ownership, Car_Ownership, Profession, CITY, STATE
- **Target**: Risk_Flag (binary: 0/1)

**Output Schema**: Binary risk classification (0 = low risk, 1 = high risk)

## Architecture Overview

This MLOps pipeline implements a complete production-ready workflow using a **custom orchestration solution** (equivalent to Metaflow/Prefect) via Docker containerization and shell scripting, ensuring local executability and reproducibility.

## How to Run the Code

### With Docker (Recommended)
Build:
```bash
docker build -t mlops-tests .
```
and run:
```bash
docker run --rm -v ${PWD}/Data:/app/Data -v ${PWD}/Model:/app/Model -v ${PWD}/logs:/app/logs mlops-tests
```
This executes the complete pipeline: data validation → model training → A/B testing → drift monitoring.

### Notes on Docker
The `docker run` command mounts:
- `logs/` for storing test logs
- `Model/` to persist models and versioned artifacts
- `Data/` to ensure the dataset is accessible

Ensure these folders exist locally, or Docker will create them.

## Logging and Reproducibility

Each run generates timestamped logs under `logs/`. Logs support reproducibility, comparison, and debugging throughout the MLOps cycle. All model versions include Git commit hashes for exact code provenance.

---

# Task 1: Pre-training Data Quality Tests

## Overview  
Implements comprehensive data validation using industry-standard testing frameworks, executed as standalone Python modules with clear expectation definitions.

## Test Implementation Strategy
Two primary test categories are implemented, derived from business requirements and domain expertise in loan risk assessment:

### Part 1: Missing Value Validation

**Implementation**: `Tests/test_missing.py`

**Test Framework**: Evidently library for production-grade data quality assessment

**Dataset Segment**: Filtered data for state "Uttar_Pradesh" (representative regional subset)

**Expectation Definition**: **0% missing values** across all features

**Reasoning**: 
- In loan risk assessment, all demographic and financial features are considered essential for accurate risk evaluation
- Missing values in income, employment history, or personal details significantly compromise model reliability
- Business requirement: complete customer profiles are mandatory for regulatory compliance
- This represents a production-grade data segment where full records are required for model inference

**Technical Implementation**:
- Executes on filtered segment: `STATE == 'Uttar_Pradesh'`
- Uses Evidently's `DataQualityTestPreset` for comprehensive null detection
- Enforces strict 0% null threshold to catch any data quality degradation early
- Provides detailed column-level reporting with clear pass/fail indicators

### Part 2: Distribution Validation Tests

**Implementation**: `Tests/test_distribution.py`

#### Income Distribution Test

**Expectation**: Income range [50,000 – 10,000,000] INR with 2% tolerance for edge cases

**Reasoning**:  
- **Lower bound (50,000 INR)**: Based on minimum annual income for working adults in India, ensuring realistic applicant profiles
- **Upper bound (10,000,000 INR)**: Represents 99.99th percentile of dataset, filtering extreme outliers that may indicate data entry errors
- **Business context**: Range captures realistic loan applicant income distribution while excluding anomalies
- **Tolerance**: 2% allowance accounts for legitimate edge cases and seasonal income variations

**Technical Details**:
- Derived from exploratory data analysis and domain expertise
- Captures 99.99% of legitimate applicant incomes
- Filters potential data corruption or entry errors

#### House_Ownership Distribution Test

**Expectation**: Categorical values restricted to `["rented", "owned", "norent_noown"]`

**Reasoning**:
- **Domain constraint**: These represent the only valid housing status categories in the business context
- **Data integrity**: Any deviation indicates data corruption, system integration issues, or data entry errors
- **Regulatory compliance**: Standardized categories required for loan risk assessment frameworks
- **Business requirement**: Clear housing status classification essential for risk modeling

**Why Not Use Evidently for Distribution Tests**:
- Current Evidently version (0.6.7) lacks built-in range and categorical validation
- Newer features (`custom_tests`) in 0.7.x+ incompatible with project dependencies  
- Custom Python assertions provide clear, assertive behavior with better readability
- Enables precise control over business logic and error messaging

---

# Task 2: Pre-deployment Model Pipeline

## Flow Orchestration

**Implementation**: Custom orchestration using `run_all_tests.py` + `entrypoint.sh` + Docker

**Architecture**: Multi-step pipeline with dependency management and error handling

**Execution Flow**:
1. **Data Validation Step**: Execute all Task 1 tests
2. **Model Training Step**: Train RandomForest with versioning  
3. **Model Validation Step**: Robustness and inference testing

## Part 1: Model Training

**Algorithm**: RandomForestClassifier selected for robustness on tabular data and interpretability

**Performance**: Achieved **89.85% accuracy** on test set

**Feature Engineering**: One-hot encoding for categorical variables, maintaining feature interpretability

**Training Configuration**:
- Default: 100 estimators, unlimited depth
- Configurable via command-line parameters for A/B testing
- Reproducible via `random_state=42`

## Part 2: Model Versioning System

**Serialization Format**: `.joblib` (superior to pickle for scikit-learn models)

**Storage Architecture**:
```
Model/
├── Current_Model/
│   ├── model_rf.joblib           # Latest model
│   └── model_metadata.json       # Comprehensive metadata
├── versions/                     # Historical versions
│   └── v_YYYYMMDD_HHMMSS/       # Timestamped snapshots
└── manifest.json                # Version registry
```

**Metadata Schema**:
```json
{
    "model_type": "RandomForestClassifier",
    "trained_at": "ISO-8601 timestamp",
    "n_features": 15,
    "feature_names": ["Income", "Age", ...],
    "n_samples": 20000,
    "accuracy": 0.8985,
    "flow_version": "configuration_id",
    "n_estimators": 100,
    "max_depth": null,
    "git_commit": "abc123..."
}
```

**Code Provenance**: Each model version includes Git commit hash for exact reproducibility

## Part 3: Inference Validation

**Implementation**: `Tests/test_inference.py`

**Validation Checks**:
- Binary output validation (only 0/1 predictions)
- No missing predictions
- Both risk classes represented in output
- Prediction count matches input count

**Error Scenarios**: Validates model behavior under deployment-like conditions

## Part 4: Robustness Testing

**Implementation**: `Model/robustness_check.py`

**Test Strategy**: Synthetic error injection to validate model resilience

**Error Scenarios Tested**:
1. **Missing values**: Simulated data quality issues
2. **Unexpected categorical values**: New categories not seen in training
3. **Type mismatches**: Data type inconsistencies
4. **Malformed inputs**: Edge cases and boundary conditions

**Expectation**: Model should fail gracefully with informative error messages rather than producing invalid predictions

**Reasoning**: In production, models encounter corrupted or unexpected data. Graceful failure prevents downstream system corruption and enables proper error handling workflows.

## Part 5: Error Handling Implementation

**Training Data Size Validation**:
```python
if len(X) < 1000:
    raise ValueError("❌ Not enough training samples (found < 1000 rows).")
```

**Reasoning**: Insufficient training data leads to unreliable models. The 1000-record threshold ensures minimum statistical validity for RandomForest training.

**System Interruption Handling**:
- Comprehensive logging for debugging interrupted training
- Atomic model saving (temporary files + rename)
- Graceful cleanup of incomplete artifacts
- Clear error reporting with actionable guidance

---

# Task 3: Post-deployment Monitoring & Operations

## Part 1: Drift Detection

**Implementation**: `Monitoring/monitor_drift.py`

**Drift Type**: **Feature drift detection** using Jensen-Shannon divergence on Income distribution

**Statistical Method**: Jensen-Shannon divergence chosen for its superior properties:
- **Boundedness**: [0, 1] range enables consistent threshold setting
- **Symmetry**: Order-independent comparison (ref vs current = current vs ref)  
- **Interpretability**: Clear scale from identical (0) to completely different (1)
- **Robustness**: Handles unequal sample sizes effectively

**Threshold**: 0.05 (5% divergence threshold)

**Reasoning**: 
- Income is the primary risk indicator in loan assessment
- 5% divergence threshold balances sensitivity vs false positives
- Historical analysis shows >5% divergence typically indicates significant market changes

**Data Segments**:
- **Reference**: `train_data.csv` (original training distribution)
- **Current**: `unseen_segment.csv` (post-deployment monitoring data)

**Unseen Data Strategy**: Deterministic split using ID hash ensures:
- No data leakage between training and monitoring
- Reproducible test scenarios
- Representative distribution sampling

## Part 2: Flow Versioning & Configuration Tracking

**Implementation**: Configurable hyperparameters via command-line interface

**Versioning Strategy**:
```bash
python Model/model_trainer.py --n_estimators 150 --max_depth 10 --flow_version ab_test_round1
```

**Configuration Tracking**: All parameters stored in:
- `model_metadata.json`: Model-specific configuration
- `manifest.json`: Cross-model version registry  
- `Model/versions/`: Complete configuration snapshots

**Git Integration**: 
```python
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"])
```
Ensures exact code reproducibility for any model version.

## Part 3: A/B Testing Framework

**Implementation**: `Model/AB_Testing/`

### Model Variants Strategy
- **Model A**: 100 trees, depth 10 (conservative approach)
- **Model B**: 150 trees, depth 15 (complex approach)

### Data Splitting Methodology

**Reproducible Randomization**:
```python
hash_vals = df["Id"].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16))
group_a = df[hash_vals % 2 == 0]
group_b = df[hash_vals % 2 == 1]
```

**Advantages**:
- **Deterministic**: Same ID always assigned to same group
- **Balanced**: Approximately 50/50 split
- **Reproducible**: Results consistent across runs
- **No leakage**: Clear separation between test groups

### Performance Comparison

**Execution**: `Model/AB_Testing/ab_test_runner.py`

**Metrics**: Accuracy comparison on held-out test data

**Example Results**:
- Model A: 88.71% accuracy
- Model B: 88.64% accuracy

### Managing Multiple A/B Tests

**Strategy for Concurrent Tests**:

1. **Test Identification**: Use unique `test_id` for each experiment
   ```bash
   python ab_test_runner.py --test_id experiment_2024_q1 --models model_v1,model_v2
   ```

2. **Logging Separation**: 
   ```
   logs/ab_test_experiment_2024_q1_20240609_105026.txt
   logs/ab_test_seasonal_models_20240609_110045.txt
   ```

3. **Configuration Registry**: Maintain experiment metadata
   ```json
   {
     "experiment_2024_q1": {
       "models": ["model_v1", "model_v2"],
       "test_period": "2024-01-01 to 2024-03-31",
       "sample_size": 10000,
       "success_metric": "accuracy"
     }
   }
   ```

4. **Data Isolation**: Use different hash seeds for concurrent tests to ensure non-overlapping groups

5. **Results Aggregation**: Centralized dashboard tracking all active experiments with statistical significance monitoring

This framework supports robust, concurrent experimentation while maintaining data integrity and clear attribution of results.

---

## Production Deployment Readiness

### Docker Integration
Complete containerization ensures:
- **Environment consistency**: Identical execution across dev/staging/prod
- **Dependency isolation**: No conflicts with host system
- **Scalability**: Ready for container orchestration platforms
- **Reproducibility**: Exact environment recreation

### Monitoring & Observability
- **Comprehensive logging**: All operations timestamped and logged
- **Performance tracking**: Model accuracy and inference time monitoring
- **Drift alerting**: Automated warnings when distribution changes detected
- **Version traceability**: Complete audit trail from code to predictions

### Quality Assurance
- **Automated testing**: Full test suite execution before deployment
- **Graceful error handling**: Robust failure modes with informative messages
- **Data validation**: Input quality checks prevent model corruption
- **Model validation**: Output quality assurance and bounds checking

This implementation provides a production-ready MLOps pipeline demonstrating industry best practices for model lifecycle management, testing, monitoring, and deployment.
