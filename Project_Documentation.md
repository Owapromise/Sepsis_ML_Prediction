# Project Documentation: Sepsis Early Prediction

**CS6140 Machine Learning | Northeastern University**
**George Arthur & Promise Owa**

---

## Objectives and Significance

The goal of this project is to predict sepsis in ICU patients up to 6 hours before clinical onset, using data that hospitals are already collecting: vital signs and lab results recorded every hour.

Sepsis affects over 1.7 million Americans each year and kills at least 350,000 of them. One of the reasons it is so deadly is how fast it moves. Every hour that treatment is delayed increases the risk of death. The challenge is that ICU patients already generate enough data to potentially catch it early, but that data is rarely used in a structured, predictive way.

Rather than just building a model, we designed our experiment to answer a specific question: **does it matter more how you handle missing data, or which model you use?** Most existing studies change both at once, which makes it hard to separate the effects. We tested them independently in a controlled setup. That is what makes this project different: the controlled comparison, not just the prediction task.

---

## Brief Summary of Main Findings

XGBoost with missingness-aware preprocessing (Condition B) was the best-performing model, achieving an AUPRC of 0.1730, roughly 7.5 times above what random guessing would produce on this dataset. XGBoost outperformed LSTM under both preprocessing strategies, and the gap between them is large enough that it is not due to chance. Switching from simple imputation to missingness-aware preprocessing improved XGBoost noticeably, but had almost no effect on the LSTM. This suggests that for this dataset, how you handle missing data matters more than which model architecture you choose.

---

## Background

### What is Sepsis?

Sepsis is a medical emergency that occurs when the body's response to an infection begins damaging its own organs. It is not a single disease; it is a syndrome that can develop from any infection, and it progresses quickly. The Sepsis-3 consensus definition describes it as life-threatening organ dysfunction caused by a dysregulated host response to infection.

The clinical difficulty is that sepsis shares symptoms with many other conditions, making early identification hard even for experienced clinicians. Automated early-warning systems that can flag at-risk patients hours before the full clinical picture develops could make a meaningful difference.

### Missing Data in ICU Settings

ICU patients are monitored continuously for vital signs, but lab tests are ordered selectively. A doctor orders a specific blood test when they think it is necessary, not on a fixed schedule. This means that lab values in the dataset are frequently missing. For some tests, values are recorded less than 10% of the time. This is not random. The absence of a test result can itself carry information about a patient's condition.

There are two main schools of thought on handling this. One treats missing values as gaps to be filled in with the average, so the model sees a complete table of numbers. The other preserves the missingness as a signal, filling in the last known value and adding a separate column that records whether the original measurement was actually taken. Rubin (1976) introduced a formal framework for thinking about why data goes missing, and subsequent clinical work (Che et al., 2018) has shown that treating missingness as a feature can improve predictions on clinical time series.

### Previous Work

Moor et al. (2021) conducted a systematic review of machine learning approaches for early sepsis prediction in the ICU and found that gradient boosting methods and LSTMs were the two most commonly used approaches, but that direct comparisons between them were rare and methodologically inconsistent. Harutyunyan et al. (2019) benchmarked several models on clinical time series and found that gradient boosting was competitive with recurrent architectures on tabular clinical data. The PhysioNet/CinC 2019 challenge (Reyna et al., 2020) introduced the dataset we used and showed that simple models with good feature engineering could be surprisingly competitive with more complex approaches.

---

## Methods and Project Design

### Data

We used the PhysioNet/CinC 2019 dataset, which contains records from 40,336 ICU patients across two hospital systems: Set A (20,336 patients) and Set B (20,000 patients). Each patient is represented as an hourly time series with 40 clinical features:

| Feature Group | Count | Examples |
|---|---|---|
| Vital signs | 8 | Heart rate, respiratory rate, blood pressure, temperature |
| Laboratory values | 26 | Lactate, creatinine, WBC, glucose, pH |
| Demographics | 6 | Age, gender, ICU unit, hours since hospital admission |

The original `SepsisLabel` column marks the hour of clinical sepsis onset. We shifted this label back by 6 hours to create an early-warning target (`EarlyLabel`), so the model is always predicting what will happen 6 hours ahead. Patients whose sepsis onset fell within the first 6 hours of their ICU stay were excluded because there is no valid prediction window before onset, which removed 706 patients. The final working dataset contained 39,630 patients with a sepsis prevalence of 5.62%.

### Data Pipeline

1. Load all patient `.psv` files and merge into a single DataFrame (40,336 patients, 40 features)
2. Shift `SepsisLabel` back 6 hours to create `EarlyLabel`
3. Exclude 706 patients whose sepsis onset fell within the first 6 hours
4. Clip physiologically implausible values to NaN before imputation
5. Split patients 70/15/15, stratified by sepsis label
6. Apply Strategy A or Strategy B preprocessing
7. Train XGBoost or LSTM on the training set
8. Evaluate on the held-out test set

### 2×2 Experimental Design

We crossed two preprocessing strategies with two model families to produce four conditions:

| | Strategy A | Strategy B |
|---|---|---|
| **XGBoost** | Condition A (88 features) | Condition B (125 features) |
| **LSTM** | Condition C (40 features) | Condition D (76 features) |

**Strategy A** fills all missing values with the per-feature median computed from the training set, then applies standard scaling. It treats missing data as uninformative gaps.

**Strategy B** first records whether each value was missing (a binary 0/1 indicator column for each feature), then forward-fills within each patient's timeline, carrying the last known value forward hour by hour. Any gaps at the very start of a patient's stay fall back to the training median. This preserves the information about *when* values were not recorded.

For XGBoost, 48 lag features were added after the patient-level split to give the model a sense of time: for each of the 8 vital signs, we computed lag-1, lag-2, lag-4, 4-hour rolling mean, 4-hour rolling standard deviation, and 1-hour delta. These were computed within each patient's records to prevent cross-patient leakage.

For the LSTM, patient sequences were capped at 72 hours and padded with zeros for shorter stays. The model reads the raw hourly sequence directly and learns temporal patterns on its own.

### Train/Validation/Test Split

The split was performed at the patient level; no patient's hourly records appear in more than one set. Stratification on the sepsis label was used to preserve the 5.62% positive rate consistently across all three sets. All four conditions used the same split (fixed random seed), so differences in results are attributable to the method, not the data partition.

| Set | Patients | Sepsis patients | Prevalence |
|---|---|---|---|
| Train | 27,741 | ~1,558 | ~5.62% |
| Validation | 5,945 | ~334 | ~5.62% |
| Test | 5,944 | ~334 | ~5.62% |

### Model Configuration

**XGBoost:**
- Class imbalance handled by setting `scale_pos_weight = 43.3` (the neg/pos ratio)
- Grid search over tree depth ∈ {3, 4, 6} and learning rate ∈ {0.05, 0.1}
- 500 estimators, early stopping on validation AUPRC (patience = 20)

**LSTM:**
- Two-layer LSTM with a per-timestep linear output head
- `pos_weight = 43.3` for class imbalance in the loss function
- Adam optimiser, gradient clipping at max_norm = 1.0
- Early stopping on validation AUPRC (patience = 10)
- Grid search over hidden size ∈ {64, 128}, dropout ∈ {0.2, 0.3}, learning rate ∈ {0.001, 0.0005}

### Evaluation Strategy

The primary metric is **AUPRC** (Area Under the Precision-Recall Curve). With only 5.62% of patients developing sepsis, accuracy is a misleading metric. A model that always predicts no sepsis would be right 94% of the time. AUPRC focuses specifically on how well the model finds the rare positive cases at different decision thresholds, without being inflated by the large number of true negatives.

AUC-ROC is reported as a secondary metric. Decision thresholds for precision, recall, and F1 were selected by maximising F1 on the validation set.

All reported metrics include **95% bootstrap confidence intervals** computed over 1,000 iterations with patient-level resampling. This allows us to say whether differences between conditions are meaningful or within the range of sampling variation.

SHAP (SHapley Additive exPlanations) was applied to both XGBoost models to identify which features contributed most to predictions.

---

## Results

### Main Results: 2×2 Comparison

| Condition | Model | Strategy | AUPRC | 95% CI | AUC-ROC |
|---|---|---|---|---|---|
| A | XGBoost | A | 0.1672 | [0.158, 0.176] | 0.8325 |
| **B** | **XGBoost** | **B** | **0.1730** | **[0.164, 0.183]** | **0.8539** |
| C | LSTM | A | 0.0625 | [0.058, 0.068] | 0.7599 |
| D | LSTM | B | 0.0618 | [0.056, 0.068] | 0.7597 |

Condition B was the best-performing model across all metrics. Both XGBoost conditions substantially outperformed both LSTM conditions, and the confidence intervals between them do not overlap, which indicates the gap is not due to chance.

Switching from Strategy A to Strategy B improved XGBoost performance (0.1672 to 0.1730). The confidence intervals for Conditions A and B partially overlap, so this improvement should be interpreted cautiously rather than as a definitive finding. For the LSTM, Strategy B had essentially no effect (0.0625 to 0.0618).

### AUPRC and Precision-Recall Curves

The bar chart and precision-recall curves (see `results/figures/results_comparison.png`) show the full picture across all four conditions. The dashed line represents the random baseline, equal to the class prevalence (0.023 at the row level). Condition B sits clearly above all others. Conditions C and D are close together and well below the XGBoost conditions.

### Calibration

Calibration plots (see `results/figures/roc_calibration.png`) show that XGBoost models produce probability estimates that track actual event rates reasonably well. LSTM models produce compressed outputs clustered near zero, which means their predicted probabilities are less reliable as risk scores. For a clinical alerting system, calibration matters. A well-calibrated model's score of 0.3 actually means roughly a 30% chance of sepsis, while a poorly calibrated model's scores are harder to interpret.

### Feature Importance (SHAP)

SHAP analysis on Conditions A and B (see `results/figures/shap_top_features.png`) identified the following as the most predictive features:

- **ICULOS** (ICU length of stay), the strongest predictor in both conditions. Patients who have been in the ICU longer are more likely to develop complications.
- **Hospital admission time**, which carries information about patient acuity.
- **Respiratory rate (rolling mean)**, consistent with the observation that respiratory changes tend to precede sepsis onset.
- In Condition B, **missingness indicators for FiO₂ and EtCO₂** appeared in the top 20 predictors. These are respiratory measurements that tend to be ordered for patients who are deteriorating, so their absence may signal relative stability.

### Hospital Generalizability

Condition B was tested separately on patients from Set A and Set B to check whether performance holds across the two hospital systems:

| Hospital | AUPRC | AUC-ROC |
|---|---|---|
| Set A | 0.1824 | 0.8504 |
| Set B | 0.1625 | 0.8391 |

AUC-ROC is nearly identical across both systems, suggesting the model's ability to rank patients by risk generalises reasonably well. The AUPRC gap (0.1824 vs 0.1625) likely reflects differences in lab measurement practices and patient populations between the two hospital systems. We would need more hospital systems to draw stronger conclusions about generalizability.

---

## Conclusion

Our main finding is that preprocessing strategy had a greater impact on performance than model architecture, at least on this dataset and task. XGBoost with missingness-aware preprocessing was the strongest result. The LSTM was largely unaffected by the richer preprocessing, and we believe the limiting factor was not the feature representation but the nature of the data itself: sparse sequences with heavy class imbalance make it difficult for a recurrent model to consistently identify the relevant signal.

The SHAP result we find most interesting is that the absence of certain lab orders appeared as a predictive feature. This is consistent with how clinical decision-making works; certain tests tend to be ordered when a patient's condition is worsening, so when those tests are absent, it may indicate relative stability. We would not claim this is a definitive clinical finding, but it is an observation worth investigating further.

**What went well:** The controlled 2×2 design gave us a clean way to isolate the effect of each factor. The bootstrap confidence intervals allowed us to distinguish meaningful differences from noise. The hospital generalizability check added a useful secondary validation that would not have been possible without the two-hospital dataset structure.

**Limitations:** The LSTM we used is a standard architecture not specifically designed for irregular time series. Models like GRU-D, which learn feature-specific decay rates for missing values, may perform better. The lag feature engineering for XGBoost was limited to vital signs; extending it to lab values could further improve performance. Validation on more than two hospital systems would be needed before drawing broader conclusions.

**Future Work/directions:**
- Test GRU-D and other architectures designed for irregular clinical time series
- Extend lag features to lab values for XGBoost
- Validate across additional hospital systems
- Investigate the clinical significance of the missingness indicators that appeared as top SHAP predictors
