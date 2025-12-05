
# 1. Environmental Humidity Prediction

Time series forecasting of atmospheric moisture levels using LSTM neural networks and ARIMA statistical models.

## Overview

This project develops dual predictive models for environmental humidity from multi-sensor data:
- **LSTM:** Deep learning approach for nonlinear temporal patterns
- **ARIMA:** Statistical approach for interpretable forecasting

## Dataset

**UCI Machine Learning Repository** - Air Quality Dataset

| Metric | Value |
|--------|-------|
| Time Period | March 2004 - April 2005 |
| Frequency | Hourly measurements |
| Total Records | ~8,500+ observations |
| Sensors | 6 environmental sensors |
| Target Variable | Relative humidity (%) |
| Missing Values | ~50% (handled via imputation) |

Notebook includes:
1. Data loading & preprocessing
2. EDA & stationarity tests
3. LSTM training (50 epochs, batch 100)
4. ARIMA fitting (2,1,2)
5. Performance comparison

## Models

### LSTM Model
- **Architecture:** 100 LSTM units → Dense(1)
- **Input:** 8 hourly lookback window
- **Loss:** MAE
- **Optimizer:** Adam

### ARIMA Model
- **Order:** ARIMA(2,1,2)
- **Preprocessing:** Log-transform + differencing
- **Stationarity Test:** ADF test (p < 0.05)

## Key Features

- Recursive label extraction for nested structures
- Log-scale normalization with inverse transformation
- Augmented Dickey-Fuller stationarity verification
- Dual-model comparison (statistical vs. ML)
- Comprehensive visualization (trends, ACF, predictions)
- Confusion matrices & performance metrics

## Technologies

- **Time Series:** statsmodels (ARIMA)
- **Deep Learning:** TensorFlow/Keras (LSTM)
- **Data:** NumPy, Pandas
- **Stats:** scipy, scikit-learn
- **Viz:** Matplotlib, Seaborn

## Results Comparison

**ARIMA Strengths:**
- ✅ 6.61% MAPE (extremely accurate)
- ✅ Interpretable parameters (p=2, d=1, q=2)
- ✅ Statistical inference available
- ✅ Fast computation

**LSTM Strengths:**
- ✅ Captures nonlinear relationships
- ✅ Multi-sensor integration
- ✅ Flexible to architecture changes
- ✅ End-to-end learning

## Future Work

- Implement ensemble methods
- Test higher-order ARIMA models
- Add exogenous variables (temperature, pressure)
- Deploy real-time prediction API
- Compare with Prophet, XGBoost


# 2. Wafer Defect Detection Using CNN

Automated classification of semiconductor wafer surface defects using deep convolutional neural networks.

## Overview

This project detects and classifies 9 wafer defect categories (8 defect types + fault-free) from the LSWMD dataset using a 5-block CNN architecture with batch normalization and spatial dropout.

## Dataset

**LSWMD (Large-Scale Wafer-Map Dataset)** - [Kaggle Link](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

| Metric | Value |
|--------|-------|
| Total Records | 811,457 wafers |
| Labeled Samples | 172,950 (21.3%) |
| Classes | 9 (8 defects + none) |
| Unique Dimensions | 632 aspect ratios |

## Architecture

**CNN Model:** 5 convolutional blocks → Flatten → Dense(512) → Dense(9)
- **Parameters:** 2.5M (9.68 MB)
- **Input:** (224, 224, 3)
- **Output:** 9-class probabilities

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 37% |
| Macro F1-Score | 0.34 |
| Best Class (Center) | Precision: 0.92, Recall: 0.55 |
| Worst Class (Loc) | Precision: 0.00, Recall: 0.00 |

**Note:** Severe class imbalance (78% fault-free) requires weighted loss or oversampling.

## Key Features

- Multi-block CNN with batch normalization
- Geometric augmentation (rotation, scaling, translation, shearing)
- Class-balanced sampling
- One-hot encoding for sparse representations
- Confusion matrix & classification report

## Improvements

- Add class weighting: `class_weight='balanced'` in `.fit()`
- Oversample minority classes (Loc, Scratch)
- Train for 100+ epochs with early stopping
- Transfer learning (ResNet50, EfficientNet)

## Technologies

- **Framework:** TensorFlow 2.19.0 / Keras 3.10.0
- **Data:** NumPy 1.26.4, Pandas 2.0.3
- **Vision:** OpenCV 4.12.0, imgaug 0.4.0
- **ML:** scikit-learn 1.3.2
- **Viz:** Matplotlib, Seaborn
