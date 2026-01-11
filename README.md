# Anomaly-Detection-for-Falling-People
Python Machine Learning


🚨 Anomaly Detection for Falling People Detection
A comprehensive, optimized anomaly detection pipeline for identifying unusual patterns in sensor data, specifically designed for falling people detection scenarios.

📋 Overview
This project implements an advanced anomaly detection system using multiple machine learning algorithms with dynamic thresholding. The system is optimized for detecting falls in sensor data, addressing the common problem of models detecting too many false positives (30-90% anomalies) by implementing smarter thresholding techniques to achieve more realistic detection rates (5-10%).

✨ Features
Multi-Algorithm Approach: Implements four different anomaly detection algorithms:

Isolation Forest

One-Class SVM

Local Outlier Factor (LOF)

Elliptic Envelope (Robust Covariance)

Dynamic Thresholding: Uses multiple threshold strategies instead of fixed contamination rates

Data Distribution Analysis: Compares training/test data to identify distribution shifts

Comprehensive Visualizations: Generates detailed plots and 3D scatter plots

Optimized Parameters: Fine-tuned hyperparameters for better performance

Automatic Report Generation: Creates JSON reports and CSV predictions

📁 Project Structure

<img width="920" height="338" alt="image" src="https://github.com/user-attachments/assets/0d97bc8c-74c3-4bd7-8abe-cd787b41abc1" />

🔧 How It Works
1. Data Analysis
Compares statistical distributions between training and test data

Identifies significant feature shifts (>30% difference)

Warns if data comes from different distributions

2. Model Training
Trains four anomaly detection algorithms

Uses actual anomaly rates for contamination estimation

Implements RobustScaler for feature scaling

Saves trained models as .pkl files

3. Dynamic Thresholding
Tests multiple threshold strategies:

Fixed (trained) contamination rate

95th percentile

Adaptive (mean + 2 standard deviations)

Median-based approach

Selects best threshold based on performance

4. Evaluation & Visualization
Compares model performance

Generates comprehensive visualizations:

Model comparison plots

Score distributions with thresholds

Performance metrics (F1, precision, recall)

3D scatter plots (if x, y, z coordinates available)

📊 Outputs
The pipeline generates:

Trained Models: Saved in optimized_models/

Predictions: CSV files with model predictions and scores

Visualizations: PNG plots comparing models and showing detections

Reports: JSON files with detailed performance metrics

🎯 Performance Metrics
The system evaluates models using:

F1-Score: Harmonic mean of precision and recall

Precision: Proportion of true anomalies among detected anomalies

Recall: Proportion of actual anomalies correctly detected

Accuracy: Overall correctness of predictions

💡 Key Insights from Implementation
Best Practices Identified:
Elliptic Envelope typically performs best for this type of data

Dynamic thresholds (95th percentile or adaptive) outperform fixed thresholds

~5-10% anomalies is realistic for normal operation

Data distribution consistency between train/test is crucial

Common Issues Addressed:
Over-detection: Previous models detected 30-90% anomalies

Threshold sensitivity: Fixed thresholds don't adapt to different datasets

Feature scaling: RobustScaler handles outliers better than StandardScaler


🔮 Future Enhancements
Planned improvements:

Real-time anomaly detection

Deep learning approaches (Autoencoders, LSTMs)

Ensemble methods combining multiple models

Online learning for adaptive thresholding

Integration with streaming data sources

📚 References
Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. ICDM.

Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. Neural Computation.

Breunig, M. M., et al. (2000). LOF: Identifying density-based local outliers. SIGMOD.
