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
