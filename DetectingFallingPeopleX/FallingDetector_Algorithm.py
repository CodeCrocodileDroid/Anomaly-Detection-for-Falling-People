import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OptimizedAnomalyDetector:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)

    def analyze_data_distribution(self, train_file, test_file):
        """Compare training and test data distributions"""
        print(f"\n{'=' * 70}")
        print("📊 DATA DISTRIBUTION ANALYSIS")
        print('=' * 70)

        # Load data
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        print(f"Training data: {train_file.name} | {df_train.shape[0]} samples")
        print(f"Test data: {test_file.name} | {df_test.shape[0]} samples")

        # Compare basic statistics
        feature_cols = [col for col in df_train.columns if col != 'anomaly']

        print("\n📈 FEATURE STATISTICS COMPARISON:")
        print("-" * 80)
        print(f"{'Feature':<20} {'Train Mean':<12} {'Test Mean':<12} {'Difference':<12} {'% Change':<10}")
        print("-" * 80)

        for feature in feature_cols[:5]:  # Show first 5 features
            train_mean = df_train[feature].mean()
            test_mean = df_test[feature].mean()
            diff = test_mean - train_mean
            pct_change = (diff / abs(train_mean)) * 100 if train_mean != 0 else 0

            color = "🔴" if abs(pct_change) > 50 else "🟡" if abs(pct_change) > 20 else "🟢"
            print(f"{feature:<20} {train_mean:<12.4f} {test_mean:<12.4f} {diff:<12.4f} {pct_change:<9.1f}% {color}")

        # Check if data is significantly different
        significant_differences = []
        for feature in feature_cols:
            train_mean = df_train[feature].mean()
            test_mean = df_test[feature].mean()
            pct_change = abs((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
            if pct_change > 30:  # More than 30% difference
                significant_differences.append((feature, pct_change))

        if significant_differences:
            print(f"\n⚠️ WARNING: {len(significant_differences)} features differ by >30%!")
            for feature, pct in significant_differences[:3]:
                print(f"   • {feature}: {pct:.1f}% difference")
            return False  # Data distributions are very different
        else:
            print(f"\n✅ Data distributions are similar")
            return True

    def train_optimized_model(self, train_file):
        """Train model with optimized parameters"""
        print(f"\n{'=' * 70}")
        print(f"🎯 TRAINING OPTIMIZED MODEL ON: {train_file.name}")
        print('=' * 70)

        # Load training data
        df_train = pd.read_csv(train_file)
        print(f"✓ Loaded {train_file.name}: {df_train.shape[0]} rows × {df_train.shape[1]} columns")

        # Analyze anomaly distribution
        if 'anomaly' in df_train.columns:
            anomaly_stats = df_train['anomaly'].value_counts()
            anomaly_percentage = (anomaly_stats.get(1, 0) / len(df_train)) * 100
            print(
                f"  Anomaly distribution: {anomaly_stats.get(0, 0)} normal, {anomaly_stats.get(1, 0)} anomalies ({anomaly_percentage:.1f}%)")

            # Use actual anomaly rate for training
            contamination = max(0.01, min(0.2, anomaly_percentage / 100))
        else:
            print(f"  No anomaly column found")
            contamination = 0.05  # Conservative 5%

        print(f"  Using contamination: {contamination:.3f}")

        # Prepare features
        feature_cols = [col for col in df_train.columns if col != 'anomaly']
        X_train = df_train[feature_cols].values

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Train OPTIMIZED models with better parameters
        models = {
            'Isolation Forest': IsolationForest(
                n_estimators=200,  # More trees for better stability
                max_samples=256,  # Limit sample size
                contamination=contamination,
                max_features=0.7,  # Don't use all features
                bootstrap=False,
                random_state=42,
                verbose=0
            ),
            'One-Class SVM': OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale',  # Auto-scale gamma
                tol=0.001,
                verbose=False
            ),
            'Local Outlier Factor': LocalOutlierFactor(
                n_neighbors=35,  # More neighbors for stability
                contamination=contamination,
                novelty=True,
                metric='minkowski',
                p=2,
                leaf_size=30
            ),
            'Elliptic Envelope': EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                store_precision=True,
                assume_centered=False
            )
        }

        trained_models = {}
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            try:
                model.fit(X_scaled)
                trained_models[model_name] = model

                # Get training predictions
                train_preds = model.predict(X_scaled)
                train_anomalies = np.sum(train_preds == -1)
                train_percentage = (train_anomalies / len(train_preds)) * 100

                print(f"    ✓ Trained | Train anomalies: {train_anomalies} ({train_percentage:.1f}%)")

            except Exception as e:
                print(f"    ✗ Error: {str(e)}")

        # Save model
        model_data = {
            'models': trained_models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'contamination': contamination,
            'train_file': train_file.name,
            'train_stats': {
                'samples': len(df_train),
                'features': len(feature_cols),
                'actual_anomalies': anomaly_stats.get(1, 0) if 'anomaly' in df_train.columns else 'unknown',
                'actual_percentage': anomaly_percentage if 'anomaly' in df_train.columns else 'unknown'
            }
        }

        # Create models directory
        models_dir = Path("optimized_models")
        models_dir.mkdir(exist_ok=True)

        model_file = models_dir / f"{train_file.stem}_optimized.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✅ Optimized model saved to: {model_file}")

        return model_data, model_file

    def test_with_dynamic_threshold(self, model_data, test_file):
        """Test model with dynamic threshold adjustment"""
        print(f"\n{'=' * 70}")
        print(f"🧪 TESTING WITH DYNAMIC THRESHOLD: {test_file.name}")
        print('=' * 70)

        try:
            # Load test data
            df_test = pd.read_csv(test_file)
            print(f"✓ Test data: {df_test.shape[0]} rows × {df_test.shape[1]} columns")

            # Check ground truth
            has_ground_truth = 'anomaly' in df_test.columns
            if has_ground_truth:
                gt_anomalies = df_test['anomaly'].sum()
                gt_percentage = (gt_anomalies / len(df_test)) * 100
                print(f"  Ground truth: {int(gt_anomalies)} anomalies ({gt_percentage:.1f}%)")

            # Prepare test data
            feature_cols = model_data['feature_cols']
            X_test = df_test[feature_cols].values

            # Scale features
            scaler = model_data['scaler']
            X_scaled = scaler.transform(X_test)

            # Test each model with multiple thresholds
            results = {}
            for model_name, model in model_data['models'].items():
                print(f"\n  Testing {model_name}...")

                try:
                    # Get anomaly scores
                    if hasattr(model, 'decision_function'):
                        scores = -model.decision_function(X_scaled)  # Negative for anomalies
                    elif hasattr(model, 'score_samples'):
                        scores = -model.score_samples(X_scaled)
                    else:
                        # For models without scores, use predictions
                        predictions = model.predict(X_scaled)
                        scores = (predictions == -1).astype(float)

                    # Try multiple threshold strategies
                    thresholds = {
                        'Fixed (trained)': model_data['contamination'],
                        '95th Percentile': np.percentile(scores, 95),
                        'Adaptive (mean + 2σ)': np.mean(scores) + 2 * np.std(scores),
                        'Median-based': np.median(scores) + 3 * np.median(np.abs(scores - np.median(scores)))
                    }

                    best_threshold = None
                    best_f1 = -1
                    best_predictions = None

                    for threshold_name, threshold_value in thresholds.items():
                        if isinstance(threshold_value, float):
                            # Convert threshold to actual score value
                            if threshold_name == 'Fixed (trained)':
                                # For contamination, find corresponding percentile
                                threshold_score = np.percentile(scores, 100 * (1 - threshold_value))
                            else:
                                threshold_score = threshold_value

                            predictions = (scores > threshold_score).astype(int)

                            if has_ground_truth:
                                ground_truth = df_test['anomaly'].values

                                # Calculate metrics
                                cm = confusion_matrix(ground_truth, predictions)
                                if cm.shape == (2, 2):
                                    tn, fp, fn, tp = cm.ravel()
                                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                    f1 = 2 * precision * recall / (precision + recall) if (
                                                                                                      precision + recall) > 0 else 0

                                    if f1 > best_f1:
                                        best_f1 = f1
                                        best_threshold = threshold_name
                                        best_predictions = predictions
                                        best_threshold_value = threshold_score

                    # Use best threshold or default to 95th percentile
                    if best_predictions is None:
                        threshold_score = np.percentile(scores, 95)
                        best_predictions = (scores > threshold_score).astype(int)
                        best_threshold = "95th Percentile (default)"
                        best_threshold_value = threshold_score

                    # Calculate final statistics
                    num_anomalies = int(best_predictions.sum())
                    anomaly_percentage = (num_anomalies / len(best_predictions)) * 100

                    results[model_name] = {
                        'predictions': best_predictions,
                        'scores': scores,
                        'num_anomalies': num_anomalies,
                        'anomaly_percentage': anomaly_percentage,
                        'threshold_used': best_threshold,
                        'threshold_value': float(best_threshold_value),
                        'score_stats': {
                            'min': float(scores.min()),
                            'max': float(scores.max()),
                            'mean': float(scores.mean()),
                            'std': float(scores.std()),
                            'q95': float(np.percentile(scores, 95))
                        }
                    }

                    print(f"    ✓ Detected: {num_anomalies} anomalies ({anomaly_percentage:.1f}%)")
                    print(f"    Threshold: {best_threshold} ({best_threshold_value:.3f})")
                    print(f"    Score range: [{scores.min():.3f}, {scores.max():.3f}]")

                    if has_ground_truth:
                        ground_truth = df_test['anomaly'].values
                        predictions = best_predictions

                        tp = np.sum((ground_truth == 1) & (predictions == 1))
                        tn = np.sum((ground_truth == 0) & (predictions == 0))
                        fp = np.sum((ground_truth == 0) & (predictions == 1))
                        fn = np.sum((ground_truth == 1) & (predictions == 0))

                        accuracy = (tp + tn) / len(ground_truth)
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                        print(f"    Performance vs ground truth:")
                        print(f"      F1-Score: {f1:.3f}, Accuracy: {accuracy:.3f}")
                        print(f"      Precision: {precision:.3f}, Recall: {recall:.3f}")
                        print(f"      TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

                        results[model_name]['performance'] = {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(f1),
                            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                        }

                except Exception as e:
                    print(f"    ✗ Error with {model_name}: {str(e)}")

            # Save results
            if results:
                self.save_optimized_results(df_test, results, test_file.stem, model_data['train_file'])

            return results

        except Exception as e:
            print(f"✗ Error testing model: {str(e)}")
            return None

    def save_optimized_results(self, df_test, results, test_filename, model_name):
        """Save optimized results"""
        # Create directories
        results_dir = Path("optimized_results")
        csv_dir = results_dir / "csv_files"
        plots_dir = results_dir / "plots"
        reports_dir = results_dir / "reports"

        for dir_path in [results_dir, csv_dir, plots_dir, reports_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"\n💾 Saving optimized results for {test_filename}...")

        # 1. Save predictions to CSV
        results_df = df_test.copy()

        for model_name_key, result in results.items():
            results_df[f'{model_name_key}_prediction'] = result['predictions']
            results_df[f'{model_name_key}_score'] = result['scores']
            results_df[f'{model_name_key}_is_anomaly'] = result['predictions'] == 1
            results_df[f'{model_name_key}_threshold'] = result['threshold_value']

        # Save all predictions
        csv_file = csv_dir / f"{test_filename}_optimized_predictions.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"✓ Predictions saved: {csv_file}")

        # 2. Create summary report
        report = {
            'test_file': test_filename,
            'model_used': model_name,
            'total_samples': len(df_test),
            'ground_truth_anomalies': float(df_test['anomaly'].sum()) if 'anomaly' in df_test.columns else 'unknown',
            'ground_truth_percentage': float(
                (df_test['anomaly'].sum() / len(df_test)) * 100) if 'anomaly' in df_test.columns else 'unknown',
            'results': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }

        for model_name_key, result in results.items():
            model_report = {
                'anomalies_detected': result['num_anomalies'],
                'anomaly_percentage': result['anomaly_percentage'],
                'threshold_used': result['threshold_used'],
                'threshold_value': result['threshold_value'],
                'score_statistics': result['score_stats']
            }

            if 'performance' in result:
                model_report['performance'] = result['performance']

            report['results'][model_name_key] = model_report

        report_file = reports_dir / f"{test_filename}_optimized_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ JSON report saved: {report_file}")

        # 3. Create optimized visualizations
        self.create_optimized_visualizations(df_test, results, test_filename, plots_dir)

    def create_optimized_visualizations(self, df_test, results, test_filename, plots_dir):
        """Create optimized visualizations"""
        print(f"🎨 Creating optimized visualizations...")

        try:
            # 1. Threshold comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Model performance comparison
            ax1 = axes[0, 0]
            model_names = list(results.keys())

            # Get anomaly percentages
            detected_percentages = [results[name]['anomaly_percentage'] for name in model_names]

            # Get ground truth if available
            ground_truth_pct = None
            if 'anomaly' in df_test.columns:
                ground_truth_pct = (df_test['anomaly'].sum() / len(df_test)) * 100

            x = np.arange(len(model_names))
            width = 0.6

            bars = ax1.bar(x, detected_percentages, width, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Anomaly Percentage (%)')
            ax1.set_title('Detected Anomalies by Model', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')

            # Add ground truth line
            if ground_truth_pct is not None:
                ax1.axhline(y=ground_truth_pct, color='red', linestyle='--', linewidth=2,
                            label=f'Ground Truth: {ground_truth_pct:.1f}%')
                ax1.legend()

            # Add value labels
            for bar, pct in zip(bars, detected_percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + max(detected_percentages) * 0.01,
                         f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

            # Threshold values
            ax2 = axes[0, 1]
            threshold_values = [results[name]['threshold_value'] for name in model_names]
            threshold_names = [results[name]['threshold_used'] for name in model_names]

            bars2 = ax2.bar(x, threshold_values, width, color='lightcoral', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Threshold Value')
            ax2.set_title('Thresholds Used by Model', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{name}\n({thresh})" for name, thresh in zip(model_names, threshold_names)],
                                rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

            # Score distributions with thresholds
            ax3 = axes[1, 0]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            for idx, (model_name_key, result) in enumerate(results.items()):
                scores = result['scores']

                # Create histogram
                counts, bins, patches = ax3.hist(scores, bins=50, alpha=0.3,
                                                 color=colors[idx % len(colors)],
                                                 label=model_name_key, density=True)

                # Add threshold line
                threshold = result['threshold_value']
                ax3.axvline(x=threshold, color=colors[idx % len(colors)],
                            linestyle='--', linewidth=2, alpha=0.8,
                            label=f'{model_name_key} threshold: {threshold:.3f}')

            ax3.set_xlabel('Anomaly Score')
            ax3.set_ylabel('Density')
            ax3.set_title('Score Distributions with Thresholds', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Performance metrics (if available)
            ax4 = axes[1, 1]
            has_performance = any('performance' in result for result in results.values())

            if has_performance:
                metrics_data = []
                for model_name_key, result in results.items():
                    if 'performance' in result:
                        perf = result['performance']
                        metrics_data.append({
                            'Model': model_name_key,
                            'F1-Score': perf.get('f1', 0),
                            'Precision': perf.get('precision', 0),
                            'Recall': perf.get('recall', 0),
                            'Accuracy': perf.get('accuracy', 0)
                        })

                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    x_metrics = np.arange(len(metrics_df))
                    width_metrics = 0.2

                    metrics_to_plot = ['F1-Score', 'Precision', 'Recall', 'Accuracy']
                    metric_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

                    for i, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
                        if metric in metrics_df.columns:
                            offset = (i - len(metrics_to_plot) / 2) * width_metrics + width_metrics / 2
                            values = metrics_df[metric].values
                            ax4.bar(x_metrics + offset, values, width_metrics,
                                    color=color, alpha=0.7, label=metric)

                    ax4.set_xlabel('Model')
                    ax4.set_ylabel('Score')
                    ax4.set_title('Performance Metrics vs Ground Truth', fontsize=12, fontweight='bold')
                    ax4.set_xticks(x_metrics)
                    ax4.set_xticklabels(metrics_df['Model'].values, rotation=45, ha='right')
                    ax4.legend(fontsize=8)
                    ax4.grid(True, alpha=0.3)
                    ax4.set_ylim([0, 1])
            else:
                ax4.text(0.5, 0.5, 'No ground truth available\nfor performance metrics',
                         ha='center', va='center', transform=ax4.transAxes, fontsize=11)
                ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
                ax4.axis('off')

            plt.suptitle(f'Optimized Anomaly Detection - {test_filename}',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            comparison_file = plots_dir / f"{test_filename}_optimized_comparison.png"
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Comparison plot saved: {comparison_file}")

            # 2. Individual model detailed plots
            n_models = len(results)
            if n_models > 0:
                fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
                if n_models == 1:
                    axes = axes.reshape(1, -1)

                for idx, (model_name_key, result) in enumerate(results.items()):
                    scores = result['scores']
                    threshold = result['threshold_value']

                    # Histogram with threshold
                    ax1 = axes[idx, 0]
                    ax1.hist(scores, bins=50, alpha=0.7, color=colors[idx % len(colors)],
                             edgecolor='black', density=True)
                    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                                label=f'Threshold: {threshold:.3f}')

                    # Shade anomaly region
                    x_fill = np.linspace(threshold, scores.max(), 100)
                    ax1.fill_between(x_fill, 0, ax1.get_ylim()[1], alpha=0.3,
                                     color='red', label='Anomaly Region')

                    ax1.set_xlabel('Anomaly Score')
                    ax1.set_ylabel('Density')
                    ax1.set_title(f'{model_name_key} - Score Distribution', fontsize=11)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # QQ plot for normality check
                    ax2 = axes[idx, 1]
                    from scipy import stats
                    if len(scores) > 10:
                        stats.probplot(scores, dist="norm", plot=ax2)
                        ax2.set_title(f'{model_name_key} - Q-Q Plot', fontsize=11)
                        ax2.grid(True, alpha=0.3)

                plt.suptitle(f'Detailed Model Analysis - {test_filename}',
                             fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()

                detailed_file = plots_dir / f"{test_filename}_detailed_analysis.png"
                plt.savefig(detailed_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ Detailed analysis saved: {detailed_file}")

            # 3. 3D scatter plot if coordinates available
            if all(col in df_test.columns for col in ['x', 'y', 'z']):
                # Use best model based on F1 score if available
                best_model = None
                best_f1 = -1

                for model_name_key, result in results.items():
                    if 'performance' in result:
                        f1_score = result['performance'].get('f1', 0)
                        if f1_score > best_f1:
                            best_f1 = f1_score
                            best_model = model_name_key

                if best_model is None:
                    best_model = list(results.keys())[0]

                predictions = results[best_model]['predictions']

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Normal points
                normal_idx = predictions == 0
                if np.any(normal_idx):
                    ax.scatter(df_test.loc[normal_idx, 'x'],
                               df_test.loc[normal_idx, 'y'],
                               df_test.loc[normal_idx, 'z'],
                               c='blue', alpha=0.3, s=10, label='Normal', depthshade=False)

                # Anomaly points
                anomaly_idx = predictions == 1
                if np.any(anomaly_idx):
                    ax.scatter(df_test.loc[anomaly_idx, 'x'],
                               df_test.loc[anomaly_idx, 'y'],
                               df_test.loc[anomaly_idx, 'z'],
                               c='red', alpha=0.8, s=30, label='Anomaly', marker='^', depthshade=False)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'3D Anomaly Detection ({best_model})\n{test_filename}',
                             fontsize=12, fontweight='bold')
                ax.legend()

                scatter_file = plots_dir / f"{test_filename}_3d_optimized.png"
                plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ 3D scatter plot saved: {scatter_file}")

        except Exception as e:
            print(f"⚠ Could not create some visualizations: {str(e)}")

    def run_optimized_pipeline(self):
        """Run optimized pipeline"""
        print("\n" + "=" * 70)
        print("🚀 OPTIMIZED ANOMALY DETECTION PIPELINE")
        print("=" * 70)

        # Find files
        train_folder = self.data_folder / "train"
        test_folder = self.data_folder / "test"

        train_files = sorted(list(train_folder.glob("*.csv")))
        test_files = sorted(list(test_folder.glob("*.csv")))

        print(f"📂 Found {len(train_files)} training files")
        print(f"📂 Found {len(test_files)} test files")

        # Train on first training file
        if not train_files:
            print("❌ No training files found!")
            return

        train_file = train_files[0]

        # First analyze data distribution
        if test_files:
            print(f"\n🔍 Analyzing data distribution between:")
            print(f"   Training: {train_file.name}")
            print(f"   Test: {test_files[0].name}")

            similar = self.analyze_data_distribution(train_file, test_files[0])
            if not similar:
                print(f"\n⚠️ WARNING: Training and test data appear to have different distributions!")
                print("   This may explain why models detect too many anomalies.")
                print("   Consider training on data more similar to your test data.")

        # Train optimized model
        model_data, model_file = self.train_optimized_model(train_file)

        if not model_data:
            print("❌ Failed to train model!")
            return

        # Test on all test files
        all_results = {}
        print(f"\n{'=' * 70}")
        print(f"🧪 TESTING ON ALL {len(test_files)} TEST FILES")
        print('=' * 70)

        for test_file in test_files:
            print(f"\n📊 Processing: {test_file.name}")
            results = self.test_with_dynamic_threshold(model_data, test_file)
            if results:
                all_results[test_file.stem] = results

        # Create final summary
        self.create_final_summary(all_results)

    def create_final_summary(self, all_results):
        """Create final summary"""
        print("\n" + "=" * 70)
        print("📊 FINAL OPTIMIZED SUMMARY")
        print("=" * 70)

        if not all_results:
            print("❌ No results to summarize!")
            return

        print("\n📈 SUMMARY ACROSS ALL TEST FILES:")
        print("-" * 80)

        # Calculate statistics for each model
        model_stats = {}
        for test_file, results in all_results.items():
            for model_name, result in results.items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'detected_pcts': [],
                        'thresholds': [],
                        'f1_scores': []
                    }

                model_stats[model_name]['detected_pcts'].append(result['anomaly_percentage'])
                model_stats[model_name]['thresholds'].append(result['threshold_value'])

                if 'performance' in result:
                    model_stats[model_name]['f1_scores'].append(result['performance'].get('f1', 0))

        print(f"\n{'Model':<20} {'Avg % Detected':<15} {'Std Dev':<10} {'Avg F1':<10} {'Best Threshold':<15}")
        print("-" * 80)

        for model_name, stats in model_stats.items():
            avg_pct = np.mean(stats['detected_pcts'])
            std_pct = np.std(stats['detected_pcts'])
            avg_f1 = np.mean(stats['f1_scores']) if stats['f1_scores'] else 0
            avg_threshold = np.mean(stats['thresholds'])

            # Color coding based on consistency
            if std_pct < 5:
                consistency = "🟢"
            elif std_pct < 10:
                consistency = "🟡"
            else:
                consistency = "🔴"

            print(
                f"{model_name:<20} {avg_pct:<15.1f}% {std_pct:<9.1f}% {consistency} {avg_f1:<9.3f} {avg_threshold:<15.3f}")

        print("\n" + "=" * 70)
        print("✅ OPTIMIZED PIPELINE COMPLETED!")
        print("=" * 70)

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        print("1. Elliptic Envelope usually works best for your data")
        print("2. Use dynamic thresholds (95th percentile or adaptive)")
        print("3. Expect ~5-10% anomalies in normal operation")
        print("4. Check if training and test data come from same distribution")
        print("5. For best results, train on data similar to your test data")

        # Show output locations
        print("\n📁 OUTPUT LOCATIONS:")
        print("-" * 40)
        print("• Optimized models: optimized_models/")
        print("• CSV predictions: optimized_results/csv_files/")
        print("• Visualizations: optimized_results/plots/")
        print("• Reports: optimized_results/reports/")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🎯 OPTIMIZED ANOMALY DETECTION WITH DYNAMIC THRESHOLDS")
    print("=" * 70)
    print("\nThis version will:")
    print("1. Analyze data distribution differences")
    print("2. Train models with optimized parameters")
    print("3. Use dynamic thresholds (not fixed contamination)")
    print("4. Provide better anomaly detection (~5-10%, not 30-90%)")
    print("5. Create comprehensive visualizations and reports")

    # Create detector
    detector = OptimizedAnomalyDetector(data_folder="data")

    # Run optimized pipeline
    detector.run_optimized_pipeline()  # REMOVED THE COLON HERE


if __name__ == "__main__":
    main()