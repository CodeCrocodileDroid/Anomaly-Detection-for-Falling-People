import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SimpleAnomalyDetector:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.model_dir = Path("anomaly_results/models")

    def check_test_files(self):
        """Check what test files are available"""
        train_folder = self.data_folder / "train"
        test_folder = self.data_folder / "test"

        train_files = sorted(list(train_folder.glob("*.csv")))
        test_files = sorted(list(test_folder.glob("*.csv")))

        print("📂 TRAINING FILES:")
        for f in train_files[:5]:  # Show first 5
            print(f"  • {f.name}")
        print(f"  ... and {len(train_files) - 5} more" if len(train_files) > 5 else "")

        print(f"\n📂 TEST FILES:")
        for f in test_files:
            print(f"  • {f.name}")

        print(f"\n🔍 MATCHING TEST FILES:")
        for train_file in train_files[:5]:  # Check first 5
            base_name = train_file.stem
            matching = [f.name for f in test_files if base_name in f.name]
            if matching:
                print(f"  {train_file.name} → {matching}")
            else:
                print(f"  {train_file.name} → NO MATCH")

        return train_files, test_files

    def load_and_test_model(self, model_file, test_file):
        """Load a trained model and test it"""
        print(f"\n{'=' * 60}")
        print(f"TESTING: {test_file.name} with {model_file.name}")
        print('=' * 60)

        try:
            # Load the trained model
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            print(f"✓ Loaded model trained on: {model_data.get('train_file', 'unknown')}")
            print(f"  Features: {len(model_data['feature_cols'])}")

            # Load test data
            df_test = pd.read_csv(test_file)
            print(f"✓ Loaded test data: {df_test.shape[0]} rows × {df_test.shape[1]} columns")

            # Get feature columns
            feature_cols = model_data['feature_cols']
            X_test = df_test[feature_cols].values

            # Scale using the saved scaler
            scaler = model_data['scaler']
            X_scaled = scaler.transform(X_test)

            # Test each model
            results = {}
            for model_name, model in model_data['models'].items():
                print(f"\n  Running {model_name}...")

                try:
                    # Get predictions
                    predictions = model.predict(X_scaled)

                    # Convert to 0/1 (1 = anomaly, 0 = normal)
                    anomaly_predictions = (predictions == -1).astype(int)

                    # Get scores
                    if hasattr(model, 'decision_function'):
                        scores = -model.decision_function(X_scaled)
                    elif hasattr(model, 'score_samples'):
                        scores = -model.score_samples(X_scaled)
                    else:
                        scores = anomaly_predictions.astype(float)

                    num_anomalies = int(anomaly_predictions.sum())
                    anomaly_percentage = (num_anomalies / len(anomaly_predictions)) * 100

                    results[model_name] = {
                        'predictions': anomaly_predictions,
                        'scores': scores,
                        'num_anomalies': num_anomalies,
                        'anomaly_percentage': anomaly_percentage
                    }

                    print(f"    ✓ Anomalies: {num_anomalies} ({anomaly_percentage:.1f}%)")
                    print(f"    Score range: [{scores.min():.3f}, {scores.max():.3f}]")

                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")

            # Save results to CSV
            self.save_test_results(df_test, results, test_file.stem, model_file.stem)

            # Create visualizations
            self.create_test_visualizations(df_test, results, test_file.stem, model_file.stem)

            return results

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None

    def save_test_results(self, df_test, results, test_filename, model_filename):
        """Save test results to CSV"""
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)

        # Create a copy of dataframe
        results_df = df_test.copy()

        # Add predictions from each model
        for model_name, result in results.items():
            results_df[f'{model_name}_prediction'] = result['predictions']
            results_df[f'{model_name}_score'] = result['scores']
            results_df[f'{model_name}_is_anomaly'] = result['predictions'] == 1

        # Save to CSV
        csv_file = results_dir / f"{test_filename}_{model_filename}_results.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\n✓ Results saved to: {csv_file}")

        # Save summary
        summary = {
            'test_file': test_filename,
            'model_file': model_filename,
            'total_samples': len(df_test),
            'results': {}
        }

        for model_name, result in results.items():
            summary['results'][model_name] = {
                'anomalies_detected': result['num_anomalies'],
                'anomaly_percentage': result['anomaly_percentage'],
                'score_stats': {
                    'min': float(result['scores'].min()),
                    'max': float(result['scores'].max()),
                    'mean': float(result['scores'].mean()),
                    'std': float(result['scores'].std())
                }
            }

        json_file = results_dir / f"{test_filename}_{model_filename}_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {json_file}")

    def create_test_visualizations(self, df_test, results, test_filename, model_filename):
        """Create visualizations for test results"""
        plots_dir = Path("test_results/plots")
        plots_dir.mkdir(exist_ok=True)

        try:
            # 1. Model comparison bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            model_names = list(results.keys())
            anomaly_counts = [results[name]['num_anomalies'] for name in model_names]
            anomaly_percentages = [results[name]['anomaly_percentage'] for name in model_names]

            # Bar chart 1: Counts
            bars1 = ax1.bar(model_names, anomaly_counts,
                            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(model_names)])
            ax1.set_title('Anomalies Detected by Model', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Anomalies')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')

            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + max(anomaly_counts) * 0.01,
                         f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            # Bar chart 2: Percentages
            bars2 = ax2.bar(model_names, anomaly_percentages,
                            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(model_names)])
            ax2.set_title('Anomaly Percentage by Model', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Anomaly Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')

            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + max(anomaly_percentages) * 0.01,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

            plt.suptitle(f'Test Results: {test_filename} | Model: {model_filename}',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            plot_file = plots_dir / f"{test_filename}_{model_filename}_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Comparison plot saved: {plot_file}")

            # 2. Score distribution box plots
            fig, ax = plt.subplots(figsize=(10, 6))

            score_data = [results[name]['scores'] for name in model_names]
            bp = ax.boxplot(score_data, labels=model_names, patch_artist=True)

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(model_names)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title('Anomaly Score Distributions', fontsize=12, fontweight='bold')
            ax.set_ylabel('Anomaly Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # Add mean lines
            for i, scores in enumerate(score_data):
                mean_val = np.mean(scores)
                ax.axhline(y=mean_val, xmin=(i + 0.4) / len(model_names), xmax=(i + 0.6) / len(model_names),
                           color='red', linewidth=2, linestyle='--', alpha=0.7)
                ax.text(i + 1, mean_val, f'μ={mean_val:.3f}', ha='center', va='bottom',
                        fontsize=9, color='red', fontweight='bold')

            plt.tight_layout()
            boxplot_file = plots_dir / f"{test_filename}_{model_filename}_boxplots.png"
            plt.savefig(boxplot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Box plot saved: {boxplot_file}")

            # 3. 3D scatter plot if coordinates available
            if all(col in df_test.columns for col in ['x', 'y', 'z']):
                # Use first model for coloring
                first_model = list(results.keys())[0]
                predictions = results[first_model]['predictions']

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
                ax.set_title(f'3D Anomaly Detection\n{test_filename}', fontsize=12, fontweight='bold')
                ax.legend()

                scatter_file = plots_dir / f"{test_filename}_{model_filename}_3d_scatter.png"
                plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ 3D scatter plot saved: {scatter_file}")

        except Exception as e:
            print(f"⚠ Could not create some visualizations: {str(e)}")

    def run_all_tests(self):
        """Run tests for all available models and test files"""
        print("\n" + "=" * 70)
        print("🧪 RUNNING ALL TESTS")
        print("=" * 70)

        # Check what files we have
        train_files, test_files = self.check_test_files()

        # Get trained models
        model_files = list(self.model_dir.glob("*.pkl"))
        print(f"\n📦 FOUND {len(model_files)} TRAINED MODELS:")
        for mf in model_files:
            print(f"  • {mf.name}")

        if not model_files:
            print("❌ No trained models found! Train models first.")
            return

        if not test_files:
            print("❌ No test files found!")
            return

        # Test each model on each test file
        for model_file in model_files:
            model_name = model_file.stem.replace('_models', '')
            print(f"\n{'═' * 60}")
            print(f"🧠 USING MODEL: {model_name}")
            print('═' * 60)

            # Find matching test files
            matching_tests = []
            for test_file in test_files:
                # Try different matching strategies
                if model_name in test_file.stem or test_file.stem in model_name:
                    matching_tests.append(test_file)

            if not matching_tests:
                print(f"  ⚠ No matching test files found for {model_name}")
                # Try to use any test file
                matching_tests = test_files[:1]  # Use first test file
                print(f"  Using first test file instead: {matching_tests[0].name}")

            for test_file in matching_tests:
                self.load_and_test_model(model_file, test_file)


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🔍 TESTING ANOMALY DETECTION MODELS")
    print("=" * 70)

    # Create detector
    detector = SimpleAnomalyDetector(data_folder="data")

    # Run all tests
    detector.run_all_tests()

    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print("\n📁 Results saved in 'test_results/' folder:")
    print("  • CSV files with all predictions")
    print("  • JSON summary files")
    print("  • Comparison plots and box plots")
    print("  • 3D scatter plots (if coordinates available)")


if __name__ == "__main__":
    main()