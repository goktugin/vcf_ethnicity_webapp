import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os  # For file existence checks


class AncestryPCASVM:
    """
    Performs ancestry analysis using PCA for dimensionality reduction
    and SVM for classification, with cross-validation.
    """

    def __init__(self,
                 genotype_file_path: str,
                 panel_file_url: str,
                 panel_sample_id_col: str = 'sample',
                 panel_population_col: str = 'pop',
                 n_pca_components: int = 20,
                 n_cv_splits: int = 5,
                 random_state: int = 42):
        """
        Initializes the analyzer with file paths and parameters.

        Args:
            genotype_file_path (str): Path to the combined genotype CSV file.
            panel_file_url (str): URL to the panel file for population labels.
            panel_sample_id_col (str): Column name for sample IDs in the panel file.
            panel_population_col (str): Column name for population labels in the panel file.
            n_pca_components (int): Number of principal components to keep.
            n_cv_splits (int): Number of splits for Stratified K-Fold cross-validation.
            random_state (int): Random state for reproducibility.
        """
        self.genotype_file_path = genotype_file_path
        self.panel_file_url = panel_file_url
        self.panel_sample_id_col = panel_sample_id_col
        self.panel_population_col = panel_population_col
        self.n_pca_components = n_pca_components
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state

        self.X = None
        self.y_categorical = None
        self.y_numeric = None
        self.label_encoder = LabelEncoder()
        self.pipeline = None

        print("AncestryPCASVM initialized.")
        print(f"  Genotype file: {self.genotype_file_path}")
        print(f"  Panel file URL: {self.panel_file_url}")
        print(f"  PCA components: {self.n_pca_components}")
        print(f"  CV splits: {self.n_cv_splits}")

    def _load_genotype_data(self):
        """Loads the genotype data (features X)."""
        print(f"\n--- Step 2.1: Loading Feature Matrix (X) from {self.genotype_file_path} ---")
        if not os.path.exists(self.genotype_file_path):
            raise FileNotFoundError(f"Genotype file not found at: {self.genotype_file_path}")

        try:
            self.X = pd.read_csv(self.genotype_file_path, index_col=0)
            print(f"Feature matrix (X) loaded successfully. Shape: {self.X.shape}")
            if self.X.isnull().values.any():
                print(
                    f"  WARNING: Loaded X matrix contains {self.X.isnull().values.sum()} NaN values. These will be handled by the imputer.")
            else:
                print("  Loaded X matrix does not initially contain NaN values.")
        except Exception as e:
            raise ValueError(f"Error loading genotype data: {e}")

    def _load_population_labels(self):
        """Loads population labels (target y) from the panel file and aligns with X."""
        if self.X is None:
            raise ValueError("Genotype data (X) must be loaded before loading population labels.")

        print(f"\n--- Step 2.2: Preparing Target Variable (y) from Panel File URL ---")
        try:
            print(f"Loading panel file from URL: {self.panel_file_url}")
            panel_df = pd.read_csv(self.panel_file_url, sep='\t')
            print(f"Panel file loaded successfully from URL. Preview:")
            print(panel_df.head())

            if self.panel_sample_id_col not in panel_df.columns:
                raise ValueError(f"Sample ID column '{self.panel_sample_id_col}' not found in panel file.")
            if self.panel_population_col not in panel_df.columns:
                raise ValueError(f"Population label column '{self.panel_population_col}' not found in panel file.")

            panel_df_indexed = panel_df.set_index(self.panel_sample_id_col)
            common_samples = self.X.index.intersection(panel_df_indexed.index)

            if len(common_samples) == 0:
                raise ValueError("No common samples found between genotype matrix (X) and panel file.")

            print(f"\nFound {len(common_samples)} common samples between genotype matrix and panel file.")
            if len(common_samples) < len(self.X.index):
                print(
                    f"  WARNING: {len(self.X.index) - len(common_samples)} samples from X matrix not found in panel file and will be excluded.")

            self.X = self.X.loc[common_samples]
            self.y_categorical = panel_df_indexed.loc[common_samples, self.panel_population_col]

            print(f"X matrix filtered to common samples. New shape: {self.X.shape}")
            print(
                f"Categorical target variable (y_categorical) created from '{self.panel_population_col}' column. Shape: {self.y_categorical.shape}")
            print(f"Unique population/country labels and their counts:\n{self.y_categorical.value_counts()}")

            if self.y_categorical.isnull().any():
                print(
                    f"  WARNING: Target variable (y_categorical) contains {self.y_categorical.isnull().sum()} missing population labels. These samples will be removed.")
                valid_indices = self.y_categorical.dropna().index
                self.X = self.X.loc[valid_indices]
                self.y_categorical = self.y_categorical.loc[valid_indices]
                print(
                    f"  After removing samples with missing y labels, X shape: {self.X.shape}, y_categorical shape: {self.y_categorical.shape}")

            if self.X.empty or self.y_categorical.empty:
                raise ValueError(
                    "Feature matrix (X) or target variable (y_categorical) is empty after filtering for common samples and NaNs.")

        except Exception as e:
            raise ValueError(f"Error loading population labels or aligning data: {e}")

    def _preprocess_labels(self):
        """Encodes categorical labels to numeric."""
        if self.y_categorical is None:
            raise ValueError("Categorical labels (y_categorical) must be loaded before preprocessing.")

        self.y_numeric = self.label_encoder.fit_transform(self.y_categorical)
        print(f"\nCategorical target variable converted to numeric (y_numeric). Shape: {self.y_numeric.shape}")
        print(
            f"  Class labels and their numeric encodings: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        # print(f"  Numeric y distribution:\n{pd.Series(self.y_numeric).value_counts(normalize=True)}")

    def _build_pipeline(self):
        """Builds the machine learning pipeline."""
        print("\n--- Step 3: Defining SVM Model and Preprocessing Pipeline ---")
        svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True,
                        random_state=self.random_state, class_weight='balanced')
        print(f"SVM Model: {svm_model}")

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.n_pca_components, random_state=self.random_state)),
            ('svm', svm_model)
        ])
        print(f"Pipeline created with {self.n_pca_components} PCA components.")

    def run_cross_validation(self):
        """Runs Stratified K-Fold cross-validation and evaluates the model."""
        if self.X is None or self.y_numeric is None or self.pipeline is None:
            raise ValueError("Data (X, y_numeric) must be loaded and pipeline built before running cross-validation.")

        print("\n--- Step 4 & 5: Setting up and Running Cross-Validation ---")
        skf = StratifiedKFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state)
        print(f"{self.n_cv_splits}-fold StratifiedKFold cross-validator configured.")

        fold_metrics = {
            'accuracy': [], 'roc_auc_ovr_weighted': [], 'precision_weighted': [],
            'recall_weighted': [], 'f1_weighted': [], 'logloss': []
        }
        all_y_test_folds = np.array([])
        all_y_pred_folds = np.array([])

        fold_no = 1
        for train_index, test_index in skf.split(self.X, self.y_numeric):
            print(f"--- Fold {fold_no}/{self.n_cv_splits} ---")
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y_numeric[train_index], self.y_numeric[test_index]

            print(f"  Training model on {len(X_train)} samples, {X_train.shape[1]} features...")
            self.pipeline.fit(X_train, y_train)

            print(f"  Predicting on {len(X_test)} test samples...")
            y_pred = self.pipeline.predict(X_test)
            y_pred_proba = self.pipeline.predict_proba(X_test)

            all_y_test_folds = np.concatenate((all_y_test_folds, y_test))
            all_y_pred_folds = np.concatenate((all_y_pred_folds, y_pred))

            # Calculate metrics
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            try:
                if len(np.unique(y_test)) > 1 and len(np.unique(y_train)) > 1:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted',
                                            labels=np.unique(self.y_numeric))
                else:
                    roc_auc = np.nan
                fold_metrics['roc_auc_ovr_weighted'].append(roc_auc)
            except ValueError:
                fold_metrics['roc_auc_ovr_weighted'].append(np.nan)

            fold_metrics['precision_weighted'].append(
                precision_score(y_test, y_pred, average='weighted', zero_division=0, labels=np.unique(self.y_numeric)))
            fold_metrics['recall_weighted'].append(
                recall_score(y_test, y_pred, average='weighted', zero_division=0, labels=np.unique(self.y_numeric)))
            fold_metrics['f1_weighted'].append(
                f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=np.unique(self.y_numeric)))
            try:
                logloss = log_loss(y_test, y_pred_proba, labels=np.unique(self.y_numeric))
                fold_metrics['logloss'].append(logloss)
            except ValueError:
                fold_metrics['logloss'].append(np.nan)

            print(f"  Fold {fold_no} Results: Acc: {fold_metrics['accuracy'][-1]:.4f}, "
                  f"ROC AUC: {fold_metrics['roc_auc_ovr_weighted'][-1] if not np.isnan(fold_metrics['roc_auc_ovr_weighted'][-1]) else 'N/A':<7}, "
                  f"F1: {fold_metrics['f1_weighted'][-1]:.4f}, "
                  f"LogLoss: {fold_metrics['logloss'][-1] if not np.isnan(fold_metrics['logloss'][-1]) else 'N/A':<7}")
            fold_no += 1

        self._report_cv_results(fold_metrics)
        if len(all_y_test_folds) > 0 and len(all_y_pred_folds) > 0:
            self._plot_confusion_matrix(all_y_test_folds, all_y_pred_folds)
        else:
            print("\nWARNING: Could not generate confusion matrix (no test data/predictions).")

    def _report_cv_results(self, fold_metrics):
        """Reports the overall cross-validation results."""
        print("\n--- Step 6: Overall Cross-Validation Results ---")
        for metric_name, values in fold_metrics.items():
            if not all(np.isnan(values)):
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                print(f"Mean {metric_name.replace('_', ' ').capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"Mean {metric_name.replace('_', ' ').capitalize()}: Could not be calculated (all NaN)")

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plots the overall confusion matrix."""
        print("\n--- Plotting Overall Confusion Matrix (All Folds Combined) ---")
        cm_labels_numeric = np.unique(self.y_numeric)
        cm_labels_display = self.label_encoder.inverse_transform(cm_labels_numeric)

        cm = confusion_matrix(y_true, y_pred, labels=cm_labels_numeric)
        cm_df = pd.DataFrame(cm, index=cm_labels_display, columns=cm_labels_display)

        plt.figure(figsize=(max(10, len(cm_labels_display) * 0.6), max(8, len(cm_labels_display) * 0.45)))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Confusion Matrix: Ancestry Prediction (Population/Country Level)', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_figure_filename = "confusion_matrix_country_level_YlOrRd.png"
        try:
            plt.savefig(output_figure_filename, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix figure saved as '{output_figure_filename}'")
        except Exception as e:
            print(f"WARNING: Could not save confusion matrix figure: {e}")

        plt.show()  # This will pause the script until the plot window is closed.

        print("\nConfusion Matrix Values (Numeric):")
        print(cm_df)

    def train_and_evaluate(self):
        """Runs the full analysis pipeline."""
        try:
            self._load_genotype_data()
            self._load_population_labels()
            self._preprocess_labels()
            self._build_pipeline()
            self.run_cross_validation()
            print("\nAnalysis successfully completed.")
        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"\nAN ERROR OCCURRED: {e}")
            # Optionally, re-raise the error if you want the script to stop with a traceback
            # raise
        finally:
            print("--- Analysis script finished ---")


if __name__ == '__main__':
    # Configuration
    GENOTYPE_FILE = "filtrelenmis_genotip_TUM_KROMOZOM_top500_snps.csv"
    PANEL_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
    PCA_COMPONENTS = 20
    CV_SPLITS = 5
    RANDOM_SEED = 42

    # Create an analyzer instance and run the analysis
    analyzer = AncestryPCASVM(
        genotype_file_path=GENOTYPE_FILE,
        panel_file_url=PANEL_URL,
        n_pca_components=PCA_COMPONENTS,
        n_cv_splits=CV_SPLITS,
        random_state=RANDOM_SEED
    )
    analyzer.train_and_evaluate()
