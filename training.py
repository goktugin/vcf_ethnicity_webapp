import pandas as pd
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class RandomForestTrainer:
    def __init__(self, genotype_file_path, panel_file_url, output_dir, random_state=42):
        self.genotype_file_path = genotype_file_path
        self.panel_file_url = panel_file_url
        self.output_dir = output_dir
        self.random_state = random_state
        self.n_features_to_select = 5000
        self.snps_per_chunk = 7000
        self.n_pca_components = 20
        self.test_size = 0.25
        self.n_estimators = 1000
        self.max_depth = 20
        self.min_samples_split = 5
        self.min_samples_leaf = 2
        self.X = None
        self.y_categorical = None
        os.makedirs(self.output_dir, exist_ok=True)
        print("RandomForestTrainer başlatıldı.")


    def _load_and_prepare_data(self):
        """Genotip ve panel verisini yükler, birleştirir ve sütun isimlerini temizler."""
        print(f"\nAdım 1: Veri Yükleme ve Hazırlama")
        self.X = pd.read_csv(self.genotype_file_path, index_col=0, low_memory=False)
        panel_df = pd.read_csv(self.panel_file_url, sep='\t')
        panel_df_indexed = panel_df.set_index('sample')
        common_samples = self.X.index.intersection(panel_df_indexed.index)
        self.X = self.X.loc[common_samples]
        self.y_categorical = panel_df_indexed.loc[common_samples, 'pop'].copy()
        if self.y_categorical.isnull().any():
            valid_indices = self.y_categorical.dropna().index
            self.X = self.X.loc[valid_indices]
            self.y_categorical = self.y_categorical.loc[valid_indices]

        print("Sütun isimleri temizleniyor...")
        original_columns = self.X.columns.tolist()
        cleaned_columns = [col.strip("'\" ") for col in original_columns]
        self.X.columns = cleaned_columns
        print("Sütun isimleri başarıyla temizlendi.")

        print(f"Veri hazırlandı. Boyut: {self.X.shape}")

    def _perform_feature_selection(self):
        print(f"\n--- Adım 2: En İyi {self.n_features_to_select} Özelliğin Seçilmesi ---")
        imputer = SimpleImputer(strategy='most_frequent')
        X_imputed = imputer.fit_transform(self.X)
        self.X = pd.DataFrame(X_imputed, index=self.X.index, columns=self.X.columns)
        selector = SelectKBest(chi2, k=min(self.n_features_to_select, self.X.shape[1]))
        X_new = selector.fit_transform(self.X, self.y_categorical)
        selected_columns = self.X.columns[selector.get_support()]
        self.X = pd.DataFrame(X_new, index=self.X.index, columns=selected_columns)
        joblib.dump(self.X.columns.tolist(), os.path.join(self.output_dir, 'selected_snp_columns.joblib'))
        print(f"Özellik seçimi tamamlandı. Yeni veri boyutu: {self.X.shape}")

    def run(self):
        try:
            self._load_and_prepare_data()
            self._perform_feature_selection()

            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y_categorical, test_size=self.test_size,
                random_state=self.random_state, stratify=self.y_categorical #Veriyi ayırırken, her bir popülasyonun (özellikle az sayıda örneği olanların) hem eğitim hem de test setinde aynı oranda bulunmasını sağlar. Bu, modelin tüm popülasyonlar üzerinde adil bir şekilde eğitilip test edilmesini garanti eder.
            )

            print(f"\n--- Adım 3: Veri Dönüştürücüleri Eğitme ve Veriyi Dönüştürme ---")

            scalers = {}
            pcas = {}
            X_train_pca_list = []
            X_test_pca_list = []

            num_chunks = (X_train.shape[1] + self.snps_per_chunk - 1) // self.snps_per_chunk
            for i in range(num_chunks):
                chunk_num = i + 1
                print(f"  PCA Parçası işleniyor: {chunk_num}/{num_chunks}")

                start_col_idx = i * self.snps_per_chunk
                end_col_idx = (i + 1) * self.snps_per_chunk

                X_train_chunk = X_train.iloc[:, start_col_idx:end_col_idx]
                X_test_chunk = X_test.iloc[:, start_col_idx:end_col_idx]

                scaler = StandardScaler()
                X_train_chunk_scaled = scaler.fit_transform(X_train_chunk)

                n_components = min(self.n_pca_components, X_train_chunk_scaled.shape[1])
                pca = PCA(n_components=n_components, random_state=self.random_state)
                X_train_chunk_pca = pca.fit_transform(X_train_chunk_scaled)

                X_test_chunk_scaled = scaler.transform(X_test_chunk)
                X_test_chunk_pca = pca.transform(X_test_chunk_scaled)

                scalers[chunk_num] = scaler
                pcas[chunk_num] = pca

                pc_columns = [f"PC{j + 1}_Chr{chunk_num}" for j in range(n_components)]
                X_train_pca_list.append(pd.DataFrame(X_train_chunk_pca, index=X_train.index, columns=pc_columns))
                X_test_pca_list.append(pd.DataFrame(X_test_chunk_pca, index=X_test.index, columns=pc_columns))

            X_train_pca_final = pd.concat(X_train_pca_list, axis=1)
            X_test_pca_final = pd.concat(X_test_pca_list, axis=1)

            joblib.dump(scalers, os.path.join(self.output_dir, 'scalers.joblib'))
            joblib.dump(pcas, os.path.join(self.output_dir, 'pcas.joblib'))
            print("\nScaler ve PCA dönüştürücüleri başarıyla kaydedildi.")

            self._train_evaluate_and_save_model(X_train_pca_final, X_test_pca_final, y_train, y_test)

            print("\n--- Tüm işlemler başarıyla tamamlandı! ---")
        except Exception as e:
            print(f"\nİŞLEM SIRASINDA BİR HATA OLUŞTU: {e}")
            raise

    def _train_evaluate_and_save_model(self, X_train, X_test, y_train, y_test):
        print("\n--- Adım 4: Model Eğitimi, Değerlendirmesi ve Kaydedilmesi ---")
        model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state, class_weight='balanced', n_jobs=-1 #balance oneöli
        )
        model.fit(X_train, y_train)
        model_path = os.path.join(self.output_dir, 'random_forest_ethnicity_predictor.joblib')
        joblib.dump(model, model_path)
        joblib.dump(X_train.columns.tolist(), os.path.join(self.output_dir, 'model_columns.joblib'))
        print(f"\nEğitilmiş model '{model_path}' olarak kaydedildi.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Doğruluğu (Accuracy): {accuracy:.4f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SOURCE_FOLDER = os.path.join(PROJECT_ROOT, "source")
    GENOTYPE_FILE = os.path.join(SOURCE_FOLDER, "filtrelenmis_genotip_TUM_KROMOZOM_top7000_snps.csv")
    PANEL_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
    OUTPUT_MODEL_DIR = os.path.join(SOURCE_FOLDER, "machine_learning")
    trainer = RandomForestTrainer(
        genotype_file_path=GENOTYPE_FILE, panel_file_url=PANEL_URL, output_dir=OUTPUT_MODEL_DIR
    )
    trainer.run()