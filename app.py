import os
import gzip
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
import shutil
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

POPULATION_MAP = {
    'ACB': 'African Caribbean in Barbados', 'ASW': 'Americans of African Ancestry in SW USA',
    'BEB': 'Bengali in Bangladesh',
    'CDX': 'Chinese Dai in Xishuangbanna, China',
    'CEU': 'Utah Residents (CEPH) with Northern and Western European Ancestry',
    'CHB': 'Han Chinese in Beijing, China', 'CHS': 'Southern Han Chinese', 'CLM': 'Colombians from Medellin, Colombia',
    'ESN': 'Esan in Nigeria', 'FIN': 'Finnish in Finland', 'GBR': 'British in England and Scotland',
    'GIH': 'Gujarati Indian from Houston, Texas', 'GWD': 'Gambian in Western Divisions in the Gambia',
    'IBS': 'Iberian Population in Spain', 'ITU': 'Indian Telugu from the UK', 'JPT': 'Japanese in Tokyo, Japan',
    'KHV': 'Kinh in Ho Chi Minh City, Vietnam', 'LWK': 'Luhya in Webuye, Kenya', 'MSL': 'Mende in Sierra Leone',
    'MXL': 'Mexican Ancestry from Los Angeles, USA', 'PEL': 'Peruvians from Lima, Peru',
    'PJL': 'Punjabi in Lahore, Pakistan', 'PUR': 'Puerto Ricans from Puerto Rico',
    'STU': 'Sri Lankan Tamil from the UK',
    'TSI': 'Toscani in Italia (Tuscany, Italy)', 'YRI': 'Yoruba in Ibadan, Nigeria',
}

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
SOURCE_DATA_FOLDER = os.path.join(APP_ROOT, 'source')
ML_FOLDER = os.path.join(SOURCE_DATA_FOLDER, 'machine_learning')

MODEL_PATH_APP = os.path.join(ML_FOLDER, 'random_forest_ethnicity_predictor.joblib')
MODEL_COLUMNS_PATH_APP = os.path.join(ML_FOLDER, 'model_columns.joblib')
SELECTED_SNPS_PATH_APP = os.path.join(ML_FOLDER, 'selected_snp_columns.joblib')
SCALERS_PATH_APP = os.path.join(ML_FOLDER, 'scalers.joblib')
PCAS_PATH_APP = os.path.join(ML_FOLDER, 'pcas.joblib')


def genotip_kodunu_hesapla(genotip_str):
    if genotip_str in ('0/1', '1/0'): return 1
    if genotip_str == '1/1': return 2
    if genotip_str == '0/0': return 0
    return -1


class AncestryPipeline:
    def __init__(self, job_id, user_vcf_path):
        self.job_id = job_id
        self.user_vcf_path = user_vcf_path
        self.work_dir = os.path.join(UPLOAD_FOLDER, self.job_id)
        os.makedirs(self.work_dir, exist_ok=True)
        self.sample_id = f'user_{self.job_id[:8]}'
        self.min_snps_threshold = 800

        self.selected_snp_list_raw = joblib.load(SELECTED_SNPS_PATH_APP)
        self.selected_snp_list = [snp.strip("'\" ") for snp in self.selected_snp_list_raw]

        self.snp_search_db = self._prepare_snp_database(self.selected_snp_list)
        self.scalers = joblib.load(SCALERS_PATH_APP)
        self.pcas = joblib.load(PCAS_PATH_APP)
        self.model = joblib.load(MODEL_PATH_APP)
        self.model_columns = joblib.load(MODEL_COLUMNS_PATH_APP)
        self.snps_per_chunk = 7000
        app.logger.info(f"AncestryPipeline nesnesi oluşturuldu ve tüm modeller yüklendi.")

    def _prepare_snp_database(self, snp_list):
        snp_db = defaultdict(set)
        for snp in snp_list:
            parts = snp.split(':')
            if len(parts) == 4:
                snp_db[f"{parts[0].replace('chr', '')}:{parts[1]}"].add(f"{parts[2]}:{parts[3]}")
        return snp_db

    def _process_vcf(self):
        genotypes_dict = {}
        found_snps_count = 0
        opener = gzip.open if self.user_vcf_path.endswith('.gz') else open
        with opener(self.user_vcf_path, 'rt', encoding='utf-8', errors='ignore') as vcf_file:
            for line in vcf_file:
                if line.startswith('#'): continue
                cols = line.strip().split('\t')
                if len(cols) < 10: continue
                chrom, pos, _, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
                search_key = f"{chrom.replace('chr', '')}:{pos}"
                if search_key in self.snp_search_db:
                    for alt_allele in alt.split(','):
                        vcf_value = f"{ref}:{alt_allele}"
                        if vcf_value in self.snp_search_db[search_key]:
                            try:
                                gt_index = cols[8].split(':').index('GT')
                                gt_str = cols[9].split(':')[gt_index]
                                for master_snp in (s for s in self.selected_snp_list if
                                                   search_key in s and vcf_value in s):
                                    genotypes_dict[master_snp] = genotip_kodunu_hesapla(gt_str)
                                    found_snps_count += 1
                                    break
                                break
                            except (ValueError, IndexError):
                                continue
        app.logger.info(f"VCF dosyasından toplam {found_snps_count} adet eşleşen SNP bulundu.")
        return genotypes_dict, found_snps_count


    def _transform_user_data(self, user_genotypes_dict):
        app.logger.info("Kullanıcı verisi dönüştürülüyor...")
        user_series = pd.Series(user_genotypes_dict).reindex(self.selected_snp_list, fill_value=0)#ÖNEMLİ
        user_df = pd.DataFrame(user_series, columns=[self.sample_id]).T
        all_user_pca_dfs = []
        num_chunks = (len(self.selected_snp_list) + self.snps_per_chunk - 1) // self.snps_per_chunk

        for i in range(num_chunks):
            chunk_num = i + 1
            app.logger.info(f"Transforming PCA chunk {chunk_num}/{num_chunks}...")
            scaler = self.scalers[chunk_num]
            pca = self.pcas[chunk_num]
            start_col_idx = i * self.snps_per_chunk
            end_col_idx = (i + 1) * self.snps_per_chunk
            user_chunk = user_df.iloc[:, start_col_idx:end_col_idx]
            if user_chunk.empty: continue
            user_chunk_scaled = scaler.transform(user_chunk)
            user_chunk_pca = pca.transform(user_chunk_scaled)
            pc_columns = [f"PC{j + 1}_Chr{chunk_num}" for j in range(user_chunk_pca.shape[1])]
            all_user_pca_dfs.append(pd.DataFrame(user_chunk_pca, columns=pc_columns, index=[self.sample_id]))

        if not all_user_pca_dfs: raise Exception("PCA dönüşümü başarısız oldu.")

        final_user_pca_df = pd.concat(all_user_pca_dfs, axis=1)
        return final_user_pca_df

    def _make_prediction(self, features_df):
        app.logger.info("Tahmin yapılıyor...")
        features_reordered = features_df.reindex(columns=self.model_columns, fill_value=0)
        probabilities = self.model.predict_proba(features_reordered)[0]
        class_names = self.model.classes_
        full_target_names = [POPULATION_MAP.get(code, code) for code in class_names]
        percentages = [p * 100 for p in probabilities]
        results_data = {"sample_id": self.sample_id, "labels": full_target_names, "percentages": percentages,
                        "display_predictions": []}
        results_df = pd.DataFrame({'category': full_target_names, 'percentage': percentages}).sort_values(
            by='percentage', ascending=False)
        for _, row in results_df.iterrows():
            if row['percentage'] > 0.01:
                results_data["display_predictions"].append(
                    {"category": row['category'], "percentage_str": f"{row['percentage']:.2f}%"})
        dominant_row = results_df.iloc[0]
        results_data["dominant_ancestry"] = dominant_row['category']
        results_data["dominant_percentage_str"] = f"{dominant_row['percentage']:.2f}%"
        return results_data

    def cleanup(self):
        if os.path.exists(self.work_dir): shutil.rmtree(self.work_dir)

    def run(self):
        user_genotypes_dict, snp_match_count = self._process_vcf()
        if snp_match_count < self.min_snps_threshold:
            raise Exception(
                f"Analiz durduruldu: Yetersiz genetik veri. Modelin bildiği {len(self.selected_snp_list)} belirteçten sadece {snp_match_count} tanesi bulundu. Lütfen dosyanızın GRCh37 olduğundan emin olun.")

        features_df = self._transform_user_data(user_genotypes_dict)
        prediction = self._make_prediction(features_df)
        app.logger.info("Analiz tamamlandı.")
        return prediction


app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'vcf_file' not in request.files or not request.files['vcf_file'].filename:
            flash('Please select a file.', 'danger')
            return redirect(request.url)
        file = request.files['vcf_file']
        if file and (file.filename.endswith('.vcf') or file.filename.endswith('.vcf.gz')):
            pipeline = None
            try:
                job_id = str(uuid.uuid4())
                work_dir = os.path.join(UPLOAD_FOLDER, job_id)
                os.makedirs(work_dir, exist_ok=True)
                user_vcf_path = os.path.join(work_dir, file.filename)
                file.save(user_vcf_path)
                pipeline = AncestryPipeline(job_id=job_id, user_vcf_path=user_vcf_path)
                prediction_results = pipeline.run()
                return render_template('results.html', results=prediction_results)
            except Exception as e:
                app.logger.error(f"An error occurred during processing: {e}", exc_info=True)
                flash(f'{e}', "danger")
                return redirect(url_for('index'))
            finally:
                if pipeline:
                    pipeline.cleanup()
        else:
            flash('Invalid file type. Please upload a .vcf or .vcf.gz file.', 'warning')
            return redirect(request.url)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)