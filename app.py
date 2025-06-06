import gzip
from collections import defaultdict
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib
import warnings
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash

# --- POPÜLASYON KODU ÇEVİRİ SÖZLÜĞÜ ---
# 1000 Genomes Projesi'ndeki popülasyon kodları ve tam adları
POPULATION_MAP = {
    'pop_ACB': 'African Caribbean in Barbados',
    'pop_ASW': 'Americans of African Ancestry in SW USA',
    'pop_BEB': 'Bengali from Bangladesh',
    'pop_CDX': 'Chinese Dai in Xishuangbanna, China',
    'pop_CEU': 'Utah Residents (CEPH) with Northern and Western European Ancestry',
    'pop_CHB': 'Han Chinese in Beijing, China',
    'pop_CHS': 'Southern Han Chinese',
    'pop_CLM': 'Colombians from Medellin, Colombia',
    'pop_ESN': 'Esan in Nigeria',
    'pop_FIN': 'Finnish in Finland',
    'pop_GBR': 'British in England and Scotland',
    'pop_GIH': 'Gujarati Indian from Houston, Texas',
    'pop_GWD': 'Gambian in Western Divisions in the Gambia',
    'pop_IBS': 'Iberian Population in Spain',
    'pop_ITU': 'Indian Telugu from the UK',
    'pop_JPT': 'Japanese in Tokyo, Japan',
    'pop_KHV': 'Kinh in Ho Chi Minh City, Vietnam',
    'pop_LWK': 'Luhya in Webuye, Kenya',
    'pop_MSL': 'Mende in Sierra Leone',
    'pop_MXL': 'Mexican Ancestry from Los Angeles, USA',
    'pop_PEL': 'Peruvians from Lima, Peru',
    'pop_PJL': 'Punjabi from Lahore, Pakistan',
    'pop_PUR': 'Puerto Ricans from Puerto Rico',
    'pop_STU': 'Sri Lankan Tamil from the UK',
    'pop_TSI': 'Toscani in Italia',
    'pop_YRI': 'Yoruba in Ibadan, Nigeria',
}


# --- Analiz Fonksiyonları (Öncekiyle aynı) ---

def snp_veritabanini_ana_csvden_olustur(ana_csv_dosya_yolu):
    master_liste = []
    aranacak_db = defaultdict(set)
    app.logger.info(f"Reading reference SNP list from main CSV: '{ana_csv_dosya_yolu}'")
    try:
        with open(ana_csv_dosya_yolu, 'r', encoding='utf-8') as f:
            baslik_satiri = f.readline()
            snp_sutunlari = baslik_satiri.strip().split(',')[1:]
            master_liste = [snp.strip().strip("'\"") for snp in snp_sutunlari]
            for snp in master_liste:
                parcalar = snp.split(':')
                if len(parcalar) == 4:
                    kromozom, pozisyon, ref, alt = parcalar
                    arama_anahtari = f"{kromozom.replace('chr', '')}:{pozisyon}"
                    deger = f"{ref}:{alt}"
                    aranacak_db[arama_anahtari].add(deger)
        app.logger.info(f"Database created from main CSV header. {len(master_liste)} unique SNPs to search for.")
        return master_liste, aranacak_db
    except FileNotFoundError:
        app.logger.error(f"ERROR: Main CSV file not found: {ana_csv_dosya_yolu}")
        return None, None
    except Exception as e:
        app.logger.error(f"An error occurred while reading the main CSV header: {e}")
        return None, None


def genotip_kodunu_hesapla(genotip_str):
    if genotip_str in ('0/1', '1/0'): return 1
    if genotip_str == '1/1': return 2
    if genotip_str == '0/0': return 0
    return -1


def vcf_dosyasindaki_genotipleri_bul(aranacak_snpler, vcf_yolu, master_snp_formatlari_global):
    bulunan_genotipler = {}
    app.logger.info(f"Scanning VCF file: '{vcf_yolu}'")
    try:
        vcf_opener = gzip.open if vcf_yolu.endswith('.gz') else open
        with vcf_opener(vcf_yolu, 'rt', encoding='utf-8') as vcf_dosyasi:
            for satir in vcf_dosyasi:
                if satir.startswith('#'): continue
                sutunlar = satir.strip().split('\t')
                if len(sutunlar) < 10: continue
                kromozom, pozisyon, _, ref, alt = sutunlar[0], sutunlar[1], sutunlar[2], sutunlar[3], sutunlar[4]
                alt_aleller = alt.split(',')
                arama_anahtari = f"{kromozom.replace('chr', '')}:{pozisyon}"
                if arama_anahtari in aranacak_snpler:
                    for tek_alt_alel in alt_aleller:
                        vcf_deger = f"{ref}:{tek_alt_alel}"
                        if vcf_deger in aranacak_snpler[arama_anahtari]:
                            try:
                                format_alani = sutunlar[8].split(':')
                                ornek_alani = sutunlar[9].split(':')
                                gt_index = format_alani.index('GT')
                                genotip_str = ornek_alani[gt_index]
                                genotip_kodu = genotip_kodunu_hesapla(genotip_str)
                                tam_snp_str = f"{kromozom.replace('chr', '')}:{pozisyon}:{ref}:{tek_alt_alel}"
                                master_format_snp_str_chr = f"chr{kromozom.replace('chr', '')}:{pozisyon}:{ref}:{tek_alt_alel}"
                                master_format_snp_str_no_chr = f"{kromozom.replace('chr', '')}:{pozisyon}:{ref}:{tek_alt_alel}"
                                if master_format_snp_str_chr in master_snp_formatlari_global:
                                    tam_snp_str = master_format_snp_str_chr
                                elif master_format_snp_str_no_chr in master_snp_formatlari_global:
                                    tam_snp_str = master_format_snp_str_no_chr
                                bulunan_genotipler[tam_snp_str] = genotip_kodu
                                break
                            except (ValueError, IndexError):
                                continue
    except FileNotFoundError:
        app.logger.error(f"ERROR: VCF file not found: {vcf_yolu}")
        return None
    except Exception as e:
        app.logger.error(f"An error occurred while processing VCF file: {e}", exc_info=True)
        return None
    app.logger.info(f"VCF scan complete. {len(bulunan_genotipler)} matching SNP genotypes found.")
    return bulunan_genotipler


def sonuclari_ozel_formatta_yaz(master_liste, bulunan_genotipler, cikti_dosyasi):
    app.logger.info(f"Writing results to custom format file: '{cikti_dosyasi}'")
    genotip_kod_listesi = [str(bulunan_genotipler.get(snp, 0)) for snp in master_liste]
    ikinci_satir = ",".join(genotip_kod_listesi)
    try:
        os.makedirs(os.path.dirname(cikti_dosyasi), exist_ok=True)
        with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
            f.write(ikinci_satir)
        app.logger.info(f"File created successfully with {len(genotip_kod_listesi)} genotype codes: {cikti_dosyasi}")
    except Exception as e:
        app.logger.error(f"An error occurred while writing the file: {e}")


def run_pca_on_input_data(genotype_input, data_source_name, n_components, output_filepath_base,
                          imputation_strategy='mean'):
    if isinstance(genotype_input, str):
        try:
            genotype_df = pd.read_csv(genotype_input, index_col=0)
        except Exception as e:
            app.logger.error(f"ERROR ({data_source_name}): Error loading genotype file: {e}")
            return None
    else:
        genotype_df = genotype_input.copy()

    X = genotype_df.values
    sample_ids = genotype_df.index
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    X_imputed = imputer.fit_transform(X)
    if np.any(X_imputed == -1):
        X_imputed[X_imputed == -1] = np.nan
        imputer_for_minus_one = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
        X_imputed = imputer_for_minus_one.fit_transform(X_imputed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    actual_n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=actual_n_components)
    principal_components = pca.fit_transform(X_scaled)
    pc_columns = [f'PC{i + 1}' for i in range(actual_n_components)]
    return pd.DataFrame(data=principal_components, columns=pc_columns, index=sample_ids)


def extract_and_save_sample_features(input_path, output_path, sample_id, id_column):
    try:
        full_pca_df = pd.read_csv(input_path)
        sample_row_df = full_pca_df[full_pca_df[id_column] == sample_id]
        if sample_row_df.empty: return False
        features_only_df = sample_row_df.drop(columns=[id_column])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features_only_df.to_csv(output_path, index=False, header=True)
        return True
    except Exception as e:
        app.logger.error(f"Error extracting features for {sample_id}: {e}")
        return False


def load_features_and_model(features_path, model_path_val):
    try:
        features_df = pd.read_csv(features_path)
        if features_df.empty: return None, None
        ml_model = joblib.load(model_path_val)
        return features_df, ml_model
    except Exception as e:
        app.logger.error(f"Error loading features or model: {e}")
        return None, None


def get_target_column_names(y_path):
    if y_path and os.path.exists(y_path):
        try:
            y_df = pd.read_csv(y_path)
            # Eğer y_processed.csv'de ilk sütun 'SampleID' gibi bir şeyse ve model onsuz eğitildiyse,
            # bu sütunu atmak için y_df.columns[1:].tolist() kullanabilirsiniz.
            # Şimdilik tüm sütunları aldığını varsayıyoruz.
            return y_df.columns.tolist()
        except Exception as e:
            app.logger.warning(f"WARNING: Could not read column names from {os.path.basename(y_path)}: {e}.")
    return None


def make_prediction_and_get_results(model, features, target_sample_id, target_names=None):
    prediction = model.predict(features)
    raw_predictions = prediction[0]
    positive_predictions = [max(0, val) for val in raw_predictions]
    sum_of_positives = sum(positive_predictions)
    normalized_percentages = []
    if sum_of_positives > 0:
        normalized_percentages = [(val / sum_of_positives) * 100 for val in positive_predictions]
    else:
        normalized_percentages = [0.0] * len(raw_predictions)

    results_data = {"sample_id": target_sample_id, "labels": [], "percentages": [], "display_predictions": [],
                    "dominant_ancestry": "N/A", "dominant_percentage_str": "N/A", "error_message": None}

    # target_names burada kısa kodları değil, tam adları içerecek
    if target_names and len(target_names) == len(normalized_percentages):
        results_data["labels"] = target_names
        results_data["percentages"] = normalized_percentages
        results_df = pd.DataFrame({'category': target_names, 'percentage': normalized_percentages}).sort_values(
            by='percentage', ascending=False)
        for index, row in results_df.iterrows():
            results_data["display_predictions"].append(
                {"category": row['category'], "percentage_str": f"{row['percentage']:.2f}%"})
        if not results_df.empty and results_df.iloc[0]['percentage'] > 0:
            dominant_row = results_df.iloc[0]
            results_data["dominant_ancestry"] = dominant_row['category']
            results_data["dominant_percentage_str"] = f"{dominant_row['percentage']:.2f}%"
        else:
            results_data["dominant_ancestry"] = "No dominant ancestry could be determined."
    else:
        results_data["error_message"] = "Target names do not match prediction count or are unavailable."
        results_data["labels"] = [f"Population {i + 1}" for i in range(len(normalized_percentages))]
        results_data["percentages"] = normalized_percentages
        for i, val in enumerate(normalized_percentages):
            results_data["display_predictions"].append(
                {"category": f"Population {i + 1}", "percentage_str": f"{val:.2f}%"})

    return results_data


# --- Flask Uygulaması ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Uygulama Ayarları ve Dosya Yolları ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
SOURCE_DATA_FOLDER = os.path.join(APP_ROOT, 'source')
APP_RESULTS_FOLDER = os.path.join(APP_ROOT, 'results_app')
ANA_CSV_DOSYASI_REFERANS = os.path.join(SOURCE_DATA_FOLDER, 'filtrelenmis_genotip_TUM_KROMOZOM_top7000_snps.csv')
MODEL_PATH_APP = os.path.join(SOURCE_DATA_FOLDER, 'machine_learning', 'optimized_ethnicity_predictor.joblib')
Y_COLUMN_NAMES_PATH_APP = os.path.join(SOURCE_DATA_FOLDER, 'y_processed.csv')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(APP_RESULTS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
app.logger.setLevel(logging.INFO)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'vcf_file' not in request.files or request.files['vcf_file'].filename == '':
            flash('Please select a file.', 'danger')
            return redirect(request.url)
        file = request.files['vcf_file']
        if file and (file.filename.endswith('.vcf') or file.filename.endswith('.vcf.gz')):
            session_id = str(uuid.uuid4())
            session_work_dir = os.path.join(UPLOAD_FOLDER, session_id)
            os.makedirs(session_work_dir, exist_ok=True)
            user_vcf_path = os.path.join(session_work_dir, file.filename)
            file.save(user_vcf_path)
            app.logger.info(f"File '{file.filename}' uploaded. Processing ID: {session_id}")

            try:
                # Dinamik Dosya Yolları
                CIKTI_GENOTIP_TXT_DOSYASI = os.path.join(session_work_dir, 'genotip_sonuclari.txt')
                YENI_BIREY_KIMLIGI = f'user_{session_id[:8]}'
                BIRLESTIRILMIS_SNP_VERILERI_CSV = os.path.join(session_work_dir, 'birlestirilmis_snp_verileri.csv')
                SESSION_RESULTS_DIR = os.path.join(APP_RESULTS_FOLDER, session_id)
                os.makedirs(SESSION_RESULTS_DIR, exist_ok=True)
                FINAL_COMBINED_PCA_FILEPATH = os.path.join(SESSION_RESULTS_DIR,
                                                           f"pca_combined_{YENI_BIREY_KIMLIGI}.csv")
                EXTRACTED_FEATURES_FILEPATH = os.path.join(SESSION_RESULTS_DIR, f"features_{YENI_BIREY_KIMLIGI}.csv")
                PCA_SAMPLE_ID_COLUMN_NAME = "SampleID"

                # Adım 1: VCF İşleme
                master_liste_snps, aranacak_db_snps = snp_veritabanini_ana_csvden_olustur(ANA_CSV_DOSYASI_REFERANS)
                if not master_liste_snps: raise Exception("Could not create the main reference SNP list.")
                current_run_master_snp_formats = set(master_liste_snps)
                vcf_sonuclari = vcf_dosyasindaki_genotipleri_bul(aranacak_db_snps, user_vcf_path,
                                                                 current_run_master_snp_formats)
                if vcf_sonuclari is None: raise Exception("Could not read genotypes from VCF file.")
                sonuclari_ozel_formatta_yaz(master_liste_snps, vcf_sonuclari, CIKTI_GENOTIP_TXT_DOSYASI)

                # Adım 2: Veri Birleştirme
                with open(CIKTI_GENOTIP_TXT_DOSYASI, 'r') as f:
                    genotip_verileri_str = f.read().strip()
                yeni_satir_verisi = f"{YENI_BIREY_KIMLIGI},{genotip_verileri_str}\n"
                shutil.copy(ANA_CSV_DOSYASI_REFERANS, BIRLESTIRILMIS_SNP_VERILERI_CSV)
                with open(BIRLESTIRILMIS_SNP_VERILERI_CSV, 'a', encoding='utf-8') as f:
                    f.write(yeni_satir_verisi)

                # Adım 3: PCA Analizi
                master_genotype_df = pd.read_csv(BIRLESTIRILMIS_SNP_VERILERI_CSV, index_col=0)
                all_chromosome_pca_dfs_renamed = []
                total_snps = master_genotype_df.shape[1]
                snps_per_chunk = 7000
                n_pca_components = 20
                num_chunks = (total_snps + snps_per_chunk - 1) // snps_per_chunk
                for i in range(num_chunks):
                    start_col = i * snps_per_chunk
                    end_col = start_col + snps_per_chunk
                    chunk_df = master_genotype_df.iloc[:, start_col:end_col]
                    if chunk_df.empty: continue
                    pca_scores = run_pca_on_input_data(chunk_df, f"Chunk{i + 1}", n_pca_components, None)
                    if pca_scores is not None:
                        renamed_cols = {col: f"{col}_Chunk{i + 1}" for col in pca_scores.columns}
                        all_chromosome_pca_dfs_renamed.append(pca_scores.rename(columns=renamed_cols))

                final_combined_pca_scores_df = pd.concat(all_chromosome_pca_dfs_renamed, axis=1)
                final_output_df = final_combined_pca_scores_df.reset_index().rename(
                    columns={'index': PCA_SAMPLE_ID_COLUMN_NAME})
                final_output_df.to_csv(FINAL_COMBINED_PCA_FILEPATH, index=False)

                # Adım 4: Özellik Çıkarma
                if not extract_and_save_sample_features(FINAL_COMBINED_PCA_FILEPATH, EXTRACTED_FEATURES_FILEPATH,
                                                        YENI_BIREY_KIMLIGI, PCA_SAMPLE_ID_COLUMN_NAME):
                    raise Exception("Could not extract PCA features for the user sample.")

                # Adım 5: Tahmin
                features, model = load_features_and_model(EXTRACTED_FEATURES_FILEPATH, MODEL_PATH_APP)
                if features is None or model is None:
                    raise Exception("Could not load features or model for prediction.")

                # --- YENİ EKLENEN KISIM: KISA KODLARI TAM İSİMLERE ÇEVİRME ---
                target_name_codes = get_target_column_names(Y_COLUMN_NAMES_PATH_APP)
                target_names_full = None
                if target_name_codes:
                    # .get(code, code) kullanılırsa, sözlükte bulunamayan kod kendisi olarak kalır.
                    target_names_full = [POPULATION_MAP.get(code, code) for code in target_name_codes]
                # --- ÇEVİRME İŞLEMİ SONU ---

                warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

                # Tahmin fonksiyonuna tam isimleri (target_names_full) gönder
                prediction_results = make_prediction_and_get_results(model, features, YENI_BIREY_KIMLIGI,
                                                                     target_names_full)

                return render_template('results.html', results=prediction_results)

            except Exception as e:
                app.logger.error(f"An error occurred during processing (ID: {session_id}): {e}", exc_info=True)
                flash(f'An unexpected error occurred during processing: {e}', "danger")
                return redirect(url_for('index'))
            finally:
                # Geçici dosyaları temizle
                try:
                    if os.path.exists(session_work_dir): shutil.rmtree(session_work_dir)
                    if 'SESSION_RESULTS_DIR' in locals() and os.path.exists(SESSION_RESULTS_DIR): shutil.rmtree(
                        SESSION_RESULTS_DIR)
                    app.logger.info(f"Temporary files cleaned up for ID: {session_id}")
                except Exception as e_clean:
                    app.logger.error(f"Error during temporary file cleanup: {e_clean}")
        else:
            flash('Invalid file type. Please upload a .vcf or .vcf.gz file.', 'warning')
            return redirect(request.url)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)