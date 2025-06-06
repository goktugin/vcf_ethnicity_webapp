import os
import gzip
import shutil
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


# =============================================================================
# --- FONKSİYONLAR (TÜM SCRIPTLERDEN TOPLANDI) ---
# =============================================================================

# --- Script 1'den Gelen Fonksiyonlar: VCF İşleme ---

def snp_veritabanini_ana_csvden_olustur(ana_csv_dosya_yolu):
    """
    Ana CSV dosyasının başlık satırını okuyarak SNP listesini ve arama veritabanını oluşturur.
    """
    master_liste = []
    aranacak_db = defaultdict(set)
    print(f"Referans SNP listesi ana CSV dosyasından okunuyor: '{ana_csv_dosya_yolu}'")
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
        print(f"Veritabanı ana CSV başlığından oluşturuldu. Aranacak {len(master_liste)} benzersiz SNP mevcut.")
        return master_liste, aranacak_db
    except FileNotFoundError:
        print(f"HATA: Ana CSV dosyası bulunamadı: {ana_csv_dosya_yolu}")
        return None, None
    except Exception as e:
        print(f"Ana CSV başlığı okunurken bir hata oluştu: {e}")
        return None, None


def genotip_kodunu_hesapla(genotip_str):
    """VCF'ten gelen genotip bilgisini sayısal koda çevirir."""
    if genotip_str in ('0/1', '1/0'): return 1
    if genotip_str == '1/1': return 2
    if genotip_str == '0/0': return 0
    return -1


def vcf_dosyasindaki_genotipleri_bul(aranacak_snpler, vcf_yolu):
    """
    VCF dosyasını tarar ve bulunan SNP'lerin genotip kodlarını bir sözlükte toplar.
    """
    bulunan_genotipler = {}
    print(f"\nVCF dosyası taranıyor: '{vcf_yolu}'")
    try:
        with gzip.open(vcf_yolu, 'rt', encoding='utf-8') as vcf_dosyasi:
            for satir in vcf_dosyasi:
                if satir.startswith('#'): continue
                sutunlar = satir.strip().split('\t')
                kromozom, pozisyon, ref, alt = sutunlar[0], sutunlar[1], sutunlar[3], sutunlar[4]
                arama_anahtari = f"{kromozom.replace('chr', '')}:{pozisyon}"
                if arama_anahtari in aranacak_snpler:
                    vcf_deger = f"{ref}:{alt}"
                    if vcf_deger in aranacak_snpler[arama_anahtari]:
                        try:
                            format_alani = sutunlar[8].split(':')
                            ornek_alani = sutunlar[9].split(':')
                            gt_index = format_alani.index('GT')
                            genotip_str = ornek_alani[gt_index]
                            genotip_kodu = genotip_kodunu_hesapla(genotip_str)
                            tam_snp_str = f"{kromozom}:{pozisyon}:{ref}:{alt}"
                            bulunan_genotipler[tam_snp_str] = genotip_kodu
                        except (ValueError, IndexError):
                            continue
    except FileNotFoundError:
        print(f"HATA: VCF dosyası bulunamadı: {vcf_yolu}")
        return None
    print(f"VCF taraması tamamlandı. {len(bulunan_genotipler)} eşleşen SNP için genotip bilgisi bulundu.")
    return bulunan_genotipler


# --- Script 3'ten Gelen Fonksiyon: PCA Analizi ---

def run_pca_on_input_data(genotype_input, data_source_name, n_components, imputation_strategy='mean'):
    """
    Verilen bir genotip DataFrame'i üzerinde PCA gerçekleştirir.
    """
    genotype_df = genotype_input.copy()
    X = genotype_df.values
    sample_ids = genotype_df.index

    if X.shape[1] == 0:
        print(f"HATA ({data_source_name}): Genotip verisinde hiç SNP (özellik) bulunmuyor.")
        return None

    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    actual_n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    if actual_n_components < n_components:
        print(
            f"  UYARI ({data_source_name}): İstenen bileşen sayısı ({n_components}), veri boyutlarına göre {actual_n_components} olarak ayarlandı.")
    if actual_n_components == 0:
        print(f"HATA ({data_source_name}): Hesaplanacak geçerli bileşen sayısı 0.")
        return None

    pca = PCA(n_components=actual_n_components)
    principal_components = pca.fit_transform(X_scaled)
    pc_columns = [f'PC{i + 1}' for i in range(actual_n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=sample_ids)
    return pca_df


# --- Script 5'ten Gelen Fonksiyonlar: Tahminleme ---

def load_model(model_path_val):
    """Eğitilmiş modeli yükler."""
    print(f"\nEğitilmiş model yükleniyor: {model_path_val}")
    try:
        ml_model = joblib.load(model_path_val)
        print("Eğitilmiş model başarıyla yüklendi.")
        return ml_model
    except FileNotFoundError:
        print(f"HATA: Model dosyası bulunamadı: {model_path_val}")
        return None
    except Exception as e:
        print(f"HATA: Model yüklenirken bir hata oluştu: {e}")
        return None


def get_target_column_names(y_path):
    """y_processed.csv'den hedef sütun adlarını okur."""
    if y_path and os.path.exists(y_path):
        try:
            y_df = pd.read_csv(y_path)
            # Sample ID sütununu atla
            if y_df.columns[0].lower().strip() in ['sample', 'sampleid', 'id', 'Unnamed: 0'.lower()]:
                return y_df.columns[1:].tolist()
            return y_df.columns.tolist()
        except Exception as e:
            print(f"UYARI: {os.path.basename(y_path)} dosyasından sütun adları okunurken hata: {e}.")
    return None


def make_prediction_and_display(model, features, target_names, sample_id):
    """Tahmin yapar ve sonucu %100'e normalleştirerek gösterir."""
    print("\nTahmin yapılıyor...")
    try:
        # Modelin 2D array beklemesine karşılık reshape
        features_reshaped = features.values.reshape(1, -1)
        prediction = model.predict(features_reshaped)
        print("Tahmin başarılı.")
    except Exception as e:
        print(f"HATA: Tahmin sırasında bir hata oluştu: {e}")
        return

    print("\n" + "=" * 55)
    print(f"--- ÖRNEK ID: '{sample_id}' İÇİN TAHMİN SONUÇLARI ---")
    print("=" * 55)

    raw_predictions = prediction[0]
    positive_predictions = [max(0, val) for val in raw_predictions]
    sum_of_positives = sum(positive_predictions)

    normalized_percentages = []
    if sum_of_positives > 0:
        normalized_percentages = [(val / sum_of_positives) * 100 for val in positive_predictions]
    else:
        print("UYARI: Tüm tahminler 0 veya daha düşük olduğu için oranlar %0 olarak gösteriliyor.")
        normalized_percentages = [0.0] * len(raw_predictions)

    if target_names and len(target_names) == len(normalized_percentages):
        results_df_display = pd.DataFrame({
            'Soy Oranı Kategorisi': target_names,
            'Tahmini Yüzde': [f"{p:.2f}%" for p in normalized_percentages]
        }).sort_values(by='Tahmini Yüzde', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))

        print(results_df_display.to_string(index=False))

        print("-------------------------------------------------------")
        print(f"TOPLAM YÜZDE: {sum(normalized_percentages):.2f}%")

        if sum_of_positives > 0:
            dominant_ancestry_category = results_df_display.iloc[0]['Soy Oranı Kategorisi']
            max_percentage_str = results_df_display.iloc[0]['Tahmini Yüzde']
            print(f"\nBASKIN TAHMİNİ SOY: {dominant_ancestry_category} ({max_percentage_str})")
    else:
        print("Normalleştirilmiş tahmin edilen yüzdelik değerler:")
        for i, p in enumerate(normalized_percentages):
            print(f"  Kategori {i + 1}: {p:.2f}%")
        if target_names:
            print(
                f"UYARI: y_processed.csv'deki sütun sayısı ({len(target_names)}) ile tahmin sayısı ({len(raw_predictions)}) eşleşmiyor.")
    print("=" * 55)


# =============================================================================
# --- ANA YÜRÜTME BLOGU ---
# =============================================================================
if __name__ == "__main__":

    # --- ANA AYARLAR: LÜTFEN BU YOLLARI KENDİ SİSTEMİNİZE GÖRE GÜNCELLEYİN ---

    # 1. Referans 1000 Genom verisini içeren ana CSV dosyası (SNP başlıkları için kullanılır)
    ana_csv_dosyasi_referans = r'C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\anova_and_matrix_for_each_chr\filtrelenmis_genotip_TUM_KROMOZOM_top7000_snps.csv'

    # 2. Genotipleri okunacak SİZİN VCF dosyanız (sıkıştırılmış .gz formatında olmalı)
    kullanici_vcf_dosyasi = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz"

    # 3. Eğitilmiş makine öğrenmesi modelinizin yolu
    model_path = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\source\machine_learning\optimized_ethnicity_predictor.joblib"

    # 4. Tahmin edilecek etnik köken kategorilerinin adlarını içeren CSV dosyası (isteğe bağlı)
    y_column_names_path = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\machine_learning\y_processed.csv"

    # 5. Sizin örneğinize verilecek kimlik adı
    kullanici_kimligi = 'KULLANICI_ORNEGI'

    # 6. PCA analizi için parametreler
    snps_per_chromosome = 7000
    n_pca_components_per_chr = 20

    print("===== ETNİK KÖKEN TAHMİN PİPELINE'I BAŞLATILDI =====")

    # --- ADIM 1: VCF Dosyasından Genotipleri Çıkarma ---
    print("\n--- ADIM 1: VCF Dosyasından Genotipleri Çıkarma ---")
    master_liste, aranacak_db = snp_veritabanini_ana_csvden_olustur(ana_csv_dosyasi_referans)
    if not master_liste:
        exit()

    vcf_sonuclari = vcf_dosyasindaki_genotipleri_bul(aranacak_db, kullanici_vcf_dosyasi)
    if vcf_sonuclari is None:
        exit()

    kullanici_genotip_listesi = [str(vcf_sonuclari.get(snp, 0)) for snp in master_liste]
    print(f"Kullanıcıya ait {len(kullanici_genotip_listesi)} genotip kodu referans sırasına göre oluşturuldu.")

    # --- ADIM 2: Referans Veriyle Birleştirme (Hafızada) ---
    print("\n--- ADIM 2: Referans Veriyle Birleştirme (Hafızada) ---")
    try:
        master_genotype_df = pd.read_csv(ana_csv_dosyasi_referans, index_col=0)
        print(f"Referans genotip matrisi yüklendi. Boyut: {master_genotype_df.shape}")

        # Yeni kullanıcı satırını DataFrame'e ekle
        kullanici_satiri = pd.DataFrame([kullanici_genotip_listesi], columns=master_genotype_df.columns,
                                        index=[kullanici_kimligi])
        birlesik_df = pd.concat([master_genotype_df, kullanici_satiri])

        print(f"Kullanıcı verisi eklendi. Yeni birleşik matris boyutu: {birlesik_df.shape}")

    except Exception as e:
        print(f"HATA: Referans veri yüklenirken veya birleştirilirken hata oluştu: {e}")
        exit()

    # --- ADIM 3: PCA Analizi ---
    print("\n--- ADIM 3: Birleşik Veri Üzerinde PCA Analizi ---")
    all_chromosome_pca_dfs = []
    current_column_start_index = 0

    for chr_num in range(1, 23):
        print(f"  -> Kromozom {chr_num} işleniyor...")
        start_col = current_column_start_index
        end_col = current_column_start_index + snps_per_chromosome
        if start_col >= birlesik_df.shape[1]: break

        chr_subset_df = birlesik_df.iloc[:, start_col:end_col]

        pca_scores_df_for_chr = run_pca_on_input_data(
            genotype_input=chr_subset_df,
            data_source_name=f"Chr{chr_num}",
            n_components=n_pca_components_per_chr,
        )
        if pca_scores_df_for_chr is not None:
            renamed_cols = {col: f"{col}_Chr{chr_num}" for col in pca_scores_df_for_chr.columns}
            all_chromosome_pca_dfs.append(pca_scores_df_for_chr.rename(columns=renamed_cols))

        current_column_start_index = end_col

    if not all_chromosome_pca_dfs:
        print("HATA: Hiçbir kromozom için PCA sonucu üretilemedi.")
        exit()

    final_combined_pca_scores_df = pd.concat(all_chromosome_pca_dfs, axis=1)
    print(
        f"Tüm kromozomların PCA skorları birleştirildi. Nihai özellik matrisi boyutu: {final_combined_pca_scores_df.shape}")

    # --- ADIM 4: Kullanıcıya Ait PCA Özelliklerini Çıkarma ---
    print("\n--- ADIM 4: Kullanıcıya Ait PCA Özelliklerini Ayıklama ---")
    if kullanici_kimligi not in final_combined_pca_scores_df.index:
        print(f"HATA: Kullanıcı kimliği '{kullanici_kimligi}' PCA sonuçlarında bulunamadı.")
        exit()

    kullanici_pca_ozellikleri = final_combined_pca_scores_df.loc[kullanici_kimligi]
    print(f"'{kullanici_kimligi}' için {len(kullanici_pca_ozellikleri)} adet PCA özelliği başarıyla ayıklandı.")

    # --- ADIM 5: Etnik Köken Tahmini ---
    print("\n--- ADIM 5: Etnik Köken Tahmini ---")
    model = load_model(model_path)
    if model:
        target_names = get_target_column_names(y_column_names_path)
        make_prediction_and_display(model, kullanici_pca_ozellikleri, target_names, kullanici_kimligi)

    print("\n===== İŞLEM TAMAMLANDI =====")