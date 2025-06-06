# Gerekli kütüphaneleri import edelim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Sadece 3D çizim için gerekli, n_components > 3 ise kullanılmaz
import os  # Dosya ve dizin işlemleri için
import urllib.request  # URL'den dosya indirmek için
import re

# Panel dosyasının indirileceği URL
PANEL_FILE_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"


def download_file_if_not_exists(file_path, url):
    """Belirtilen yolda dosya yoksa verilen URL'den indirir."""
    if not os.path.exists(file_path):
        print(f"Dosya ({file_path}) bulunamadı. URL'den indiriliyor: {url}")
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Dizin oluşturuldu: {directory}")

            urllib.request.urlretrieve(url, file_path)
            print(f"Dosya başarıyla '{file_path}' konumuna indirildi.")
            return True
        except Exception as e:
            print(f"HATA: Dosya URL'den indirilemedi ({url}): {e}")
            print(
                "Lütfen internet bağlantınızı ve URL'yi kontrol edin veya dosyayı manuel olarak indirip doğru yolu belirtin.")
            return False
    else:
        print(f"Dosya yerel olarak bulundu: {file_path}")
        return True


def perform_dynamic_pca_and_plot(genotype_csv_path, panel_file_local_path, n_components, output_dir,
                                 output_plot_filename_prefix="pca_plot", output_csv_filename_prefix="pca_results"):
    """
    Verilen genotip matrisi ve popülasyon bilgileri ile dinamik boyutta PCA yapar ve (eğer <=3 ise) grafiğini çizer.
    Panel dosyasını gerekirse URL'den indirir.

    Args:
        genotype_csv_path (str): Genotip matrisini içeren csv dosyasının yolu.
        panel_file_local_path (str): 1000 Genomes panel dosyasının yerel yolu (buraya indirilecek veya buradan okunacak).
        n_components (int): PCA için temel bileşen sayısı.
        output_dir (str): Oluşturulacak PCA sonuçlarının (CSV ve grafik) kaydedileceği dizin yolu.
        output_plot_filename_prefix (str): Oluşturulacak PCA grafiği dosyasının adı için ön ek.
        output_csv_filename_prefix (str): Oluşturulacak PCA sonuçları csv dosyasının adı için ön ek.
    """
    # Adım 1: Panel dosyasını indir (eğer yerelde yoksa)
    if not download_file_if_not_exists(panel_file_local_path, PANEL_FILE_URL):
        return  # Panel dosyası indirilemezse fonksiyondan çık

    print(f"\nGenotip matrisi yükleniyor: {genotype_csv_path}")
    try:
        genotype_df = pd.read_csv(genotype_csv_path, index_col=0)
    except FileNotFoundError:
        print(f"HATA: Genotip csv dosyası bulunamadı: {genotype_csv_path}")
        return
    except Exception as e:
        print(f"csv dosyası okunurken bir hata oluştu: {e}")
        return
    print(f"Genotip matrisi {genotype_df.shape[0]} örnek ve {genotype_df.shape[1]} SNP ile yüklendi.")

    print(f"\nPopülasyon bilgileri yükleniyor: {panel_file_local_path}")
    try:
        panel_df = pd.read_csv(panel_file_local_path, sep='\t')
    except FileNotFoundError:
        print(f"HATA: Panel dosyası ({panel_file_local_path}) bulunamadı ve indirilemedi.")
        return
    except Exception as e:
        print(f"Panel dosyası okunurken bir hata oluştu: {e}")
        return

    sample_to_super_pop = pd.Series(panel_df.super_pop.values, index=panel_df['sample']).to_dict()
    genotype_df['super_pop'] = genotype_df.index.map(sample_to_super_pop)
    original_sample_count = len(genotype_df)
    genotype_df.dropna(subset=['super_pop'], inplace=True)
    if len(genotype_df) < original_sample_count:
        print(
            f"UYARI: {original_sample_count - len(genotype_df)} örnek için panel dosyasında popülasyon bilgisi bulunamadı ve analizden çıkarıldı.")
    if genotype_df.empty or 'super_pop' not in genotype_df.columns or len(genotype_df) == 0:
        print("HATA: Popülasyon bilgileri genotip verisiyle eşleştirilemedi veya filtrelenmiş veri kalmadı.")
        return
    print(f"{len(genotype_df)} örnek için popülasyon bilgileri başarıyla eşleştirildi.")

    snp_columns = [col for col in genotype_df.columns if col != 'super_pop']
    X = genotype_df[snp_columns].copy()
    y_labels = genotype_df['super_pop'].copy()

    # n_components değerinin veri boyutunu aşmamasını sağlama
    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        print(
            f"UYARI: İstediğiniz bileşen sayısı ({n_components}) veri boyutundan ({max_components}) büyük. Bileşen sayısı {max_components} olarak ayarlanıyor.")
        n_components = max_components

    if n_components < 1:
        print("HATA: Bileşen sayısı 1'den küçük olamaz.")
        return

    print("\nEksik genotip değerleri (NaN) ortalama ile dolduruluyor...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)
    print("Eksik değer doldurma tamamlandı.")

    print("\nSNP verisi standartlaştırılıyor (ortalama=0, std=1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print("Standartlaştırma tamamlandı.")

    print(f"\nPCA uygulanıyor (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    print("PCA tamamlandı.")

    # PC sütun adlarını dinamik olarak oluştur
    pc_columns = [f'PC{i + 1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pc_columns, index=genotype_df.index)
    pca_df['super_pop'] = y_labels.loc[pca_df.index]

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"\nİlk {n_components} Temel Bileşenin Açıkladığı Varyans Oranları:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"  PC{i + 1}: {ratio * 100:.2f}%")
    print(f"  Toplam Açıklanan Varyans (ilk {n_components} PC): {sum(explained_variance_ratio) * 100:.2f}%")

    # Çıktı dizininin var olduğundan emin ol
    os.makedirs(output_dir, exist_ok=True)

    # PCA sonuçlarını csv'ye kaydetme
    chromosome_match = re.search(r"chr(\w+)", genotype_csv_path)
    chromosome_str = chromosome_match.group(1) if chromosome_match else "unknown_chr"
    output_csv_filename = os.path.join(output_dir,
                                       f"{output_csv_filename_prefix}_{n_components}components_{chromosome_str}.csv")
    try:
        print(f"\nPCA sonuçları '{output_csv_filename}' dosyasına kaydediliyor...")
        pca_df.to_csv(output_csv_filename, index=True)
        print(f"PCA sonuçları başarıyla '{output_csv_filename}' olarak kaydedildi.")
    except Exception as e:
        print(f"HATA: PCA sonuçları csv'ye kaydedilirken bir sorun oluştu: {e}")

    # Grafik çizimi sadece 2 veya 3 bileşen için mümkün
    if n_components == 2:
        print("\n2D PCA grafiği oluşturuluyor...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        super_populations = sorted(pca_df['super_pop'].unique())
        pop_colors = {'AFR': 'red', 'AMR': 'green', 'EAS': 'blue', 'EUR': 'purple', 'SAS': 'orange'}

        for i, super_pop in enumerate(super_populations):
            indices_to_plot = pca_df['super_pop'] == super_pop
            ax.scatter(pca_df.loc[indices_to_plot, 'PC1'],
                       pca_df.loc[indices_to_plot, 'PC2'],
                       label=super_pop,
                       color=pop_colors.get(super_pop,
                                            plt.cm.get_cmap('viridis')(i / max(1, len(super_populations) - 1))),
                       s=50, alpha=0.7)

        ax.set_xlabel(f'Temel Bileşen 1 ({explained_variance_ratio[0] * 100:.2f}%)', fontsize=12)
        ax.set_ylabel(f'Temel Bileşen 2 ({explained_variance_ratio[1] * 100:.2f}%)', fontsize=12)
        ax.set_title(f'1000 Genomes Örnekleri için 2D PCA (n_components={n_components})', fontsize=16)
        ax.legend(title='Süper Popülasyonlar')
        ax.grid(True)
        output_plot_filename = os.path.join(output_dir,
                                            f"{output_plot_filename_prefix}_{n_components}components_{chromosome_str}.png")
        try:
            plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
            print(f"Grafik '{output_plot_filename}' olarak kaydedildi.")
        except Exception as e:
            print(f"Grafik kaydedilirken hata oluştu: {e}")
        plt.show()

    elif n_components == 3:
        print("\n3D PCA grafiği oluşturuluyor...")
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        super_populations = sorted(pca_df['super_pop'].unique())
        pop_colors = {'AFR': 'red', 'AMR': 'green', 'EAS': 'blue', 'EUR': 'purple', 'SAS': 'orange'}

        for i, super_pop in enumerate(super_populations):
            indices_to_plot = pca_df['super_pop'] == super_pop
            ax.scatter(pca_df.loc[indices_to_plot, 'PC1'],
                       pca_df.loc[indices_to_plot, 'PC2'],
                       pca_df.loc[indices_to_plot, 'PC3'],
                       label=super_pop,
                       color=pop_colors.get(super_pop, plt.cm.get_cmap('viridis')(
                           i / max(1, len(super_populations) - 1) if len(super_populations) > 1 else 0)),
                       s=50, alpha=0.7)

        ax.set_xlabel(f'Temel Bileşen 1 ({explained_variance_ratio[0] * 100:.2f}%)', fontsize=12)
        ax.set_ylabel(f'Temel Bileşen 2 ({explained_variance_ratio[1] * 100:.2f}%)', fontsize=12)
        ax.set_zlabel(f'Temel Bileşen 3 ({explained_variance_ratio[2] * 100:.2f}%)', fontsize=12)
        ax.set_title(f'1000 Genomes Örnekleri için 3D PCA (n_components={n_components})', fontsize=16)
        ax.legend(title='Süper Popülasyonlar')
        ax.grid(True)
        output_plot_filename = os.path.join(output_dir,
                                            f"{output_plot_filename_prefix}_{n_components}components_{chromosome_str}.png")
        try:
            plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
            print(f"Grafik '{output_plot_filename}' olarak kaydedildi.")
        except Exception as e:
            print(f"Grafik kaydedilirken hata oluştu: {e}")
        plt.show()
    else:
        print(f"\n{n_components} bileşenli PCA için grafik çizilemiyor (sadece 2D veya 3D grafikler desteklenir).")


if __name__ == '__main__':
    # 1. Genotip matrisi CSV dosyasının yolu (bu, input dosyanız)
    csv_dosya_yolu = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\genotip_matrisi_tum_snpler_chr22.csv"

    # 2. Panel dosyasının bilgisayarınızda kaydedileceği/bulunacağı yol ve adı (bu da bir input dosyası)
    panel_dosya_yerel_yolu = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\integrated_call_samples_v3.20130502.ALL.panel"

    # 3. Oluşturulacak PCA sonuçlarının (CSV ve grafik) kaydedileceği dizin yolu (bu output dizini)
    output_kayit_dizini = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\source\results"

    # Kullanıcıdan temel bileşen sayısını al
    while True:
        try:
            num_components = int(input("PCA için temel bileşen sayısını girin (örneğin, 2, 3, 5): "))
            if num_components >= 1:
                break
            else:
                print("Lütfen 1 veya daha büyük bir tam sayı girin.")
        except ValueError:
            print("Geçersiz giriş. Lütfen bir sayı girin.")

    print(f"\nGenotip CSV dosyası yolu: {csv_dosya_yolu}")
    print(f"Panel dosyası için yerel yol: {panel_dosya_yerel_yolu}")
    print(f"Panel dosyası URL: {PANEL_FILE_URL}")
    print(f"PCA sonuçlarının kaydedileceği dizin: {output_kayit_dizini}")
    print(f"Seçilen Temel Bileşen Sayısı: {num_components}")

    # Ana fonksiyonu çağıralım
    # output_kayit_dizini parametresini ekledik
    perform_dynamic_pca_and_plot(csv_dosya_yolu, panel_dosya_yerel_yolu, num_components, output_kayit_dizini)