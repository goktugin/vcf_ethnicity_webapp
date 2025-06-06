import pandas as pd
import os


class SNPFilter:
    """
    SNP'leri p-değerlerine göre filtrelemek ve genotip matrisini güncellemek için bir sınıf.
    """

    def __init__(self, genotype_filepath, pvalue_filepath):
        """
        Sınıfı başlatır.

        Args:
            genotype_filepath (str): Genotip matrisini içeren CSV dosyasının yolu.
            pvalue_filepath (str): SNP p-değerlerini içeren CSV dosyasının yolu.
        """
        self.genotype_filepath = genotype_filepath
        self.pvalue_filepath = pvalue_filepath
        self.genotype_df = None
        self.pvalue_df = None
        self.snp_id_col = None  # Gerçek sütun adı burada saklanacak
        self.p_value_col = None  # Gerçek sütun adı burada saklanacak

        print(f"SNPFilter başlatıldı.")
        print(f"  Genotip dosyası: {self.genotype_filepath}")
        print(f"  P-değeri dosyası: {self.pvalue_filepath}")

    def load_data(self, snp_id_col_spec, p_value_col_spec):
        """
        Genotip ve p-değeri dosyalarını yükler.
        Sütun belirteçleri (spec) sütun adı (str) veya 0-tabanlı sütun indeksi (int) olabilir.

        Args:
            snp_id_col_spec (str or int): P-değeri dosyasındaki SNP ID sütununun adı veya indeksi.
            p_value_col_spec (str or int): P-değeri dosyasındaki p-değeri sütununun adı veya indeksi.

        Returns:
            bool: Veriler başarıyla yüklendiyse True, aksi takdirde False.
        """
        try:
            print(f"\nGenotip matrisi yükleniyor: {self.genotype_filepath}")
            self.genotype_df = pd.read_csv(self.genotype_filepath, index_col=0)
            print(f"  Genotip matrisi başarıyla yüklendi. Boyut: {self.genotype_df.shape} (örnekler, SNP'ler)")
            print(f"  Genotip matrisindeki ilk 5 SNP (sütun): {self.genotype_df.columns[:5].tolist()}")

            print(f"\nP-değeri dosyası yükleniyor: {self.pvalue_filepath}")
            self.pvalue_df = pd.read_csv(self.pvalue_filepath)
            print(f"  P-değeri dosyası başarıyla yüklendi. Boyut: {self.pvalue_df.shape}")

            # SNP ID sütun adını belirle
            if isinstance(snp_id_col_spec, str):
                actual_snp_id_col_name = snp_id_col_spec
            elif isinstance(snp_id_col_spec, int):
                if 0 <= snp_id_col_spec < len(self.pvalue_df.columns):
                    actual_snp_id_col_name = self.pvalue_df.columns[snp_id_col_spec]
                    print(
                        f"  Bilgi: SNP ID sütunu için {snp_id_col_spec}. indeks (0 tabanlı) kullanılıyor: '{actual_snp_id_col_name}'")
                else:
                    print(
                        f"  HATA: P-değeri dosyasında SNP ID için geçersiz sütun indeksi: {snp_id_col_spec}. Toplam sütun sayısı: {len(self.pvalue_df.columns)}")
                    return False
            else:
                print(
                    f"  HATA: SNP ID sütun belirteci ('{snp_id_col_spec}') geçersiz tipte ({type(snp_id_col_spec)}). String (isim) veya int (indeks) olmalı.")
                return False

            # P-değeri sütun adını belirle
            if isinstance(p_value_col_spec, str):
                actual_p_value_col_name = p_value_col_spec
            elif isinstance(p_value_col_spec, int):
                if 0 <= p_value_col_spec < len(self.pvalue_df.columns):
                    actual_p_value_col_name = self.pvalue_df.columns[p_value_col_spec]
                    print(
                        f"  Bilgi: P-değeri sütunu için {p_value_col_spec}. indeks (0 tabanlı) kullanılıyor: '{actual_p_value_col_name}'")
                else:
                    print(
                        f"  HATA: P-değeri dosyasında P-değeri için geçersiz sütun indeksi: {p_value_col_spec}. Toplam sütun sayısı: {len(self.pvalue_df.columns)}")
                    return False
            else:
                print(
                    f"  HATA: P-değeri sütun belirteci ('{p_value_col_spec}') geçersiz tipte ({type(p_value_col_spec)}). String (isim) veya int (indeks) olmalı.")
                return False

            # Belirlenen sütun adlarının varlığını kontrol et
            if actual_snp_id_col_name not in self.pvalue_df.columns:
                print(f"  HATA: P-değeri dosyasında SNP ID sütunu ('{actual_snp_id_col_name}') bulunamadı.")
                print(f"  Dosyadaki mevcut sütunlar: {self.pvalue_df.columns.tolist()}")
                return False
            if actual_p_value_col_name not in self.pvalue_df.columns:
                print(f"  HATA: P-değeri dosyasında p-değeri sütunu ('{actual_p_value_col_name}') bulunamadı.")
                print(f"  Dosyadaki mevcut sütunlar: {self.pvalue_df.columns.tolist()}")
                return False

            self.snp_id_col = actual_snp_id_col_name
            self.p_value_col = actual_p_value_col_name
            print(f"  P-değeri dosyasında kullanılacak SNP ID sütunu (çözümlendi): '{self.snp_id_col}'")
            print(f"  P-değeri dosyasında kullanılacak P-değeri sütunu (çözümlendi): '{self.p_value_col}'")

            return True

        except FileNotFoundError as e:
            print(f"  HATA: Dosya bulunamadı - {e}")
            return False
        except Exception as e:
            print(f"  HATA: Veri yüklenirken bir hata oluştu - {e}")
            return False

    def get_top_snps(self, num_snps_to_select):
        """
        P-değerlerine göre en iyi N SNP'yi seçer.

        Args:
            num_snps_to_select (int): Seçilecek en iyi SNP sayısı.

        Returns:
            list: En iyi SNP ID'lerinin listesi veya bir hata durumunda None.
        """
        if self.pvalue_df is None or self.snp_id_col is None or self.p_value_col is None:
            print(
                "HATA: P-değeri verisi veya sütun adları yüklenmemiş/belirlenmemiş. Lütfen önce load_data() metodunu çalıştırın.")
            return None

        try:
            print(f"\nEn iyi {num_snps_to_select} SNP seçiliyor ('{self.p_value_col}' sütununa göre)...")
            # P-değerlerini sayısal tipe dönüştürmeyi dene, hatalı olanları NaN yapar
            self.pvalue_df[self.p_value_col] = pd.to_numeric(self.pvalue_df[self.p_value_col], errors='coerce')
            # NaN değerleri olan satırları sıralamadan önce kaldır (veya en sona at)
            pvalue_df_cleaned = self.pvalue_df.dropna(subset=[self.p_value_col])
            if len(pvalue_df_cleaned) < len(self.pvalue_df):
                print(
                    f"  UYARI: P-değeri sütununda ('{self.p_value_col}') {len(self.pvalue_df) - len(pvalue_df_cleaned)} adet sayısal olmayan veya eksik değer bulundu ve sıralamadan çıkarıldı.")

            sorted_pvalues = pvalue_df_cleaned.sort_values(by=self.p_value_col, ascending=True)

            if num_snps_to_select > len(sorted_pvalues):
                print(
                    f"  UYARI: İstenen SNP sayısı ({num_snps_to_select}), geçerli p-değerine sahip SNP sayısından ({len(sorted_pvalues)}) fazla.")
                print(f"  Mevcut tüm geçerli SNP'ler ({len(sorted_pvalues)}) seçilecek.")
                num_snps_to_select = len(sorted_pvalues)

            top_snps_df = sorted_pvalues.head(num_snps_to_select)
            top_snp_ids = top_snps_df[self.snp_id_col].tolist()

            print(
                f"  P-değeri dosyasından ('{self.snp_id_col}' sütunundan) {len(top_snp_ids)} adet en iyi SNP ID'si başarıyla seçildi.")
            if not top_snp_ids and num_snps_to_select > 0:  # Sadece gerçekten SNP seçilmesi bekleniyorsa uyarı ver
                print(
                    f"  UYARI: P-değeri dosyasından hiç SNP seçilemedi. İstenen sayı: {num_snps_to_select}. P-değeri sütununda geçerli veri olmayabilir.")
            return top_snp_ids

        except KeyError as e:
            print(
                f"  HATA: P-değeri dosyasında belirtilen sütun adı ('{e}') bulunamadı. self.snp_id_col='{self.snp_id_col}', self.p_value_col='{self.p_value_col}'")
            return None
        except Exception as e:
            print(f"  HATA: En iyi SNP'ler seçilirken bir hata oluştu - {e}")
            return None

    def filter_genotype_matrix(self, top_snp_ids):
        """
        Genotip matrisini verilen SNP ID'lerine göre filtreler.

        Args:
            top_snp_ids (list): Filtreleme için kullanılacak SNP ID'lerinin listesi.

        Returns:
            pandas.DataFrame: Filtrelenmiş genotip matrisi veya bir hata durumunda None.
        """
        if self.genotype_df is None:
            print("HATA: Genotip verisi yüklenmemiş. Lütfen önce load_data() metodunu çalıştırın.")
            return None
        if top_snp_ids is None or not top_snp_ids:
            print("HATA: Filtreleme için geçerli SNP ID listesi sağlanmadı veya boş.")
            return None

        try:
            print(f"\nGenotip matrisi {len(top_snp_ids)} adet seçilmiş SNP'ye göre filtreleniyor...")

            snps_present_in_genotype = [snp for snp in top_snp_ids if snp in self.genotype_df.columns]

            if not snps_present_in_genotype:
                print("  HATA: P-değeri dosyasından seçilen en iyi SNP'lerden hiçbiri genotip matrisinde bulunamadı.")
                print(f"    P-değeri dosyasından ilk 5 SNP adayı: {top_snp_ids[:5]}")
                print(f"    Genotip matrisindeki ilk 5 SNP: {self.genotype_df.columns[:5].tolist()}")
                return None

            print(
                f"  Seçilen {len(top_snp_ids)} SNP'den {len(snps_present_in_genotype)} tanesi genotip matrisinde bulundu ve kullanılacak.")
            if len(snps_present_in_genotype) < len(top_snp_ids):
                print(
                    f"  UYARI: {len(top_snp_ids) - len(snps_present_in_genotype)} adet SNP, genotip matrisinde bulunmadığı için filtrelenmiş matrise dahil edilmeyecek.")

            filtered_genotype_df = self.genotype_df[snps_present_in_genotype]
            print(f"  Genotip matrisi başarıyla filtrelendi. Yeni boyut: {filtered_genotype_df.shape}")
            return filtered_genotype_df

        except Exception as e:
            print(f"  HATA: Genotip matrisi filtrelenirken bir hata oluştu - {e}")
            return None

    def save_filtered_matrix(self, filtered_df, output_filepath):
        """
        Filtrelenmiş genotip matrisini CSV dosyasına kaydeder.

        Args:
            filtered_df (pandas.DataFrame): Kaydedilecek filtrelenmiş genotip matrisi.
            output_filepath (str): Çıktı CSV dosyasının yolu.

        Returns:
            bool: Kaydetme başarılıysa True, aksi takdirde False.
        """
        if filtered_df is None:
            print("HATA: Kaydedilecek filtrelenmiş veri yok.")
            return False
        if filtered_df.empty:
            print("UYARI: Filtrelenmiş veri boş. Çıktı dosyası oluşturulacak ancak içi boş olacak.")

        try:
            print(f"\nFiltrelenmiş genotip matrisi kaydediliyor: {output_filepath}")
            # Çıktı dizini mevcut değilse oluştur
            output_dir = os.path.dirname(output_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"  Çıktı dizini oluşturuldu: {output_dir}")

            filtered_df.to_csv(output_filepath, index=True)
            print(f"  Filtrelenmiş genotip matrisi başarıyla '{output_filepath}' olarak kaydedildi.")
            return True
        except Exception as e:
            print(f"  HATA: Filtrelenmiş matris kaydedilirken bir hata oluştu - {e}")
            return False

    def run_filtering_pipeline(self, num_snps_to_select, output_filepath,
                               snp_id_col_specification, p_value_col_specification):
        """
        Tüm filtreleme adımlarını çalıştırır: yükle, en iyi SNP'leri seç, filtrele, kaydet.

        Args:
            num_snps_to_select (int): Seçilecek en iyi SNP sayısı.
            output_filepath (str): Çıktı CSV dosyasının yolu.
            snp_id_col_specification (str or int): P-değeri dosyasındaki SNP ID sütununun adı veya 0-tabanlı indeksi.
            p_value_col_specification (str or int): P-değeri dosyasındaki p-değeri sütununun adı veya 0-tabanlı indeksi.
        """
        print("=" * 50)
        print("SNP Filtreleme İşlemi Başlatılıyor...")
        print("=" * 50)

        if not self.load_data(snp_id_col_specification, p_value_col_specification):
            print("\nVeri yükleme hatası nedeniyle işlem durduruldu.")
            return

        top_snp_ids = self.get_top_snps(num_snps_to_select)
        if top_snp_ids is None or not top_snp_ids:
            print("\nEn iyi SNP'ler seçilemediği (veya boş liste döndüğü) için işlem durduruldu.")
            return

        filtered_genotype_df = self.filter_genotype_matrix(top_snp_ids)
        if filtered_genotype_df is None:
            print("\nGenotip matrisi filtrelenemediği için işlem durduruldu.")
            return

        if self.save_filtered_matrix(filtered_genotype_df, output_filepath):
            print("\nSNP filtreleme işlemi başarıyla tamamlandı!")
        else:
            print("\nSNP filtreleme işlemi hatalarla tamamlandı.")
        print("=" * 50)


if __name__ == '__main__':
    # ----- KULLANICI TARAFINDAN AYARLANACAK PARAMETRELER -----
    genotype_file = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\genotip_matrisi_tum_snpler_chr22.csv"
    pvalue_file = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\anova_snps_chr22_pvalues.csv"

    # P-değeri dosyasındaki sütunları belirtme:
    # SNP ID Sütunu:
    # Eğer sütun adını biliyorsanız, tırnak içinde yazın: örn. "MarkerName"
    # Eğer sütunun sıra numarasını biliyorsanız (0'dan başlar), sayıyı yazın:
    # Sizin belirttiğiniz gibi 3. sütun SNP ID ise, indeks olarak 2 kullanın.
    pvalue_snp_column_spec = 2

    # P-Değeri Sütunu:
    # Eğer sütun adını biliyorsanız, tırnak içinde yazın: örn. "P.Value"
    # Eğer sütunun sıra numarasını biliyorsanız (0'dan başlar), sayıyı yazın:
    pvalue_p_column_spec = "P_Value"

    # ----- ÇIKTI DOSYASI AYARLARI -----
    # Çıktı klasörünü tanımlayın
    output_directory = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results"

    if not os.path.exists(genotype_file):
        print(f"HATA: Genotip dosyası bulunamadı: {genotype_file}")
        exit()
    if not os.path.exists(pvalue_file):
        print(f"HATA: P-değeri dosyası bulunamadı: {pvalue_file}")
        exit()

    while True:
        try:
            num_top_snps_input = input("Kaç adet en düşük p-değerine sahip SNP seçmek istersiniz? (örn: 500, 5000): ")
            num_top_snps = int(num_top_snps_input)
            if num_top_snps <= 0:
                print("Lütfen pozitif bir tam sayı girin.")
            else:
                break
        except ValueError:
            print("Geçersiz giriş. Lütfen bir sayı girin.")

    # Otomatik olarak kaydedilecek çıktı dosya adı
    output_file = os.path.join(output_directory, f"filtrelenmis_genotip_top{num_top_snps}_snps.csv")
    print(f"\nFiltrelenmiş genotip matrisi otomatik olarak şuraya kaydedilecek: '{output_file}'")

    snp_filter_tool = SNPFilter(genotype_filepath=genotype_file, pvalue_filepath=pvalue_file)

    snp_filter_tool.run_filtering_pipeline(
        num_snps_to_select=num_top_snps,
        output_filepath=output_file,
        snp_id_col_specification=pvalue_snp_column_spec,
        p_value_col_specification=pvalue_p_column_spec
    )