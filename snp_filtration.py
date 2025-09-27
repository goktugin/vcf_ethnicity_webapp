import pandas as pd
import os


class SNPFilter:
    """
    SNP'leri p-değerlerine göre filtrelemek ve genotip matrisini güncellemek için bir sınıf.
    """

    def _init_(self, genotype_filepath, pvalue_filepath):
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
        self.snp_id_col = None  # Gerçek SNP ID sütun adı burada saklanacak
        self.p_value_col = None  # Gerçek P-değeri sütun adı burada saklanacak

        print(f"SNPFilter başlatıldı.")
        print(f"  Genotip dosyası: {self.genotype_filepath}")
        print(f"  P-değeri dosyası: {self.pvalue_filepath}")

    def load_data(self, snp_id_col_spec, p_value_col_spec):

        try:
            print(f"\nGenotip matrisi yükleniyor: {self.genotype_filepath}")
            self.genotype_df = pd.read_csv(self.genotype_filepath, index_col=0)
            print(f"  Genotip matrisi başarıyla yüklendi. Boyut: {self.genotype_df.shape} (örnekler, SNP'ler)")
            print(f"  Genotip matrisindeki ilk 5 SNP (sütun): {self.genotype_df.columns[:5].tolist()}")

            print(f"\nP-değeri dosyası yükleniyor: {self.pvalue_filepath}")
            self.pvalue_df = pd.read_csv(self.pvalue_filepath)
            print(f"  P-değeri dosyası başarıyla yüklendi. Boyut: {self.pvalue_df.shape}")

            # SNP ID sütun adını belirle
            resolved_snp_id_col_name = None
            if isinstance(snp_id_col_spec, str):
                resolved_snp_id_col_name = snp_id_col_spec
            elif isinstance(snp_id_col_spec, int):
                if 0 <= snp_id_col_spec < len(self.pvalue_df.columns):
                    resolved_snp_id_col_name = self.pvalue_df.columns[snp_id_col_spec]
                    print(
                        f"  Bilgi: SNP ID sütunu için {snp_id_col_spec}. indeks (0 tabanlı) kullanılıyor: '{resolved_snp_id_col_name}'")
                else:
                    print(
                        f"  HATA: P-değeri dosyasında SNP ID için geçersiz sütun indeksi: {snp_id_col_spec}. Toplam sütun sayısı: {len(self.pvalue_df.columns)}")
                    return False
            else:
                print(
                    f"  HATA: SNP ID sütun belirteci ('{snp_id_col_spec}') geçersiz tipte ({type(snp_id_col_spec)}). String (isim) veya int (indeks) olmalı.")
                return False

            # P-değeri sütun adını belirle
            resolved_p_value_col_name = None
            if isinstance(p_value_col_spec, str):
                resolved_p_value_col_name = p_value_col_spec
            elif isinstance(p_value_col_spec, int):
                if 0 <= p_value_col_spec < len(self.pvalue_df.columns):
                    resolved_p_value_col_name = self.pvalue_df.columns[p_value_col_spec]
                    print(
                        f"  Bilgi: P-değeri sütunu için {p_value_col_spec}. indeks (0 tabanlı) kullanılıyor: '{resolved_p_value_col_name}'")
                else:
                    print(
                        f"  HATA: P-değeri dosyasında P-değeri için geçersiz sütun indeksi: {p_value_col_spec}. Toplam sütun sayısı: {len(self.pvalue_df.columns)}")
                    return False
            else:
                print(
                    f"  HATA: P-değeri sütun belirteci ('{p_value_col_spec}') geçersiz tipte ({type(p_value_col_spec)}). String (isim) veya int (indeks) olmalı.")
                return False

            # Belirlenen sütun adlarının varlığını kontrol et
            if resolved_snp_id_col_name not in self.pvalue_df.columns:
                print(f"  HATA: P-değeri dosyasında SNP ID sütunu ('{resolved_snp_id_col_name}') bulunamadı.")
                print(f"  Dosyadaki mevcut sütunlar: {self.pvalue_df.columns.tolist()}")
                return False
            if resolved_p_value_col_name not in self.pvalue_df.columns:
                print(f"  HATA: P-değeri dosyasında p-değeri sütunu ('{resolved_p_value_col_name}') bulunamadı.")
                print(f"  Dosyadaki mevcut sütunlar: {self.pvalue_df.columns.tolist()}")
                return False

            self.snp_id_col = resolved_snp_id_col_name
            self.p_value_col = resolved_p_value_col_name  # Düzeltilmiş atama
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
            self.pvalue_df[self.p_value_col] = pd.to_numeric(self.pvalue_df[self.p_value_col], errors='coerce')
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
            if not top_snp_ids and num_snps_to_select > 0:
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

    # run_filtering_pipeline metodu, her bir kromozom için ayrı ayrı çağrılmayacağından
    # ve genel akış ana blokta yönetileceğinden bu haliyle gerekli olmayabilir
    # veya tek bir kromozom işleme için korunabilir. Şimdilik olduğu gibi bırakıyorum.
    def run_filtering_pipeline(self, num_snps_to_select, output_filepath,
                               snp_id_col_specification, p_value_col_specification):
        """
        Tek bir dosya çifti için tüm filtreleme adımlarını çalıştırır.
        """
        print("=" * 50)
        print(f"Tekil SNP Filtreleme İşlemi Başlatılıyor ({os.path.basename(self.genotype_filepath)})...")
        print("=" * 50)

        if not self.load_data(snp_id_col_specification, p_value_col_specification):
            print("\nVeri yükleme hatası nedeniyle işlem durduruldu.")
            return None  # DataFrame döndürmesi bekleniyorsa

        top_snp_ids = self.get_top_snps(num_snps_to_select)
        if top_snp_ids is None or not top_snp_ids:
            print("\nEn iyi SNP'ler seçilemediği (veya boş liste döndüğü) için işlem durduruldu.")
            return None

        filtered_genotype_df = self.filter_genotype_matrix(top_snp_ids)
        if filtered_genotype_df is None:
            print("\nGenotip matrisi filtrelenemediği için işlem durduruldu.")
            return None

        if self.save_filtered_matrix(filtered_genotype_df, output_filepath):
            print("\nTekil SNP filtreleme işlemi başarıyla tamamlandı!")
        else:
            print("\nTekil SNP filtreleme işlemi hatalarla tamamlandı.")
        print("=" * 50)
        return filtered_genotype_df


if _name_ == '_main_':
    # ----- KULLANICI TARAFINDAN AYARLANACAK PARAMETRELER -----
    # Dosya yolları için şablonlar (kromozom numarası {} ile belirtilecek)
    base_directory = r"C:\Users\goktugin\PycharmProjects\PythonProject\vcf_ethnicity_predictor\results\anova_and_matrix_for_each_chr"
    genotype_file_template = os.path.join(base_directory, "genotip_matrisi_tum_snpler_chr{}.csv")
    pvalue_file_template = os.path.join(base_directory, "anova_snps_chr{}_pvalues.csv")

    # P-değeri dosyasındaki sütunları belirtme:
    pvalue_snp_column_spec = 2  # Örnek: 3. sütun SNP ID ise, indeks 2
    pvalue_p_column_spec = "P_Value"  # Örnek: Sütun adı "P_Value" ise

    # ----- ÇIKTI DOSYASI AYARLARI -----
    output_directory = base_directory  # Çıktıyı aynı results klasörüne kaydedelim

    # Kullanıcıdan her kromozom için kaç SNP seçileceğini al
    while True:
        try:
            num_top_snps_input = input(
                "Her bir kromozom için kaç adet en düşük p-değerine sahip SNP seçmek istersiniz? (örn: 100, 500): ")
            num_top_snps_per_chromosome = int(num_top_snps_input)
            if num_top_snps_per_chromosome <= 0:
                print("Lütfen pozitif bir tam sayı girin.")
            else:
                break
        except ValueError:
            print("Geçersiz giriş. Lütfen bir sayı girin.")

    # Birleştirilmiş sonuçlar için çıktı dosyası adı
    combined_output_filename = f"filtrelenmis_genotip_TUM_KROMOZOM_top{num_top_snps_per_chromosome}_snps.csv"
    combined_output_filepath = os.path.join(output_directory, combined_output_filename)
    print(f"\nFiltrelenmiş ve birleştirilmiş genotip matrisi şuraya kaydedilecek: '{combined_output_filepath}'")

    all_filtered_chromosome_dfs = []  # Her kromozomdan gelen filtrelenmiş DataFrame'leri tutacak liste

    print("\n" + "=" * 60)
    print("TÜM KROMOZOMLAR İÇİN SNP FİLTRELEME VE BİRLEŞTİRME İŞLEMİ BAŞLATILIYOR...")
    print("=" * 60)

    for chr_num in range(1, 23):  # Kromozom 1'den 22'ye kadar
        current_genotype_file = genotype_file_template.format(chr_num)
        current_pvalue_file = pvalue_file_template.format(chr_num)

        print(f"\n===== KROMOZOM {chr_num} İŞLENİYOR =====")

        if not os.path.exists(current_genotype_file):
            print(f"HATA: Genotip dosyası bulunamadı, atlanıyor: {current_genotype_file}")
            continue
        if not os.path.exists(current_pvalue_file):
            print(f"HATA: P-değeri dosyası bulunamadı, atlanıyor: {current_pvalue_file}")
            continue

        # Her kromozom için yeni bir SNPFilter nesnesi oluştur
        snp_filter_tool_chr = SNPFilter(genotype_filepath=current_genotype_file,
                                        pvalue_filepath=current_pvalue_file)

        # Verileri yükle
        if not snp_filter_tool_chr.load_data(pvalue_snp_column_spec, pvalue_p_column_spec):
            print(f"Kromozom {chr_num} için veri yükleme hatası. Bu kromozom atlanıyor.")
            continue

        # En iyi SNP'leri seç
        top_snp_ids_chr = snp_filter_tool_chr.get_top_snps(num_top_snps_per_chromosome)
        if top_snp_ids_chr is None or not top_snp_ids_chr:
            print(f"Kromozom {chr_num} için en iyi SNP'ler seçilemedi veya hiç SNP bulunamadı. Bu kromozom atlanıyor.")
            continue

        # Genotip matrisini filtrele
        filtered_genotype_df_chr = snp_filter_tool_chr.filter_genotype_matrix(top_snp_ids_chr)
        if filtered_genotype_df_chr is None or filtered_genotype_df_chr.empty:
            print(f"Kromozom {chr_num} için genotip matrisi filtrelenemedi veya sonuç boş. Bu kromozom atlanıyor.")
            continue

        all_filtered_chromosome_dfs.append(filtered_genotype_df_chr)
        print(f"Kromozom {chr_num} için {filtered_genotype_df_chr.shape[1]} SNP başarıyla işlendi ve eklendi.")
        print("=" * 30)

    # Tüm filtrelenmiş DataFrame'leri birleştir
    if not all_filtered_chromosome_dfs:
        print(
            "\nUYARI: Hiçbir kromozomdan filtrelenmiş veri toplanamadı. Birleştirilmiş çıktı dosyası oluşturulmayacak.")
    else:
        print(f"\n{len(all_filtered_chromosome_dfs)} adet kromozoma ait filtrelenmiş veriler birleştiriliyor...")
        # Örneklerin (satırların) aynı olduğu varsayımıyla yatay birleştirme (axis=1)
        # Eğer örnek ID'leri (index) farklılık gösteriyorsa veya sıraları bozuksa,
        # birleştirmeden önce hizalama gerekebilir. Şu anki yapı index_col=0 ile yüklendiği için
        # pd.concat index'e göre hizalama yapacaktır.
        try:
            final_combined_df = pd.concat(all_filtered_chromosome_dfs, axis=1)
            print(
                f"  Birleştirme tamamlandı. Toplam {final_combined_df.shape[1]} SNP içeren birleşik matris oluşturuldu.")
            print(f"  Birleşik matris boyutu: {final_combined_df.shape} (örnekler, SNP'ler)")

            # Birleştirilmiş matrisi kaydet
            # SNPFilter sınıfındaki save_filtered_matrix metodunu yeniden kullanabiliriz.
            # Geçici bir SNPFilter nesnesi oluşturup (dosya yolları önemli değil, sadece metodu kullanacağız)
            # veya save_filtered_matrix'i statik bir metoda dönüştürebiliriz ya da doğrudan burada kaydedebiliriz.
            # Basitlik adına doğrudan burada kaydedelim:

            if final_combined_df.empty:
                print("UYARI: Birleştirilmiş veri boş. Çıktı dosyası oluşturulacak ancak içi boş olacak.")

            output_main_dir = os.path.dirname(combined_output_filepath)
            if output_main_dir and not os.path.exists(output_main_dir):
                os.makedirs(output_main_dir, exist_ok=True)
                print(f"  Çıktı dizini oluşturuldu: {output_main_dir}")

            final_combined_df.to_csv(combined_output_filepath, index=True)
            print(
                f"\nBirleştirilmiş ve filtrelenmiş genotip matrisi başarıyla '{combined_output_filepath}' olarak kaydedildi.")

            # Sütun adı tekrarı kontrolü (opsiyonel ama faydalı)
            if final_combined_df.columns.duplicated().any():
                num_duplicates = final_combined_df.columns.duplicated().sum()
                print(
                    f"  UYARI: Birleştirilmiş DataFrame'de {num_duplicates} adet mükerrer SNP ID'si (sütun adı) bulundu!")
                print(
                    f"    Mükerrer sütunlardan bazıları: {final_combined_df.columns[final_combined_df.columns.duplicated()].tolist()[:5]}")
                print(
                    f"    Bu durum, farklı kromozom dosyalarında aynı SNP ID'lerinin olmasına (beklenmedik) veya bir hataya işaret edebilir.")


        except Exception as e:
            print(f"  HATA: Filtrelenmiş DataFrame'ler birleştirilirken veya kaydedilirken bir hata oluştu - {e}")

    print("\n" + "=" * 60)
    print("SNP FİLTRELEME VE BİRLEŞTİRME İŞLEMİ TAMAMLANDI.")
    print("=" * 60)