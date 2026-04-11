# Petro·ML — Petrophysics Machine Learning App

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run petro_ml_app.py
```

## Fitur

- Upload LAS: multi-file training + 1 test well
- Pilih log input bebas: GR, NPHI, RHOB, RT
- Pilih target: VSH, PHIE, SW
- Toggle derived features: DN_SEP, NPHI-RHOB Crossover, ZONE_ENC
- Parameter LightGBM interaktif
- Plot log 6-track: GR | RHOB+NPHI | RT | VSH | PHIE | SW (aktual vs prediksi)
- Scatter plot aktual vs prediksi
- Download hasil prediksi CSV

## Urutan Prediksi

Model dilatih secara berurutan:
1. VSH → hasilnya (VSH_PRED) dipakai sebagai feature untuk PHIE & SW
2. PHIE → hasilnya (PHIE_PRED) dipakai sebagai feature untuk SW
3. SW → menggunakan VSH_PRED + PHIE_PRED + crossover features

## Notes

- RT nilai ≤ 0 otomatis di-NaN (tidak valid untuk log10)
- VSH = 0 dihapus dari training (dianggap log error)
- Baris dengan NaN di log input dihapus sebelum training
