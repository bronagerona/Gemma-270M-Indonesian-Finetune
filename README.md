# Gemma 3 270M Indonesian Fine-Tune

Tujuan: fine-tuning **Gemma 3 270M IT** untuk respons bahasa Indonesia yang lebih alami dan komentar/instruksi singkat untuk karakter game, lalu mengekspor model ke **GGUF Q8_0**.

## Status

Pipeline awal disiapkan, tetapi **model hasil fine-tune belum tersedia** sampai workflow berhasil selesai.

## Cara menjalankan

1. Buka [google/gemma-3-270m-it di Hugging Face](https://huggingface.co/google/gemma-3-270m-it), masuk, dan setujui ketentuan akses Gemma.
2. Buat Hugging Face access token dengan izin baca model.
3. Di repo ini buka **Settings → Secrets and variables → Actions → New repository secret**. Nama secret harus `HF_TOKEN`, isinya token Hugging Face. Jangan pernah menaruh token di file atau chat.
4. Buka **Actions → Train Indonesian Gemma GGUF → Run workflow**.
5. Workflow menjalankan fine-tuning CPU, mengonversi model ke GGUF, lalu membuat GitHub Release dengan file `*.Q8_0.gguf` jika semua langkah sukses.

## Batasan penting

- Workflow **hanya berjalan saat tombol Run workflow ditekan**; tidak otomatis mulai ketika file di-commit.
- GitHub-hosted runner menggunakan CPU; proses bisa lambat dan batas waktu runner berlaku. Belum ada jaminan workflow akan selesai pada percobaan pertama.
- Hasil bergantung pada dataset dan evaluasi; fine-tuning tidak otomatis menjamin bahasa Indonesia selalu lebih bagus.
- Ukuran final GGUF Q8_0 harus diukur setelah ekspor; target di bawah 500 MB belum dapat dijamin.
- Workflow menggunakan bobot Hugging Face asli untuk training, bukan mencoba melatih langsung file GGUF yang sudah terkuantisasi.
- Model Gemma tetap tunduk pada lisensi/ketentuan Google Gemma. Periksa ketentuan sebelum mendistribusikan hasil.

## Dataset

Contoh awal ada di `data/train.jsonl`. Tambahkan contoh bahasa Indonesia yang natural, singkat, dan relevan dengan gaya karakter game. Setiap baris berupa JSON dengan kunci `instruction` dan `response`.

## Lisensi

Kode pipeline di repo ini MIT. Lisensi bobot model dasar dan hasil turunannya mengikuti ketentuan Gemma yang berlaku; lisensi kode ini tidak menggantikannya.
