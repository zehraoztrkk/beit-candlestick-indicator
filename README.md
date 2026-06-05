# 🚀 BEiT Candlestick Indicator - Canlı Al/Sat Strateji Paneli

Bu proje, finansal piyasalardaki mum grafiklerini analiz ederek yatırımcılara anlık karar destek mekanizması sunmak amacıyla geliştirilmiş bilgisayarlı görü (Computer Vision) tabanlı bir derin öğrenme sistemidir.

## 🧠 Model Mimarisi
Projenin omurgasında Transfer Learning mantığıyla **Vision Transformer (ViT)** tabanlı `microsoft/beit-base-finetuned-ade-640-640` modeli kullanılmıştır. Model, grafiklerdeki mum formasyonlarını bütünsel (Self-Attention) olarak inceler.

### Eklenen Özel Katmanlar (Custom Head):
* **Linear Katmanı:** 768 -> 512 boyut indirgeme.
* **ReLU Aktivasyonu:** Doğrusal olmayan karmaşık kalıpların çözülmesi.
* **Dropout (0.3):** Aşırı öğrenmeyi (Overfitting) engelleme.
* **Linear Çıkış:** 3 ana sınıf için (AL/SAT/TUT) skor üretimi.

## 🛠️ Kullanılan Teknolojies
* **PyTorch & Torchvision** (Model Eğitimi)
* **Transformers (Hugging Face)** (BEiT Altyapısı)
* **Gradio** (Canlı Web Arayüzü)
* **OpenCV & PIL** (Görüntü İşleme)

## 📊 Eğitim Parametreleri
* **Learning Rate:** 2e-5 (Fine-tuning için optimum hassasiyet)
* **Epoch:** 8
* **Batch Size:** 4
* **Loss Function:** Sınıf ağırlıklı CrossEntropyLoss `[2.5, 1.0, 1.0]`

## 🌍 Canlı Demo (Deployment)
Model Hugging Face Space üzerinde mikroservis olarak çalışmaktadır.
👉 https://huggingface.co/spaces/ceruka/candle-graph
