# 🥇 Gold Price Time Series Prediction

Dự đoán giá vàng theo chuỗi thời gian sử dụng 4 kiến trúc deep learning: **LSTM, BiLSTM, GRU, CNN**. Bao gồm cả hai chế độ dự đoán: **one-step** (1 ngày tới) và **multi-step** (10 ngày tới).

---

## ✨ Features

- **EDA & Visualization** — trực quan hóa biến động giá vàng 2013–2023
- **4 model architectures** — LSTM, BiLSTM, GRU, CNN-1D
- **One-step forecasting** — dự đoán 1 ngày, dùng kết quả để forecast 15 ngày tiếp theo (recursive)
- **Multi-step forecasting** — dự đoán trực tiếp 10 ngày, chuỗi 30 ngày bằng cách ghép các block
- **3 evaluation metrics** — MAPE, Accuracy, RMSE

---

## 📊 Dataset

**File:** `Gold Price (2013-2023).csv`

| Thông số | Giá trị |
|----------|---------|
| Phạm vi thời gian | 2013 – 2023 |
| Tập train | 2013 – 2021 |
| Tập test | 2022 (256 observations) |

**Các cột dữ liệu:**

| Cột | Mô tả | Ghi chú |
|-----|-------|---------|
| `Date` | Ngày giao dịch | → datetime |
| `Price` | Giá đóng cửa (USD) | Target |
| `Open` | Giá mở cửa | Feature |
| `High` | Giá cao nhất ngày | Feature |
| `Low` | Giá thấp nhất ngày | Feature |
| `Vol.` | Khối lượng giao dịch (K) | Feature (remove 'K') |
| `Change %` | % thay đổi ngày | **Bị loại bỏ** |

---

## 🔄 Pipeline

```
Raw CSV
    ↓
Data Cleaning
  - Xóa hàng NaN
  - Strip dấu phẩy trong số
  - Remove 'K' từ cột Volume
  - Chuyển dtype: Date→datetime, còn lại→float64
  - Sắp xếp tăng dần theo ngày
    ↓
Drop 'Change %'
    ↓
MinMaxScaler → [0, 1]
    ↓
        ┌──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼
 One-Step Forecasting           Multi-Step Forecasting
 Lookback: 30 ngày              Input: 90 ngày
 Output: 1 ngày                 Output: 10 ngày
        ↓                              ↓
 LSTM / BiLSTM / GRU / CNN      LSTM / GRU
        ↓                              ↓
 Recursive 15-day forecast      Chained 30-day forecast
        ↓                              ↓
        └──────────────────────────────┘
                    ↓
           MAPE / Accuracy / RMSE
```

---

## 🤖 One-Step Models

### LSTM

```
Input (30, 5)
  → LSTM(128, return_sequences=True)
  → Dropout(0.2)
  → LSTM(64, return_sequences=False)
  → Dense(32)
  → Output(1)
```

### GRU

```
Input (30, 5)
  → GRU(64)
  → Dropout(0.1)
  → Dense(32)
  → Dropout(0.1)
  → Output(1)
```

### BiLSTM

```
Input (30, 5)
  → Bidirectional(LSTM(64))
  → Dropout(0.1)
  → Dense(32)
  → Dropout(0.1)
  → Output(1)
```

### CNN-1D

```
Input (30, 5)
  → Conv1D(64 filters, kernel_size=3)
  → MaxPooling1D(pool_size=3)
  → Flatten
  → Dense(64)
  → Dropout(0.1)
  → Output(1)
```

**Training config (One-Step):**

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 64 – 128 |
| Validation split | 10% |
| Optimizer | Adam |
| Loss | MSE |

---

## 🤖 Multi-Step Models

### LSTM Multi-Step

```
Input (90, 5)
  → LSTM(64)
  → Dropout(0.1)
  → Dense(32)
  → Output(50)  # reshape → (10, 5)
```

### GRU Multi-Step

```
Input (90, 5)
  → GRU(64)
  → Dropout(0.1)
  → Dense(32)
  → Output(50)  # reshape → (10, 5)
```

**Training config (Multi-Step):**

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 128 |
| Validation split | 10% |
| Optimizer | Adam |
| Loss | MSE |

---

## 📏 Evaluation Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **MAPE** | `mean(|y - ŷ| / y) × 100` | % sai lệch trung bình |
| **Accuracy** | `(1 - MAPE) × 100` | % chính xác |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Biên độ dao động lỗi |

---

## 🔮 Forecasting Strategy

**One-Step (Recursive):**
```
[day 1..30] → predict day 31
[day 2..31] → predict day 32
...
→ forecast 15 future days
```

**Multi-Step (Chained blocks):**
```
[day 1..90]  → predict days 91–100
[day 11..100] → predict days 101–110
...
→ forecast 30 future days
```

> **Lưu ý:** Multi-step forecast xa (>10 ngày) cho độ chính xác thấp hơn do giá vàng thiếu tính tuần hoàn rõ ràng.

---

## 📁 Project Structure

```
Gold-price-prediction/
├── gold-price-time-series-prediction.ipynb   # Toàn bộ pipeline: EDA, train, evaluate, forecast
└── Gold Price (2013-2023).csv                # Dataset giá vàng lịch sử
```

---

## 🚀 Getting Started

### Yêu cầu

- Python 3.8+
- Jupyter Notebook
- GPU (khuyến nghị cho training nhanh hơn)

### Cài đặt

```bash
git clone https://github.com/fong-AI/Gold-price-prediction.git
cd Gold-price-prediction
pip install tensorflow keras pandas numpy scikit-learn matplotlib
```

### Chạy notebook

```bash
jupyter notebook gold-price-time-series-prediction.ipynb
```

---

## 📦 Dependencies

| Thư viện | Mục đích |
|----------|---------|
| `tensorflow` / `keras` | LSTM, GRU, BiLSTM, CNN models |
| `pandas` | Xử lý dữ liệu |
| `numpy` | Ma trận, sliding window |
| `scikit-learn` | MinMaxScaler, metrics |
| `matplotlib` | Visualization |

---

## 📝 Notes

- `Change %` bị loại vì nó là derived feature (tính từ Price), không phải independent input
- Temporal split nghiêm ngặt (train ≤ 2021, test = 2022) — tránh data leakage
- Input shape 3D: `(samples, timesteps, features)` phù hợp cả 4 kiến trúc
- BiLSTM xử lý chuỗi theo cả hai chiều — thường tốt hơn LSTM đơn hướng trên time series có pattern phức tạp

---

## 📄 License

This project is for personal / educational use.
