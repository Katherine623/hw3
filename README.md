# 垃圾郵件分類系統 (Spam/Ham Classifier)

這是一個使用機器學習技術的垃圾郵件分類系統，能夠分析郵件內容並預測它是垃圾郵件（spam）還是正常郵件（ham）。

## 功能特點

- 即時文本分類預測
- 互動式數據視覺化
- 模型性能指標展示
- 響應式使用者界面

## 技術棧

- Python 3.8+
- Streamlit
- scikit-learn
- NLTK
- Pandas
- Plotly

## 安裝指南

1. 克隆專案：
```bash
git clone [你的 GitHub 倉庫 URL]
cd [專案目錄]
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 運行應用：
```bash
streamlit run app.py
```

## 使用說明

1. 在文本輸入框中輸入要分類的郵件內容
2. 系統會立即顯示分類結果和預測機率
3. 可以切換到數據分析頁面查看更多統計信息

## 數據集

使用了經過清理的 SMS Spam Collection 數據集，包含了標記為垃圾郵件和正常郵件的短信內容。

## 授權

MIT License