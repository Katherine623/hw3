# 垃圾郵件分類系統 (Spam/Ham Classifier)

> **課程：** Machine Learning 2025  
> **作業：** Homework 3 - Email Spam Classification  
> **學號：** 5114056002

## 📋 專案簡介

這是一個使用機器學習技術的垃圾郵件分類系統，能夠分析郵件內容並預測它是垃圾郵件（spam）還是正常郵件（ham）。本專案遵循 **CRISP-DM** 方法論和 **OpenSpec** 規範驅動開發流程。

## 🎯 專案目標

- 實作完整的機器學習管線（資料預處理 → 模型訓練 → 評估 → 部署）
- 比較多種分類演算法的效能
- 提供互動式 Web 界面進行即時預測
- 視覺化模型性能指標和資料分布

## 🚀 線上展示

**Streamlit 應用程式：** https://btf8ctwa2exth9qpdw9log.streamlit.app/

## 📊 資料集

- **來源：** SMS Spam Collection Dataset
- **檔案：** `sms_spam_clean.csv`
- **規模：** 5,572 筆訊息
- **類別：** ham（正常郵件）和 spam（垃圾郵件）
- **特徵：** 文本內容及預處理後的清理文本

## 🛠️ 技術棧

### 核心技術
- **程式語言：** Python 3.8+
- **機器學習：** scikit-learn
- **資料處理：** pandas, numpy
- **視覺化：** plotly, matplotlib
- **Web 框架：** Streamlit

### 機器學習模型
- ✅ Random Forest Classifier（已實作）
- 🔄 Logistic Regression（規劃中）
- 🔄 Naïve Bayes（規劃中）
- 🔄 Support Vector Machine（規劃中）

## 📁 專案結構

```
5114056002_HW3/
├── openspec/
│   ├── project.md          # 專案上下文文件
│   ├── AGENTS.md           # AI 協作工作流程指南
│   └── proposals/          # 功能變更提案
├── app.py                  # 主要 Streamlit 應用程式
├── requirements.txt        # Python 依賴套件
├── sms_spam_clean.csv      # 資料集
├── README.md               # 專案說明文件
└── .gitignore              # Git 忽略規則
```

## 🔧 安裝指南

### 1. 克隆專案

```bash
git clone https://github.com/Katherine623/hw3.git
cd hw3
```

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 執行應用程式

```bash
streamlit run app.py
```

應用程式將在瀏覽器中自動開啟，預設網址為 `http://localhost:8501`

## 💡 使用說明

### 預測功能
1. 在「預測」分頁中的文本輸入框輸入要分類的郵件內容
2. 系統會即時顯示分類結果（spam 或 ham）
3. 顯示預測機率和信心度

### 資料分析
1. 切換到「數據分析」分頁
2. 查看資料集統計資訊
3. 檢視模型性能指標

## 📈 模型性能

目前使用 Random Forest Classifier：
- **準確率：** ~97%
- **特徵：** TF-IDF 向量化（top 1000 features）
- **訓練/測試比例：** 80/20

## 🎓 CRISP-DM 流程

### Phase 1: 商業理解
- 目標：建立垃圾郵件自動分類系統
- 成功標準：準確率 > 95%

### Phase 2: 資料理解
- 探索 SMS 資料集
- 分析類別分布
- 識別資料品質問題

### Phase 3: 資料準備
- 文本清理
- TF-IDF 向量化
- 訓練/測試集分割

### Phase 4: 建模
- 實作多種分類器
- 超參數調整
- 模型比較

### Phase 5: 評估
- 準確率、精確率、召回率、F1 分數
- 混淆矩陣
- ROC 曲線

### Phase 6: 部署
- Streamlit 互動式應用
- GitHub 版本控制
- Streamlit Cloud 部署

## 📝 OpenSpec 工作流程

本專案採用 OpenSpec 規範驅動開發：

1. **專案上下文：** 參閱 `openspec/project.md`
2. **AI 協作指南：** 參閱 `openspec/AGENTS.md`
3. **功能提案：** 新功能請先建立提案於 `openspec/proposals/`

### 建立新功能提案
```bash
# 範例提案格式
openspec/proposals/001-feature-name.md
```

## 🔄 版本控制慣例

### Commit 訊息格式
```
[Phase N] 簡短描述

詳細說明（如需要）

Related to: openspec/proposals/XXX-proposal-name.md
```

### 範例
```
Phase 2 – Modeling: Add Naïve Bayes classifier

實作 MultinomialNB 模型並加入模型比較儀表板

Related to: openspec/proposals/002-naive-bayes.md
```

## 🐛 已知問題與限制

- 目前僅實作 Random Forest 模型
- 視覺化功能較為基礎
- 尚未實作模型持久化

## 🚧 未來改進

- [ ] 新增 Logistic Regression 模型
- [ ] 新增 Naïve Bayes 模型
- [ ] 新增 SVM 模型
- [ ] 實作混淆矩陣視覺化
- [ ] 新增 ROC 曲線圖
- [ ] 實作模型比較儀表板
- [ ] 新增批次預測功能
- [ ] 實作模型儲存與載入

## 📚 參考資源

- [Hands-On AI for Cybersecurity - GitHub](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
- [教學影片播放清單](https://www.youtube.com/watch?v=FeCCYFK0TJ8&list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c)
- [OpenSpec 教學](https://www.youtube.com/watch?v=ANjiJQQIBo0)
- [Streamlit 文件](https://docs.streamlit.io/)
- [scikit-learn 文件](https://scikit-learn.org/)

## 👤 作者

**姓名：** Katherine623  
**學號：** 5114056002  
**課程：** Machine Learning 2025  

## 📄 授權

MIT License

---

**最後更新：** 2025-10-22