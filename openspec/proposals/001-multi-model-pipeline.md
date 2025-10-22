# Change Proposal: Multi-Model ML Pipeline with Enhanced Visualizations

**Status:** Approved  
**Date:** 2025-10-22  
**Author:** Katherine623

## Motivation

目前的應用程式僅使用單一模型（Random Forest），且視覺化功能有限。為了滿足作業要求並提供更完整的機器學習專案，需要：

1. 實作多種分類演算法並進行比較
2. 增加更豐富的視覺化功能
3. 提供更詳細的模型評估指標
4. 改善使用者介面和互動體驗

## Proposed Changes

### 1. 檔案：`app.py` - 重構並擴展主應用程式

#### 1.1 增加模型實作
- **Logistic Regression**
- **Multinomial Naïve Bayes**
- **Support Vector Machine (LinearSVC)**
- **Random Forest** (保留現有)

#### 1.2 增強資料預處理
- 顯示資料探索統計
- 提供多種文本清理選項
- 視覺化類別分布

#### 1.3 擴展視覺化功能
- **混淆矩陣（Confusion Matrix）**
- **ROC 曲線和 AUC 分數**
- **PR 曲線（Precision-Recall Curve）**
- **模型比較圖表**
- **特徵重要性圖**（適用於支援的模型）
- **詞雲圖**（spam vs ham）

#### 1.4 改善評估指標
- Accuracy
- Precision
- Recall
- F1-Score
- 訓練時間比較
- 預測速度比較

### 2. 新增檔案：`models.py` - 模型管理模組（可選）

將模型訓練邏輯分離到獨立模組：
```python
class SpamClassifier:
    def __init__(self, model_type='random_forest')
    def train(self, X_train, y_train)
    def predict(self, X)
    def evaluate(self, X_test, y_test)
```

### 3. 更新檔案：`requirements.txt`

新增需要的套件：
```
streamlit
pandas
scikit-learn
numpy
plotly
matplotlib
seaborn
wordcloud
```

## Implementation Plan

### Phase 1: 模型實作（優先）
1. 實作 4 種分類器
2. 建立模型訓練和評估函數
3. 實作模型比較功能

### Phase 2: 視覺化增強
1. 混淆矩陣
2. ROC 曲線
3. 模型比較圖表
4. 詞雲圖（如果時間允許）

### Phase 3: UI 改善
1. 重新設計頁面布局
2. 新增模型選擇選項
3. 新增側邊欄設定
4. 改善響應式設計

## Testing Plan

### 功能測試
- [ ] 所有模型都能成功訓練
- [ ] 預測功能正常運作
- [ ] 所有視覺化圖表正確顯示
- [ ] 指標計算準確

### 性能測試
- [ ] 頁面載入時間 < 5 秒
- [ ] 預測回應時間 < 1 秒
- [ ] 記憶體使用合理

### 相容性測試
- [ ] 在 Streamlit Cloud 成功部署
- [ ] 瀏覽器相容性（Chrome, Firefox, Safari）
- [ ] 行動裝置顯示正常

## Documentation Updates

### 需要更新的文件
1. **README.md**
   - 更新功能清單
   - 新增模型說明
   - 更新使用指南
   - 新增螢幕截圖

2. **openspec/project.md**
   - 更新技術棧
   - 更新模型列表

3. **程式碼註解**
   - 為所有新函數添加 docstring
   - 註解複雜的邏輯

## Success Criteria

- ✅ 實作至少 3 種不同的分類演算法
- ✅ 提供模型比較功能
- ✅ 顯示混淆矩陣和 ROC 曲線
- ✅ 整體準確率 > 95%
- ✅ 應用程式在 Streamlit Cloud 成功運行
- ✅ 程式碼清晰、有註解且遵循 PEP 8

## Risks and Mitigation

### 風險 1: 套件相容性問題
**影響：** 部署失敗  
**可能性：** 中  
**緩解策略：** 
- 使用經過測試的套件版本
- 在本地完整測試後才部署
- 準備精簡版備案（移除視覺化套件）

### 風險 2: 模型訓練時間過長
**影響：** 使用者體驗差  
**可能性：** 低  
**緩解策略：**
- 使用 `@st.cache_data` 快取模型
- 限制特徵數量（TF-IDF max_features）
- 使用輕量級模型參數

### 風險 3: 記憶體限制
**影響：** Streamlit Cloud 應用崩潰  
**可能性：** 低  
**緩解策略：**
- 優化資料載入
- 避免同時訓練所有模型
- 使用惰性載入

## Timeline

- **Day 1 (2025-10-22):** Phase 1 完成 - 實作所有模型
- **Day 2 (2025-10-23):** Phase 2 完成 - 視覺化功能
- **Day 3 (2025-10-24):** Phase 3 完成 - UI 改善和測試
- **Day 4 (2025-10-25):** 文件更新和最終調整
- **截止日 (2025-11-05):** 專案提交

## Approval

- **Developer:** Katherine623 ✅
- **AI Assistant:** GitHub Copilot ✅
- **Status:** Ready for implementation

---

**Next Steps:**
1. 開始實作 Phase 1：多模型實作
2. 測試每個模型的基本功能
3. 進行部署前測試
