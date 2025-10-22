import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="垃圾郵件分類系統", page_icon="", layout="wide")
st.title(" 垃圾郵件分類系統")
st.markdown("使用多種機器學習模型進行垃圾郵件分類")

with st.sidebar:
    st.title("設定選項")
    model_choice = st.selectbox("選擇分類器", ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)", "所有模型比較"])
    test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
    max_features = st.slider("最大特徵數 (TF-IDF)", 500, 2000, 1000, 100)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("sms_spam_clean.csv", encoding='utf-8')
        # 清理欄位名稱（移除空格）
        df.columns = df.columns.str.strip()
        # 確保必要欄位存在
        if 'text_clean' not in df.columns:
            st.error(f"找不到 'text_clean' 欄位。現有欄位: {list(df.columns)}")
            return None
        if 'col_0' not in df.columns:
            st.error(f"找不到 'col_0' 欄位。現有欄位: {list(df.columns)}")
            return None
        return df
    except Exception as e:
        st.error(f"讀取資料錯誤: {str(e)}")
        return None

def train_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Naïve Bayes":
        model = MultinomialNB()
    else:
        model = LinearSVC(random_state=42, max_iter=2000)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "model": model, "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_time": train_time, "y_pred": y_pred, "y_prob": y_prob
    }

def create_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Ham", "Spam"], y=["Ham", "Spam"], colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True))
    fig.update_layout(title=f"{model_name} - 混淆矩陣", xaxis_title="預測類別", yaxis_title="實際類別", height=400)
    return fig

def create_roc_curve(y_test, y_prob, model_name):
    if y_prob is None:
        return None
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="隨機", line=dict(dash="dash")))
    fig.update_layout(title=f"{model_name} - ROC 曲線", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
    return fig

df = load_data()
if df is None:
    st.stop()

# 資料預處理和驗證
try:
    # 移除空值
    df = df.dropna(subset=['text_clean', 'col_0'])
    
    # 確保 text_clean 是字串類型
    df['text_clean'] = df['text_clean'].astype(str)
    
    # 移除空白文本
    df = df[df['text_clean'].str.strip() != '']
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["text_clean"])
    y = (df["col_0"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
except Exception as e:
    st.error(f"資料處理錯誤: {str(e)}")
    st.error(f"資料框欄位: {list(df.columns)}")
    st.error(f"資料框形狀: {df.shape}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([" 數據概覽", " 模型訓練", " 即時預測", " 性能分析"])

with tab1:
    st.header("數據集概覽")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總樣本數", len(df))
    with col2:
        st.metric("垃圾郵件", (df["col_0"] == "spam").sum())
    with col3:
        st.metric("正常郵件", (df["col_0"] == "ham").sum())
    with col4:
        st.metric("垃圾比例", f"{(df['col_0'] == 'spam').mean():.1%}")
    col1, col2 = st.columns(2)
    with col1:
        class_dist = df["col_0"].value_counts()
        fig = px.pie(values=class_dist.values, names=class_dist.index, title="類別分布")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_temp = df.copy()
        df_temp["length"] = df_temp["text_clean"].str.len()
        fig = px.histogram(df_temp, x="length", color="col_0", title="郵件長度分布")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("模型訓練與評估")
    if model_choice == "所有模型比較":
        model_names = ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"]
        all_results = []
        for model_name in model_names:
            result = train_model(model_name, X_train, y_train, X_test, y_test)
            all_results.append(result)
        metrics_df = pd.DataFrame([{"模型": r["model_name"], "準確率": f"{r['accuracy']:.4f}", "精確率": f"{r['precision']:.4f}", "召回率": f"{r['recall']:.4f}", "F1分數": f"{r['f1']:.4f}", "訓練時間": f"{r['train_time']:.3f}s"} for r in all_results])
        st.dataframe(metrics_df, use_container_width=True)
        fig = make_subplots(rows=2, cols=2, subplot_titles=("準確率", "精確率", "召回率", "F1分數"))
        models = [r["model_name"] for r in all_results]
        positions = [(1,1), (1,2), (2,1), (2,2)]
        metrics = ["accuracy", "precision", "recall", "f1"]
        for idx, metric in enumerate(metrics):
            row, col = positions[idx]
            values = [r[metric] for r in all_results]
            fig.add_trace(go.Bar(x=models, y=values), row=row, col=col)
            fig.update_yaxes(range=[0, 1], row=row, col=col)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        result = train_model(model_choice, X_train, y_train, X_test, y_test)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("準確率", f"{result['accuracy']:.4f}")
        with col2:
            st.metric("精確率", f"{result['precision']:.4f}")
        with col3:
            st.metric("召回率", f"{result['recall']:.4f}")
        with col4:
            st.metric("F1分數", f"{result['f1']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            fig_cm = create_confusion_matrix(y_test, result["y_pred"], model_choice)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            if result["y_prob"] is not None:
                fig_roc = create_roc_curve(y_test, result["y_prob"], model_choice)
                st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    st.header("即時預測")
    pred_model = st.selectbox("選擇模型", ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"])
    user_input = st.text_area("輸入郵件內容：", height=150)
    if user_input:
        result = train_model(pred_model, X_train, y_train, X_test, y_test)
        model = result["model"]
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(user_vector)[0]
            spam_prob = prob[1]
        else:
            spam_prob = None
        if prediction == 1:
            st.error(" 垃圾郵件 (Spam)")
        else:
            st.success(" 正常郵件 (Ham)")
        if spam_prob is not None:
            st.progress(float(spam_prob), text=f"垃圾郵件機率: {spam_prob:.2%}")

with tab4:
    st.header("性能分析")
    st.markdown("""
    ### 模型說明
    1. **Random Forest**: 集成學習，高準確率
    2. **Logistic Regression**: 簡單快速，可解釋性強
    3. **Naïve Bayes**: 適合文本分類
    4. **SVM Linear**: 高維空間效果好
    
    ### CRISP-DM 方法論
    本專案遵循標準的 CRISP-DM 流程：
    - Phase 1: 商業理解
    - Phase 2: 資料理解
    - Phase 3: 資料準備
    - Phase 4: 建模
    - Phase 5: 評估
    - Phase 6: 部署
    """)

st.markdown("---")
st.markdown("作者：Katherine623 | 學號：5114056002 | [GitHub](https://github.com/Katherine623/hw3)")
