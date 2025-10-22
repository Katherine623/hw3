import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import time

st.set_page_config(page_title="垃圾郵件分類系統", page_icon="📧", layout="wide")
st.title("📧 垃圾郵件分類系統")
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
        df.columns = df.columns.str.strip()
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

try:
    df = df.dropna(subset=['text_clean', 'col_0'])
    df['text_clean'] = df['text_clean'].astype(str)
    df = df[df['text_clean'].str.strip() != '']
    df = df.reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["text_clean"])
    y = (df["col_0"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
except Exception as e:
    st.error(f"資料處理錯誤: {str(e)}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 數據概覽", "🤖 模型訓練", "🔮 即時預測", "📈 性能分析"])

with tab1:
    st.header("數據集概覽")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總樣本數", len(df), delta="100%")
    with col2:
        spam_count = (df["col_0"] == "spam").sum()
        st.metric("垃圾郵件", spam_count, delta=f"{spam_count/len(df)*100:.1f}%")
    with col3:
        ham_count = (df["col_0"] == "ham").sum()
        st.metric("正常郵件", ham_count, delta=f"{ham_count/len(df)*100:.1f}%")
    with col4:
        spam_ratio = (df['col_0'] == 'spam').mean()
        st.metric("垃圾比例", f"{spam_ratio:.1%}", delta="不平衡" if spam_ratio < 0.3 else "平衡")
    
    col1, col2 = st.columns(2)
    with col1:
        class_dist = df["col_0"].value_counts()
        fig = px.pie(values=class_dist.values, names=class_dist.index, 
                     title="類別分布", hole=0.4,
                     color_discrete_map={'ham':'#636EFA', 'spam':'#EF553B'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_temp = df.copy()
        df_temp["length"] = df_temp["text_clean"].str.len()
        fig = px.box(df_temp, x="col_0", y="length", color="col_0",
                     title="郵件長度分布（箱型圖）",
                     labels={"col_0": "類別", "length": "文字長度"},
                     color_discrete_map={'ham':'#636EFA', 'spam':'#EF553B'})
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        df_temp["word_count"] = df_temp["text_clean"].str.split().str.len()
        fig = px.histogram(df_temp, x="word_count", color="col_0", 
                          title="詞數分布", nbins=50,
                          labels={"word_count": "詞數", "count": "數量"},
                          color_discrete_map={'ham':'#636EFA', 'spam':'#EF553B'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        all_words = ' '.join(df[df['col_0']=='spam']['text_clean']).split()
        word_freq = Counter(all_words).most_common(15)
        fig = go.Figure(go.Bar(
            x=[w[1] for w in word_freq],
            y=[w[0] for w in word_freq],
            orientation='h',
            marker=dict(color='#EF553B')
        ))
        fig.update_layout(title="垃圾郵件最常見詞彙 Top 15",
                         xaxis_title="出現次數", yaxis_title="詞彙",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📝 資料樣本")
    sample_type = st.radio("選擇樣本類型", ["垃圾郵件", "正常郵件"], horizontal=True)
    sample_df = df[df['col_0'] == ('spam' if sample_type == '垃圾郵件' else 'ham')].sample(5)
    st.dataframe(sample_df[['col_0', 'text_clean']].rename(columns={'col_0': '類別', 'text_clean': '內容'}), 
                use_container_width=True, hide_index=True)

with tab2:
    st.header("模型訓練與評估")
    if model_choice == "所有模型比較":
        model_names = ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"]
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for idx, model_name in enumerate(model_names):
            status_text.text(f"正在訓練 {model_name}...")
            result = train_model(model_name, X_train, y_train, X_test, y_test)
            all_results.append(result)
            progress_bar.progress((idx + 1) / len(model_names))
        status_text.text("✅ 所有模型訓練完成！")
        
        metrics_df = pd.DataFrame([{"模型": r["model_name"], "準確率": f"{r['accuracy']:.4f}", "精確率": f"{r['precision']:.4f}", "召回率": f"{r['recall']:.4f}", "F1分數": f"{r['f1']:.4f}", "訓練時間": f"{r['train_time']:.3f}s"} for r in all_results])
        st.dataframe(metrics_df, use_container_width=True)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("準確率", "精確率", "召回率", "F1分數"))
        models = [r["model_name"] for r in all_results]
        positions = [(1,1), (1,2), (2,1), (2,2)]
        metrics = ["accuracy", "precision", "recall", "f1"]
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        for idx, metric in enumerate(metrics):
            row, col = positions[idx]
            values = [r[metric] for r in all_results]
            fig.add_trace(go.Bar(x=models, y=values, marker_color=colors), row=row, col=col)
            fig.update_yaxes(range=[0, 1], row=row, col=col)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_time = go.Figure(go.Bar(
                x=models,
                y=[r['train_time'] for r in all_results],
                marker_color=colors,
                text=[f"{r['train_time']:.3f}s" for r in all_results],
                textposition='auto'
            ))
            fig_time.update_layout(title="模型訓練時間比較", 
                                  xaxis_title="模型", yaxis_title="時間（秒）")
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            fig_radar = go.Figure()
            for idx, r in enumerate(all_results):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[r['accuracy'], r['precision'], r['recall'], r['f1'], r['accuracy']],
                    theta=['準確率', '精確率', '召回率', 'F1分數', '準確率'],
                    fill='toself',
                    name=r['model_name']
                ))
            fig_radar.update_layout(title="模型性能雷達圖",
                                   polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        with st.spinner(f"正在訓練 {model_choice} 模型..."):
            result = train_model(model_choice, X_train, y_train, X_test, y_test)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("準確率", f"{result['accuracy']:.4f}", 
                     delta=f"{(result['accuracy']-0.5)*100:.1f}%" if result['accuracy'] > 0.5 else None)
        with col2:
            st.metric("精確率", f"{result['precision']:.4f}")
        with col3:
            st.metric("召回率", f"{result['recall']:.4f}")
        with col4:
            st.metric("F1分數", f"{result['f1']:.4f}")
        with col5:
            st.metric("訓練時間", f"{result['train_time']:.3f}s")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_cm = create_confusion_matrix(y_test, result["y_pred"], model_choice)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            if result["y_prob"] is not None:
                fig_roc = create_roc_curve(y_test, result["y_prob"], model_choice)
                st.plotly_chart(fig_roc, use_container_width=True)
        
        if result["y_prob"] is not None:
            col1, col2 = st.columns(2)
            with col1:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, result["y_prob"])
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, 
                                           mode='lines', fill='tozeroy',
                                           name='PR Curve'))
                fig_pr.update_layout(title=f"{model_choice} - Precision-Recall 曲線",
                                    xaxis_title="Recall", yaxis_title="Precision",
                                    height=400)
                st.plotly_chart(fig_pr, use_container_width=True)
            
            with col2:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=result["y_prob"][y_test==0], 
                                               name='Ham', opacity=0.7, marker_color='#636EFA'))
                fig_dist.add_trace(go.Histogram(x=result["y_prob"][y_test==1], 
                                               name='Spam', opacity=0.7, marker_color='#EF553B'))
                fig_dist.update_layout(title="預測分數分布", barmode='overlay',
                                      xaxis_title="Spam 機率", yaxis_title="樣本數",
                                      height=400)
                st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("即時預測")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        pred_model = st.selectbox("選擇模型", ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"])
    with col2:
        use_example = st.button("📝 使用範例郵件", use_container_width=True)
    
    example_spam = "URGENT! You have won $1,000,000! Click here to claim your prize NOW! Limited time offer!"
    example_ham = "Hey, are you free for lunch tomorrow? Let me know what time works for you."
    
    if use_example:
        example_type = st.radio("選擇範例類型", ["垃圾郵件範例", "正常郵件範例"], horizontal=True)
        user_input = st.text_area("輸入郵件內容：", 
                                 value=example_spam if example_type == "垃圾郵件範例" else example_ham,
                                 height=150)
    else:
        user_input = st.text_area("輸入郵件內容：", height=150, 
                                 placeholder="請輸入要分類的郵件內容...")
    
    if user_input:
        with st.spinner("分析中..."):
            result = train_model(pred_model, X_train, y_train, X_test, y_test)
            model = result["model"]
            user_vector = vectorizer.transform([user_input])
            prediction = model.predict(user_vector)[0]
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(user_vector)[0]
                spam_prob = prob[1]
                ham_prob = prob[0]
            else:
                spam_prob = None
                ham_prob = None
        
        st.subheader("📊 預測結果")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction == 1:
                st.error("### ⚠️ 垃圾郵件 (Spam)", icon="⚠️")
            else:
                st.success("### ✅ 正常郵件 (Ham)", icon="✅")
        
        if spam_prob is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("垃圾郵件機率", f"{spam_prob:.2%}", 
                         delta="高風險" if spam_prob > 0.7 else "低風險")
                st.progress(float(spam_prob))
            with col2:
                st.metric("正常郵件機率", f"{ham_prob:.2%}",
                         delta="可信" if ham_prob > 0.7 else "存疑")
                st.progress(float(ham_prob))
            
            fig_prob = go.Figure(data=[go.Pie(
                labels=['Ham', 'Spam'],
                values=[ham_prob, spam_prob],
                hole=0.5,
                marker_colors=['#636EFA', '#EF553B']
            )])
            fig_prob.update_layout(title="分類機率分布", height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        st.subheader("📝 文本統計")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("字元數", len(user_input))
        with col2:
            st.metric("詞數", len(user_input.split()))
        with col3:
            st.metric("平均詞長", f"{np.mean([len(w) for w in user_input.split()]):.1f}")
        with col4:
            st.metric("大寫字元比例", f"{sum(1 for c in user_input if c.isupper())/len(user_input)*100:.1f}%")

with tab4:
    st.header("性能分析與專案文件")
    
    st.subheader("🎯 模型特性比較")
    comparison_data = {
        "模型": ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"],
        "優點": [
            "高準確率、處理非線性關係",
            "快速、可解釋性強",
            "適合文本分類、訓練快速",
            "高維空間效果好、記憶體效率高"
        ],
        "缺點": [
            "訓練時間較長、模型較大",
            "僅能處理線性關係",
            "假設特徵獨立",
            "對參數敏感"
        ],
        "最適用場景": [
            "需要高準確率的生產環境",
            "需要快速訓練和推理",
            "文本分類基準模型",
            "大規模文本分類"
        ]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.subheader("📋 CRISP-DM 方法論")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🔄 專案流程階段
        1. **商業理解** - 垃圾郵件過濾需求
        2. **資料理解** - SMS 資料集分析
        3. **資料準備** - 文本清理與向量化
        4. **建模** - 多模型訓練與比較
        5. **評估** - 性能指標分析
        6. **部署** - Streamlit Cloud 部署
        """)
    
    with col2:
        phases = ['商業理解', '資料理解', '資料準備', '建模', '評估', '部署']
        progress = [100, 100, 100, 100, 100, 100]
        fig_progress = go.Figure(go.Bar(
            x=progress,
            y=phases,
            orientation='h',
            marker=dict(color='#00CC96'),
            text=[f"{p}%" for p in progress],
            textposition='inside'
        ))
        fig_progress.update_layout(title="專案完成度", xaxis_title="完成度 (%)",
                                  height=300, xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_progress, use_container_width=True)
    
    st.subheader("🛠️ 技術棧")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **機器學習**
        - scikit-learn
        - TF-IDF Vectorization
        - 4種分類算法
        """)
    with col2:
        st.markdown("""
        **資料處理**
        - Pandas
        - NumPy
        - 文本預處理
        """)
    with col3:
        st.markdown("""
        **視覺化與部署**
        - Streamlit
        - Plotly
        - GitHub + Streamlit Cloud
        """)
    
    st.subheader("📊 性能基準指標")
    st.info("""
    **評估標準**：
    - ✅ 準確率 > 95%：優秀
    - ✅ 精確率 > 90%：減少誤判
    - ✅ 召回率 > 85%：降低漏檢
    - ✅ F1分數 > 90%：平衡性能
    """)

st.markdown("---")
st.markdown("作者：Katherine623 | 學號：5114056002 | [GitHub](https://github.com/Katherine623/hw3)")
