import streamlit as stimport streamlit as stimport streamlit as st

import pandas as pd

import numpy as npimport pandas as pdimport pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizerimport numpy as npfrom sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import train_test_splitfrom sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVCfrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score, precision_score, recall_score, f1_score,from sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import accuracy_score

    confusion_matrix, roc_curve, auc

)from sklearn.linear_model import LogisticRegression

import plotly.express as px

import plotly.graph_objects as gofrom sklearn.naive_bayes import MultinomialNB# Set page configuration

from plotly.subplots import make_subplots

import timefrom sklearn.svm import LinearSVCst.set_page_config(



# Set page configurationfrom sklearn.metrics import (    page_title="Spam/Ham Classifier",

st.set_page_config(

    page_title="垃圾郵件分類系統",    accuracy_score, precision_score, recall_score, f1_score,    page_icon="📧",

    page_icon="📧",

    layout="wide"    confusion_matrix, roc_curve, auc    layout="wide"

)

))

# Title

st.title("📧 垃圾郵件分類系統")import plotly.express as px

st.markdown("使用多種機器學習模型進行垃圾郵件分類")

import plotly.graph_objects as go# Add title and description

# Sidebar

with st.sidebar:from plotly.subplots import make_subplotsst.title("垃圾郵件分類系統 - Spam/Ham Classifier")

    st.title("設定選項")

    import timest.markdown("""

    model_choice = st.selectbox(

        "選擇分類器",這是一個使用機器學習技術的垃圾郵件分類系統。系統能夠分析郵件內容，並預測它是垃圾郵件（spam）還是正常郵件（ham）。

        ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)", "所有模型比較"]

    )# Set page configuration""")

    

    test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)st.set_page_config(

    max_features = st.slider("最大特徵數 (TF-IDF)", 500, 2000, 1000, 100)

    page_title="垃圾郵件分類系統",# Load and preprocess data

# Load data

@st.cache_data    page_icon="📧",@st.cache_data

def load_data():

    try:    layout="wide"def load_data():

        df = pd.read_csv('sms_spam_clean.csv')

        return df)    try:

    except Exception as e:

        st.error(f"讀取資料錯誤: {str(e)}")        df = pd.read_csv('sms_spam_clean.csv')

        return None

# Title        return df

# Train model

@st.cache_datast.title("📧 垃圾郵件分類系統")    except Exception as e:

def train_model(model_name, X_train, y_train, X_test, y_test):

    if model_name == "Random Forest":st.markdown("使用多種機器學習模型進行垃圾郵件分類")        st.error(f"讀取資料錯誤: {str(e)}")

        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_name == "Logistic Regression":        return None

        model = LogisticRegression(max_iter=1000, random_state=42)

    elif model_name == "Naïve Bayes":# Sidebar

        model = MultinomialNB()

    else:  # SVMwith st.sidebar:# Load the data

        model = LinearSVC(random_state=42, max_iter=2000)

        st.title("設定選項")df = load_data()

    start_time = time.time()

    model.fit(X_train, y_train)    

    train_time = time.time() - start_time

        model_choice = st.selectbox(if df is None:

    y_pred = model.predict(X_test)

            "選擇分類器",    st.stop()

    # Get probabilities if available

    if hasattr(model, 'predict_proba'):        ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)", "所有模型比較"]

        y_prob = model.predict_proba(X_test)[:, 1]

    else:    )# Create tabs

        y_prob = None

        tab1, tab2 = st.tabs(["預測", "數據分析"])

    return {

        'model': model,    test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)

        'model_name': model_name,

        'accuracy': accuracy_score(y_test, y_pred),    max_features = st.slider("最大特徵數 (TF-IDF)", 500, 2000, 1000, 100)with tab1:

        'precision': precision_score(y_test, y_pred),

        'recall': recall_score(y_test, y_pred),    st.header("郵件分類預測")

        'f1': f1_score(y_test, y_pred),

        'train_time': train_time,# Load data    

        'y_pred': y_pred,

        'y_prob': y_prob@st.cache_data    # Text input for prediction

    }

def load_data():    user_input = st.text_area("請輸入要分類的郵件內容：", height=100)

def create_confusion_matrix(y_test, y_pred, model_name):

    cm = confusion_matrix(y_test, y_pred)    try:    

    fig = go.Figure(data=go.Heatmap(

        z=cm,        df = pd.read_csv('sms_spam_clean.csv')    # Model training and prediction

        x=['Ham', 'Spam'],

        y=['Ham', 'Spam'],        return df    if user_input:

        colorscale='Blues',

        text=cm,    except Exception as e:        # Prepare the model

        texttemplate='%{text}',

        showscale=True        st.error(f"讀取資料錯誤: {str(e)}")        vectorizer = TfidfVectorizer(max_features=1000)

    ))

    fig.update_layout(        return None        X = vectorizer.fit_transform(df['text_clean'])

        title=f'{model_name} - 混淆矩陣',

        xaxis_title='預測類別',        y = (df['col_0'] == 'spam').astype(int)

        yaxis_title='實際類別',

        height=400# Train model        

    )

    return fig@st.cache_data        # Split the data



def create_roc_curve(y_test, y_prob, model_name):def train_model(model_name, X_train, y_train, X_test, y_test):        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y_prob is None:

        return None    if model_name == "Random Forest":        

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)        model = RandomForestClassifier(n_estimators=100, random_state=42)        # Train the model

    

    fig = go.Figure()    elif model_name == "Logistic Regression":        model = RandomForestClassifier(n_estimators=100, random_state=42)

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',

                             name=f'{model_name} (AUC = {roc_auc:.3f})'))        model = LogisticRegression(max_iter=1000, random_state=42)        model.fit(X_train, y_train)

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',

                             name='隨機', line=dict(dash='dash')))    elif model_name == "Naïve Bayes":        

    fig.update_layout(title=f'{model_name} - ROC 曲線',

                      xaxis_title='False Positive Rate',        model = MultinomialNB()        # Make prediction

                      yaxis_title='True Positive Rate',

                      height=400)    else:  # SVM        user_vector = vectorizer.transform([user_input])

    return fig

        model = LinearSVC(random_state=42, max_iter=2000)        prediction = model.predict(user_vector)

# Load data

df = load_data()            probability = model.predict_proba(user_vector)

if df is None:

    st.stop()    start_time = time.time()        



# Prepare data    model.fit(X_train, y_train)        # Show results

vectorizer = TfidfVectorizer(max_features=max_features)

X = vectorizer.fit_transform(df['text_clean'])    train_time = time.time() - start_time        col1, col2 = st.columns(2)

y = (df['col_0'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)            



# Tabs    y_pred = model.predict(X_test)        with col1:

tab1, tab2, tab3, tab4 = st.tabs(["📊 數據概覽", "🤖 模型訓練", "🔮 即時預測", "📈 性能分析"])

                st.subheader("預測結果")

with tab1:

    st.header("數據集概覽")    # Get probabilities if available            if prediction[0] == 1:

    col1, col2, col3, col4 = st.columns(4)

    with col1:    if hasattr(model, 'predict_proba'):                st.error("⚠️ 這可能是垃圾郵件 (Spam)")

        st.metric("總樣本數", len(df))

    with col2:        y_prob = model.predict_proba(X_test)[:, 1]            else:

        st.metric("垃圾郵件", (df['col_0'] == 'spam').sum())

    with col3:    else:                st.success("✅ 這可能是正常郵件 (Ham)")

        st.metric("正常郵件", (df['col_0'] == 'ham').sum())

    with col4:        y_prob = None        

        st.metric("垃圾比例", f"{(df['col_0'] == 'spam').mean():.1%}")

                with col2:

    col1, col2 = st.columns(2)

    with col1:    return {            st.subheader("預測機率")

        class_dist = df['col_0'].value_counts()

        fig = px.pie(values=class_dist.values, names=class_dist.index, title="類別分布")        'model': model,            st.write(f"垃圾郵件機率: {probability[0][1]:.2%}")

        st.plotly_chart(fig, use_container_width=True)

            'model_name': model_name,            st.write(f"正常郵件機率: {probability[0][0]:.2%}")

    with col2:

        df_temp = df.copy()        'accuracy': accuracy_score(y_test, y_pred),            

        df_temp['length'] = df_temp['text_clean'].str.len()

        fig = px.histogram(df_temp, x='length', color='col_0', title="郵件長度分布")        'precision': precision_score(y_test, y_pred),            # Display probability bar

        st.plotly_chart(fig, use_container_width=True)

        'recall': recall_score(y_test, y_pred),            st.progress(float(probability[0][1]))

with tab2:

    st.header("模型訓練與評估")        'f1': f1_score(y_test, y_pred),

    

    if model_choice == "所有模型比較":        'train_time': train_time,with tab2:

        model_names = ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"]

        all_results = []        'y_pred': y_pred,    st.header("數據分析")

        

        for model_name in model_names:        'y_prob': y_prob    

            result = train_model(model_name, X_train, y_train, X_test, y_test)

            all_results.append(result)    }    # Display basic statistics

        

        metrics_df = pd.DataFrame([{    st.subheader("數據統計")

            '模型': r['model_name'],

            '準確率': f"{r['accuracy']:.4f}",def create_confusion_matrix(y_test, y_pred, model_name):    st.write("垃圾郵件比例:", f"{(df['col_0'] == 'spam').mean():.2%}")

            '精確率': f"{r['precision']:.4f}",

            '召回率': f"{r['recall']:.4f}",    cm = confusion_matrix(y_test, y_pred)    st.write("總數據量:", len(df))

            'F1分數': f"{r['f1']:.4f}",

            '訓練時間': f"{r['train_time']:.3f}s"    fig = go.Figure(data=go.Heatmap(    

        } for r in all_results])

        st.dataframe(metrics_df, use_container_width=True)        z=cm,    # Model performance metrics

        

        # Comparison chart        x=['Ham', 'Spam'],    st.subheader("模型性能")

        fig = make_subplots(rows=2, cols=2,

                           subplot_titles=('準確率', '精確率', '召回率', 'F1分數'))        y=['Ham', 'Spam'],    

        models = [r['model_name'] for r in all_results]

        positions = [(1,1), (1,2), (2,1), (2,2)]        colorscale='Blues',    # Prepare data for metrics

        metrics = ['accuracy', 'precision', 'recall', 'f1']

                text=cm,    X = TfidfVectorizer(max_features=1000).fit_transform(df['text_clean'])

        for idx, metric in enumerate(metrics):

            row, col = positions[idx]        texttemplate='%{text}',    y = (df['col_0'] == 'spam').astype(int)

            values = [r[metric] for r in all_results]

            fig.add_trace(go.Bar(x=models, y=values), row=row, col=col)        showscale=True    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            fig.update_yaxes(range=[0, 1], row=row, col=col)

            ))    

        fig.update_layout(height=600, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)    fig.update_layout(    model = RandomForestClassifier(n_estimators=100, random_state=42)

        

    else:        title=f'{model_name} - 混淆矩陣',    model.fit(X_train, y_train)

        result = train_model(model_choice, X_train, y_train, X_test, y_test)

                xaxis_title='預測類別',    

        col1, col2, col3, col4 = st.columns(4)

        with col1:        yaxis_title='實際類別',    y_pred = model.predict(X_test)

            st.metric("準確率", f"{result['accuracy']:.4f}")

        with col2:        height=400    

            st.metric("精確率", f"{result['precision']:.4f}")

        with col3:    )    # Display metrics

            st.metric("召回率", f"{result['recall']:.4f}")

        with col4:    return fig    col1, col2 = st.columns(2)

            st.metric("F1分數", f"{result['f1']:.4f}")

            

        col1, col2 = st.columns(2)

        with col1:def create_roc_curve(y_test, y_prob, model_name):    with col1:

            fig_cm = create_confusion_matrix(y_test, result['y_pred'], model_choice)

            st.plotly_chart(fig_cm, use_container_width=True)    if y_prob is None:        st.metric("模型準確率", f"{accuracy_score(y_test, y_pred):.2%}")

        

        with col2:        return None    

            if result['y_prob'] is not None:

                fig_roc = create_roc_curve(y_test, result['y_prob'], model_choice)    fpr, tpr, _ = roc_curve(y_test, y_prob)    with col2:

                st.plotly_chart(fig_roc, use_container_width=True)

    roc_auc = auc(fpr, tpr)        st.metric("預測正確率", f"{accuracy_score(y_test, y_pred):.2%}")

with tab3:

    st.header("即時預測")    

    

    pred_model = st.selectbox("選擇模型", ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"])    fig = go.Figure()# Footer

    user_input = st.text_area("輸入郵件內容：", height=150)

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',st.markdown("---")

    if user_input:

        result = train_model(pred_model, X_train, y_train, X_test, y_test)                             name=f'{model_name} (AUC = {roc_auc:.3f})'))st.markdown("作者：[Your Name] | [GitHub]()")

        model = result['model']

            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',

        user_vector = vectorizer.transform([user_input])                             name='隨機', line=dict(dash='dash')))

        prediction = model.predict(user_vector)[0]    fig.update_layout(title=f'{model_name} - ROC 曲線',

                              xaxis_title='False Positive Rate',

        if hasattr(model, 'predict_proba'):                      yaxis_title='True Positive Rate',

            prob = model.predict_proba(user_vector)[0]                      height=400)

            spam_prob = prob[1]    return fig

        else:

            spam_prob = None# Load data

        df = load_data()

        if prediction == 1:if df is None:

            st.error("⚠️ 垃圾郵件 (Spam)")    st.stop()

        else:

            st.success("✅ 正常郵件 (Ham)")# Prepare data

        vectorizer = TfidfVectorizer(max_features=max_features)

        if spam_prob is not None:X = vectorizer.fit_transform(df['text_clean'])

            st.progress(float(spam_prob), text=f"垃圾郵件機率: {spam_prob:.2%}")y = (df['col_0'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

with tab4:

    st.header("性能分析")# Tabs

    st.markdown("""tab1, tab2, tab3, tab4 = st.tabs(["📊 數據概覽", "🤖 模型訓練", "🔮 即時預測", "📈 性能分析"])

    ### 模型說明

    with tab1:

    1. **Random Forest**: 集成學習，高準確率    st.header("數據集概覽")

    2. **Logistic Regression**: 簡單快速，可解釋性強    col1, col2, col3, col4 = st.columns(4)

    3. **Naïve Bayes**: 適合文本分類    with col1:

    4. **SVM Linear**: 高維空間效果好        st.metric("總樣本數", len(df))

        with col2:

    ### CRISP-DM 方法論        st.metric("垃圾郵件", (df['col_0'] == 'spam').sum())

        with col3:

    本專案遵循標準的 CRISP-DM 流程：        st.metric("正常郵件", (df['col_0'] == 'ham').sum())

    - Phase 1: 商業理解    with col4:

    - Phase 2: 資料理解        st.metric("垃圾比例", f"{(df['col_0'] == 'spam').mean():.1%}")

    - Phase 3: 資料準備    

    - Phase 4: 建模    col1, col2 = st.columns(2)

    - Phase 5: 評估    with col1:

    - Phase 6: 部署        class_dist = df['col_0'].value_counts()

    """)        fig = px.pie(values=class_dist.values, names=class_dist.index, title="類別分布")

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")    

st.markdown("作者：Katherine623 | 學號：5114056002 | [GitHub](https://github.com/Katherine623/hw3)")    with col2:

        df_temp = df.copy()
        df_temp['length'] = df_temp['text_clean'].str.len()
        fig = px.histogram(df_temp, x='length', color='col_0', title="郵件長度分布")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("模型訓練與評估")
    
    if model_choice == "所有模型比較":
        model_names = ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"]
        all_results = []
        
        for model_name in model_names:
            result = train_model(model_name, X_train, y_train, X_test, y_test)
            all_results.append(result)
        
        metrics_df = pd.DataFrame([{
            '模型': r['model_name'],
            '準確率': f"{r['accuracy']:.4f}",
            '精確率': f"{r['precision']:.4f}",
            '召回率': f"{r['recall']:.4f}",
            'F1分數': f"{r['f1']:.4f}",
            '訓練時間': f"{r['train_time']:.3f}s"
        } for r in all_results])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Comparison chart
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('準確率', '精確率', '召回率', 'F1分數'))
        models = [r['model_name'] for r in all_results]
        positions = [(1,1), (1,2), (2,1), (2,2)]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
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
            fig_cm = create_confusion_matrix(y_test, result['y_pred'], model_choice)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            if result['y_prob'] is not None:
                fig_roc = create_roc_curve(y_test, result['y_prob'], model_choice)
                st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    st.header("即時預測")
    
    pred_model = st.selectbox("選擇模型", ["Random Forest", "Logistic Regression", "Naïve Bayes", "SVM (Linear)"])
    user_input = st.text_area("輸入郵件內容：", height=150)
    
    if user_input:
        result = train_model(pred_model, X_train, y_train, X_test, y_test)
        model = result['model']
        
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(user_vector)[0]
            spam_prob = prob[1]
        else:
            spam_prob = None
        
        if prediction == 1:
            st.error("⚠️ 垃圾郵件 (Spam)")
        else:
            st.success("✅ 正常郵件 (Ham)")
        
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
