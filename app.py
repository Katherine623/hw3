import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Spam/Ham Classifier",
    page_icon="📧",
    layout="wide"
)

# Add title and description
st.title("垃圾郵件分類系統 - Spam/Ham Classifier")
st.markdown("""
這是一個使用機器學習技術的垃圾郵件分類系統。系統能夠分析郵件內容，並預測它是垃圾郵件（spam）還是正常郵件（ham）。
""")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the CSV file
        file_path = os.path.join(current_dir, 'sms_spam_clean.csv')
        
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please make sure the data file 'sms_spam_clean.csv' is in the same directory as the app.")
        return None

# Load the data
df = load_data()

if df is None:
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["預測", "數據分析"])

with tab1:
    st.header("郵件分類預測")
    
    # Text input for prediction
    user_input = st.text_area("請輸入要分類的郵件內容：", height=100)
    
    # Model training and prediction
    if user_input:
        # Prepare the model
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['text_clean'])
        y = (df['col_0'] == 'spam').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make prediction
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        probability = model.predict_proba(user_vector)
        
        # Show results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("預測結果")
            if prediction[0] == 1:
                st.error("⚠️ 這可能是垃圾郵件 (Spam)")
            else:
                st.success("✅ 這可能是正常郵件 (Ham)")
        
        with col2:
            st.subheader("預測機率")
            st.write(f"垃圾郵件機率: {probability[0][1]:.2%}")
            st.write(f"正常郵件機率: {probability[0][0]:.2%}")
            
            # Display probability bar
            st.progress(float(probability[0][1]))

with tab2:
    st.header("數據分析")
    
    # Display class distribution
    st.subheader("類別分布")
    class_dist = df['col_0'].value_counts()
    fig = px.pie(values=class_dist.values, names=class_dist.index, 
                 title="Ham vs Spam Distribution")
    st.plotly_chart(fig)
    
    # Display message length distribution
    st.subheader("郵件長度分布")
    df['message_length'] = df['text_clean'].str.len()
    fig = px.histogram(df, x='message_length', color='col_0', 
                      title="Message Length Distribution by Class",
                      labels={'message_length': 'Message Length', 'col_0': 'Class'})
    st.plotly_chart(fig)
    
    # Model performance metrics
    st.subheader("模型性能")
    
    # Prepare data for metrics
    X = TfidfVectorizer(max_features=1000).fit_transform(df['text_clean'])
    y = (df['col_0'] == 'spam').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("模型準確率", f"{accuracy_score(y_test, y_pred):.2%}")
    
    with col2:
        st.metric("預測正確率", f"{accuracy_score(y_test, y_pred):.2%}")

# Footer
st.markdown("---")
st.markdown("作者：[Your Name] | [GitHub]()")
