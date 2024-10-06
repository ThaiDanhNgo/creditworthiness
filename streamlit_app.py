import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
import streamlit as st

@st.cache_resource
def create_and_train_model():
    df = pd.read_csv('https://raw.githubusercontent.com/ThaiDanhNgo/creditworthiness/refs/heads/main/final%20project.csv')
    
    # Tiền xử lý dữ liệu
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    df['state'] = df.address.str[-5:]
    df = df[(df['home_ownership'] != 'ANY') & (df['home_ownership'] != 'OTHER') & (df['home_ownership'] != 'NONE')]
    df['mort_ratio'] = round(df['mort_acc'] / df['open_acc'], 4)
    
    # Loại bỏ ngoại lai theo IQR của boxplot
    df = df[df['loan_amnt'] <= 38000]
    df = df[df['int_rate'] <= 25.83]
    df = df[df['annual_inc'] <= 154500]
    df = df[df['dti'] <= 41.05]
    df = df[df['revol_bal'] <= 40600]
    df = df[df['revol_util'] <= 124.7]
    df = df[df['total_acc'] <= 57]
    df = df[df['mort_ratio'] <= 0.6667]
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    df['issue_d'] = df['issue_d'].dt.year.astype(int)
    selected_columns = [
        'loan_amnt', 'term', 'int_rate', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
        'loan_status', 'purpose', 'dti', 'revol_bal', 'revol_util', 'state', 'mort_ratio']
    new_df = df[selected_columns]
    
    # Mã hóa các biến phân loại
    categorical_columns = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'state']
    encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        new_df[col] = le.fit_transform(new_df[col])
        encoders[col] = le
    
    X = new_df.drop(columns=['loan_status'])
    y = new_df['loan_status']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Xử lý mất cân bằng dữ liệu với SMOTE
    sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, random_state=42, test_size=0.3)
    
    # Huấn luyện mô hình
    et = ExtraTreesClassifier()
    et.fit(X_train, Y_train)
    
    return et, encoders, scaler

# Giao diện người dùng với Streamlit
def main():
    # Tạo và huấn luyện mô hình
    model, encoders, scaler = create_and_train_model()
    
    # Tiêu đề và CSS
    st.markdown("""
        <style>
        .small-text {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 12px;
            color: gray;
        }
        </style>
        <div class="small-text">
        © Copyright: Creditworthiness Project - authorized by Ngo Danh Thai
        </div>
        """, unsafe_allow_html=True)
    
    st.title('Loan Status Prediction')
    
    # Nhập thông tin từ người dùng
    st.header('🎯 Loan Decision Attributes')
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt_value = st.number_input('Loan Amount', min_value=1000)
        verification_status_value = st.selectbox('Verification Status', ['Verified', 'Source Verified', 'Not Verified'])
    with col2:
        term_value = st.selectbox('Term', [' 36 months', ' 60 months'])
        purpose_value = st.selectbox('Purpose', [
            'debt_consolidation', 'credit_card', 'home_improvement', 'other',
            'major_purchase', 'small_business', 'car', 'medical', 'moving',
            'vacation', 'house', 'wedding', 'renewable_energy'
        ])
    with col3:
        int_rate_value = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, step=0.01)
        issue_d_value = st.number_input('Issue Date', min_value=2012, max_value=2024)
    
    st.header('💳 Credit History')
    col4, col5, col6 = st.columns(3)
    with col4:
        revol_bal_value = st.number_input('Revolving Balance', min_value=0.0)
    with col5:
        revol_util_value = st.number_input('Revolving Utilization Rate %', min_value=0.0, step=0.01)
    with col6:
        mort_ratio_value = st.number_input('Mortgage Ratio', min_value=0.0, max_value=1.0, step=0.01)
    
    st.header('📑 Financial Health')
    col7, col8, col9 = st.columns(3)
    with col7:
        annual_inc_value = st.number_input('Annual Income', min_value=0.0)
        state_value = st.selectbox('State', ['70466', '30723', '48052', '22690', '00813', '29597', '05113', '11650', '86630', '93700'])
    with col8:
        emp_length_value = st.selectbox('Employment Length', [
            '< 1 year', '1 year', '2 years', '3 years', '4 years',
            '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
        ])
        dti_value = st.number_input('Debt-to-Income Ratio %', min_value=0.0, step=0.01)
    with col9:
        home_ownership_value = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE'])
    
    # Nút Dự đoán
    if st.button('Predict'):
        # Chuẩn bị dữ liệu đầu vào
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt_value],
            'term': [term_value],
            'int_rate': [int_rate_value],
            'emp_length': [emp_length_value],
            'home_ownership': [home_ownership_value],
            'annual_inc': [annual_inc_value],
            'verification_status': [verification_status_value],
            'issue_d': [issue_d_value],
            'purpose': [purpose_value],
            'dti': [dti_value],
            'revol_bal': [revol_bal_value],
            'revol_util': [revol_util_value],
            'state': [state_value],
            'mort_ratio': [mort_ratio_value]
        })
    
        # Mã hóa các biến phân loại
        categorical_columns = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'state']
        for col in categorical_columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])
    
        # Chuẩn hóa dữ liệu đầu vào
        input_data_scaled = scaler.transform(input_data)
        input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)
    
        # Dự đoán và tính toán xác suất
        prediction = model.predict(input_data_scaled)
        probabilities = model.predict_proba(input_data_scaled)
    
        # Trích xuất xác suất cho mỗi lớp
        prob_fully_paid = probabilities[0][1] * 100  # Xác suất của lớp 'Fully Paid'
        prob_charged_off = probabilities[0][0] * 100  # Xác suất của lớp 'Charged Off'
    
        # Hiển thị dự đoán và xác suất
        if prob_fully_paid > prob_charged_off:
            st.success(f'Predicted Loan Status: {prob_fully_paid:.2f}% Fully Paid')
        else:
            st.error(f'Predicted Loan Status: {prob_charged_off:.2f}% Charged Off')

if __name__ == "__main__":
    main()
