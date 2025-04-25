import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# بارگذاری مدل‌ها و ابزارها
model = joblib.load('diabetes_model_compressed.pkl')
scaler = joblib.load('scaler_msc.joblib')
le_gender = joblib.load('le_gender.joblib')
model_columns = joblib.load('model_columns.joblib')

# تنظیمات صفحه
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title('Diabetes Prediction App')
st.subheader("Check your diabetes risk based on health parameters.")



# فرم ورودی
st.header('Fill the form below')

gender = st.radio("Gender:", ['Male', 'Female'])
age = st.slider("Age:", 1, 100, 30)
bmi = st.number_input("BMI:", 10.0, 50.0, step=0.1, value=25.0)
hba1c = st.number_input("HbA1c Level:", 3.0, 15.0, step=0.1, value=5.5)
glucose = st.slider("Blood Glucose Level:", 50, 300, 120)
smoking = st.selectbox("Smoking History:", ['never', 'former', 'current', 'not current', 'ever', 'No Info'])

# تابع پیش‌بینی
def predict():
    # ساخت دیتافریم اولیه
    X = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        'smoking_history': smoking
    }])

    # پردازش gender
    try:
        X['gender'] = le_gender.transform([gender])[0]
    except:
        st.error("⚠️ Gender value not recognized.")
        return
        
    def simplify_smoking(x):
        if x in ['former', 'ever', 'not current']:
            return 'past'
        elif x == 'never':
            return 'never'
        elif x == 'current':
            return 'current'
        else:
            return 'unknown'

    X['smoking_history'] = X['smoking_history'].apply(simplify_smoking)

    # one-hot برای smoking_history
    X = pd.get_dummies(X, columns=['smoking_history'], drop_first=False)

    # اضافه کردن ستون‌های غایب
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[model_columns]

    # استانداردسازی ویژگی‌ها
    cols_to_scale = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    # پیش‌بینی
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    # نمایش نتیجه
    if prediction == 0:
        st.success(f"✅ You are healthy! (Probability: {proba[0]:.2f})")
    else:
        st.error(f"⚠️ You may be diabetic. (Probability: {proba[1]:.2f})")

    # نمودار احتمال
    fig, ax = plt.subplots()
    ax.bar(['Healthy', 'Diabetic'], proba, color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probability')
    st.pyplot(fig)

# دکمه پیش‌بینی
if st.button("Predict"):
    predict()

# پانوشت
st.caption("App created by MJ Shadfar - [GitHub](https://github.com/mj72git/Diabete-Prediction)")
# نمایش عکس
try:
    st.image("1.jpg", width=100)
except:
    st.warning("⚠️ Could not load image.")