"""Interactive exploration of the benchmark (not a clinical application)."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title='Explainable ML Research Demo', layout='wide')
st.title('Explainable ML: Breast Cancer Benchmark')
st.warning('Educational research demonstration only. Not a medical device or diagnostic tool.')
model_path = Path('artifacts/model.joblib')
if not model_path.exists():
    st.info('First run: python -m src.train')
    st.stop()
loaded = joblib.load(model_path)  # Only load files you generated and trust.
data = load_breast_cancer(as_frame=True)
X = data.data
st.write('Explore predictions for existing benchmark samples. No personal medical information is collected.')
index = st.slider('Benchmark sample index', 0, len(X) - 1, 0)
row = X.iloc[[index]][loaded['features']]
prob = float(loaded['model'].predict_proba(row)[0, 1])
st.metric('Model-estimated malignant class probability', f'{prob:.1%}')
st.caption('This probability is model output for a dataset example, not a calibrated individual risk estimate.')
st.dataframe(row.T.rename(columns={index: 'Value'}), use_container_width=True)
importance_path = Path('artifacts/feature_importance.csv')
if importance_path.exists():
    st.subheader('Global permutation importance')
    st.bar_chart(pd.read_csv(importance_path).head(12).set_index('feature')['importance_mean'])
