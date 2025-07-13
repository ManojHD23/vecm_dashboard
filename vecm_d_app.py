import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
import matplotlib.pyplot as plt

st.set_page_config(page_title="Milk Societies VECM Dashboard", layout="wide")
st.title("📈 Milk Production: VECM Diagnostic Dashboard")

# Load Data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    pivot_df = df.pivot(index='Date', columns='Name of the Society', values='Milk_Production')
    return pivot_df.dropna()

# Constants
FILE_PATH = r"C:\Users\Manoj\Desktop\waste_folder\data\milk_timeseries_cleaned.csv"
data = load_data(FILE_PATH)
societies = sorted(data.columns)

# Sidebar Inputs
st.sidebar.header("🔧 Model Configuration")
group_choice = st.sidebar.selectbox("Select Group", ("Group 1 (Society 1–10)", "Group 2 (Society 11–20)"))
k_ar_diff = st.sidebar.slider("Select Lag Differences (k_ar_diff)", 1, 3, 1)
coint_rank = st.sidebar.slider("Select Cointegration Rank", 1, 5, 3)

# Group selection based on full match
if group_choice == "Group 1 (Society 1–10)":
    group_cols = societies[:10]
else:
    group_cols = societies[10:]

group_df = data[group_cols]

# Normalize
group_scaled = pd.DataFrame(StandardScaler().fit_transform(group_df),
                            index=group_df.index,
                            columns=group_df.columns)

# Fit VECM
model = VECM(group_scaled, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic='ci')
vecm_result = model.fit()
st.success(f"✅ VECM model fitted for {group_choice}")

# Display Cointegration Summary
st.subheader("📊 Cointegration Vectors (Beta)")
beta_df = pd.DataFrame(vecm_result.beta.T, columns=group_cols)
st.dataframe(beta_df.style.format("{:.4f}"))

# Error Correction Terms
st.subheader("🔄 Error Correction Terms (Alpha)")
alpha_df = pd.DataFrame(vecm_result.alpha, columns=[f"EC{i+1}" for i in range(coint_rank)], index=group_cols)
st.dataframe(alpha_df.style.background_gradient(cmap="coolwarm").format("{:.3f}"))

# Influential Societies (based on |alpha|)
st.subheader("🌟 Influential Societies")
influence = alpha_df.abs().sum(axis=1).sort_values(ascending=False)
st.bar_chart(influence, use_container_width=True)

# Stability Check
st.subheader("🧮 Stability Check")
try:
    eigvals = np.linalg.eigvals(vecm_result.coefs.T @ vecm_result.coefs)
    modulus = np.abs(eigvals)
    is_stable = all(modulus < 1)
    st.write("**Eigenvalue Moduli:**", modulus.round(4).tolist())
    st.markdown(f"✅ **Stable**" if is_stable else "❌ **Not Stable**")
except:
    st.warning("Could not compute eigenvalues. Model might be degenerate.")

# Residual Diagnostics
st.subheader("📉 Residual Diagnostics")
acorr_results = {}
normality_results = {}
for i, col in enumerate(group_cols):
    resid = vecm_result.resid[:, i]
    acorr_pval = acorr_ljungbox(resid, lags=[1], return_df=True)['lb_pvalue'].iloc[0]
    normality_pval = normal_ad(resid)[1]
    acorr_results[col] = acorr_pval
    normality_results[col] = normality_pval

residual_df = pd.DataFrame({
    "Autocorr p-value": acorr_results,
    "Normality (AD) p-value": normality_results
}).sort_index()
st.dataframe(residual_df.style.format("{:.4f}"))

# Download CSV Report
st.subheader("⬇️ Download Summary")
st.download_button("Download Summary CSV", residual_df.to_csv().encode('utf-8'), "vecm_diagnostics.csv")

# Footer
st.markdown("---")
st.markdown("📍 Made by **Manoj H D** | VECM Insight Dashboard")
