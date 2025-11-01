import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# config
st.set_page_config(page_title="Machine Learning na Saúde - Diabetes", layout="wide")
st.title("Machine Learning na Saúde - Diabetes")

# dataset
@st.cache_data
def load_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()

# sidebar
st.sidebar.header("Configurações")
test_size = st.sidebar.slider("Teste (%)", 10, 30, 15, step=5) / 100
random_state = st.sidebar.number_input("Random state", 0, 9999, 42)
n_estimators = st.sidebar.slider("Árvores", 200, 1200, 600, step=100)
max_depth_opt = st.sidebar.selectbox("Profundidade máxima", ("Sem limite", 8, 12, 16, 20), index=0)
max_depth = None if max_depth_opt == "Sem limite" else int(max_depth_opt)
k_clusters = st.sidebar.slider("Clusters", 2, 8, 3, step=1)

# eda
st.subheader("1. Exploração de Dados")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Pacientes", df.shape[0])
col2.metric("Atributos", df.shape[1] - 1)
col3.metric("Alvo", "target")

with st.expander("Estatísticas"):
    st.write(df.describe().T)

st.write("Histograma")
var_to_plot = st.selectbox("Variável:", df.columns)
fig, ax = plt.subplots()
ax.hist(df[var_to_plot], bins=20, color="skyblue", edgecolor="black")
ax.set_title(f"{var_to_plot}")
st.pyplot(fig)

with st.expander("Correlação"):
    corr = df.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))

# random forest
st.subheader("2. Random Forest")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state,
    n_jobs=-1,
    max_features="sqrt"
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

colA, colB = st.columns(2)
colA.metric("MSE", f"{mse:.2f}")
colB.metric("R²", f"{r2:.3f}")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.7)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax2.set_xlabel("Real")
ax2.set_ylabel("Previsto")
st.pyplot(fig2)

st.write("Importância das variáveis")
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances)

# k-means
st.subheader("3. K-Means")

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df_clusters = df.copy()
df_clusters["cluster"] = clusters

st.write("Tamanho dos clusters")
st.table(df_clusters["cluster"].value_counts().sort_index().rename("Pacientes"))

st.write("Médias por cluster")
cluster_means = df_clusters.groupby("cluster").mean(numeric_only=True)
st.dataframe(cluster_means.style.background_gradient(cmap="RdYlGn_r"))

st.write("Gráfico dos clusters")
feat_x = st.selectbox("Eixo X:", X.columns, index=2)
feat_y = st.selectbox("Eixo Y:", X.columns, index=4)
fig4, ax4 = plt.subplots()
scatter = ax4.scatter(df_clusters[feat_x], df_clusters[feat_y],
                      c=df_clusters["cluster"], cmap="tab10", alpha=0.7)
ax4.set_xlabel(feat_x)
ax4.set_ylabel(feat_y)
plt.colorbar(scatter, ax=ax4, label="Cluster")
st.pyplot(fig4)
