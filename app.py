import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ML na Saúde - Diabetes", layout="wide")

st.markdown("""
<style>
.main { padding: 1rem 2.5rem 1.5rem 2.5rem; }
h1, h2, h3 { margin-bottom: 0.4rem; }
p, label, .stMarkdown, .stDataFrame { font-size: 0.9rem !important; }
section[data-testid="stSidebar"] > div { padding: 0.5rem 1rem; }
[data-testid="stMetricValue"] { font-size: 1.2rem; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; }
[data-testid="stDataFrame"] div { font-size: 0.78rem !important; }
.block-container { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Configurações")
test_size = st.sidebar.slider("Teste (%)", 10, 30, 15, step=5) / 100
random_state = st.sidebar.number_input("Random state", 0, 9999, 42)
n_estimators = st.sidebar.slider("Árvores", 100, 1200, 600, step=100)
max_depth_opt = st.sidebar.selectbox("Profundidade máxima", ("Sem limite", 8, 12, 16, 20), index=0)
max_depth = None if max_depth_opt == "Sem limite" else int(max_depth_opt)
k_clusters = st.sidebar.slider("Clusters (K-Means)", 2, 8, 3)

@st.cache_data
def load_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state,
    n_jobs=-1,
    max_features="sqrt",
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df_clusters = df.copy()
df_clusters["cluster"] = clusters
cluster_means = df_clusters.groupby("cluster").mean(numeric_only=True)
cluster_sizes = df_clusters["cluster"].value_counts().sort_index()

st.markdown("<h1 style='text-align:center;'>Machine Learning na Saúde - Diabetes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Análise exploratória, regressão e agrupamento</p>", unsafe_allow_html=True)
st.divider()

aba1, aba2, aba3, aba4 = st.tabs([
    "Visão geral",
    "EDA",
    "Modelo supervisionado",
    "Modelo não supervisionado"
])

with aba1:
    st.subheader("Resumo do projeto")
    st.write("""
    Aplicação feita em Streamlit utilizando o dataset de diabetes do scikit-learn
    para demonstrar aprendizado de máquina na área da saúde.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes", df.shape[0])
    c2.metric("Atributos", df.shape[1] - 1)
    c3.metric("Alvo", "target")

    st.write("Fluxo:")
    st.markdown(
        "- **EDA**: visualizar dados e correlações\n"
        "- **Supervisionado**: prever progressão da doença (Random Forest)\n"
        "- **Não supervisionado**: agrupar pacientes por semelhança (K-Means)"
    )

    c4, c5 = st.columns(2)
    c4.metric("MSE", f"{mse:.2f}")
    c5.metric("R²", f"{r2:.3f}")

    st.write("Importância das variáveis (atual):")
    st.dataframe(importances.to_frame("importância"), use_container_width=True, height=220)

with aba2:
    st.subheader("1. Exploração de Dados")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes", df.shape[0])
    c2.metric("Atributos", df.shape[1] - 1)
    c3.metric("Alvo", "target")

    with st.expander("Ver dados"):
        st.dataframe(df.head(15), use_container_width=True, height=240)

    with st.expander("Estatísticas"):
        st.dataframe(df.describe().T, use_container_width=True, height=240)

    colA, colB = st.columns([1, 1])
    with colA:
        st.write("Histograma")
        var = st.selectbox("Selecione uma variável:", df.columns, key="eda_var")
        fig, ax = plt.subplots(figsize=(3.0, 2.2))
        ax.hist(df[var], bins=20, color="#6CA6CD", edgecolor="black")
        ax.set_xlabel(var)
        ax.set_ylabel("freq")
        st.pyplot(fig, use_container_width=False)

    with colB:
        st.write("Correlação")
        corr = df.corr(numeric_only=True)
        st.dataframe(
            corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
            use_container_width=True,
            height=250,
        )

with aba3:
    st.subheader("2. Modelo supervisionado – Random Forest")

    c1, c2 = st.columns(2)
    c1.metric("MSE", f"{mse:.2f}")
    c2.metric("R²", f"{r2:.3f}")

    st.write("Real x Previsto:")
    left, mid, right = st.columns([0.35, 0.3, 0.35])
    with mid:
        fig2, ax2 = plt.subplots(figsize=(2.8, 2.2))
        ax2.scatter(y_test, y_pred, alpha=0.5, color="#87CEFA", s=18)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1)
        ax2.set_xlabel("Real")
        ax2.set_ylabel("Previsto")
        st.pyplot(fig2, use_container_width=True)

    st.write("Importância das variáveis:")
    st.bar_chart(importances, height=210)

with aba4:
    st.subheader("3. Modelo não supervisionado – K-Means")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Tamanho dos clusters")
        st.table(cluster_sizes.rename("Pacientes"))
    with c2:
        st.write("Média das variáveis por cluster")
        st.dataframe(
            cluster_means.style.background_gradient(cmap="RdYlGn_r"),
            use_container_width=True,
            height=220,
        )

    st.write("Visualização 2D dos clusters")
    feat_x = st.selectbox("Eixo X", X.columns, index=2, key="kmeans_x")
    feat_y = st.selectbox("Eixo Y", X.columns, index=4, key="kmeans_y")

    l2, m2, r2c = st.columns([0.35, 0.3, 0.35])
    with m2:
        fig3, ax3 = plt.subplots(figsize=(2.8, 2.2))
        scatter = ax3.scatter(
            df_clusters[feat_x],
            df_clusters[feat_y],
            c=df_clusters["cluster"],
            cmap="tab10",
            alpha=0.6,
            s=20,
        )
        ax3.set_xlabel(feat_x)
        ax3.set_ylabel(feat_y)
        plt.colorbar(scatter, ax=ax3, label="Cluster", fraction=0.04, pad=0.03)
        st.pyplot(fig3, use_container_width=True)

with st.expander("🧠 Informações adicionais"):
    st.markdown("### Interpretações técnicas")
    st.markdown("#### Variáveis do dataset")
    st.write("""
    - **age** – Idade padronizada do paciente.
    - **sex** – Sexo (0 ou 1), também padronizado.
    - **bmi** – Índice de Massa Corporal (IMC), associado à obesidade.
    - **bp** – Pressão arterial média.
    - **s1** – Colesterol total (lipídios totais no sangue).
    - **s2** – LDL/VLDL (colesterol “ruim”).
    - **s3** – HDL (colesterol “bom”).
    - **s4** – Triglicerídeos (gordura no sangue).
    - **s5** – Glicose plasmática.
    - **s6** – Índice metabólico complementar.
    - **target** – Progressão da doença (alvo da previsão).
    """)

    st.markdown("#### O que cada gráfico representa")
    st.write("""
    - **Histograma:** distribuição dos valores de cada variável.
    - **Correlação:** força da relação entre variáveis.
    - **Real x Previsto:** qualidade da previsão do modelo.
    - **Importância das variáveis:** peso de cada variável no modelo Random Forest.
    - **Clusters (K-Means):** grupos de pacientes com perfis semelhantes.
    """)

    st.markdown("---")
    st.markdown("### Interpretações práticas")
    st.write("""
    - Pacientes com **IMC (bmi)** e **pressão arterial (bp)** altos apresentam **maior risco de progressão** da doença.
    - A variável **s5 (glicose plasmática)** confirma o impacto direto do controle glicêmico no diabetes.
    - O K-Means permite separar **perfis clínicos distintos**:
        - Cluster com altos valores de **bmi** e **s5** → grupo de **alto risco metabólico**.
        - Cluster com valores equilibrados → grupo **de controle ou baixo risco**.
    - Aplicações práticas:
        - Apoio à decisão médica.
        - Monitoramento de pacientes com risco metabólico.
        - Estudos populacionais e prevenção em saúde pública.
    """)
