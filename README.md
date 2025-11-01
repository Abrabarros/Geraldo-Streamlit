# Machine Learning na Saúde - Diabetes

Aplicação desenvolvida em **Python** com **Streamlit**, para demonstrar o uso de **aprendizado de máquina supervisionado e não supervisionado** na área da saúde.

---

##  Objetivo

O projeto tem como objetivo aplicar **técnicas de Machine Learning** para prever e analisar a **progressão da diabetes** a partir de dados clínicos.

O trabalho foi dividido em três partes:
1. **Exploração de Dados (EDA)**  
2. **Modelo Supervisionado (Random Forest Regressor)**  
3. **Modelo Não Supervisionado (K-Means)**

---

##  Dataset

O dataset utilizado é o `load_diabetes` do **scikit-learn**.  
Ele contém **442 pacientes** e **10 variáveis clínicas** padronizadas.

Atributos principais:
- `age`: idade (padronizada)  
- `sex`: sexo (codificado)  
- `bmi`: índice de massa corporal (IMC)  
- `bp`: pressão arterial média  
- `s1` a `s6`: medidas laboratoriais  
- `target`: progressão da doença (variável alvo)

---

##  Tecnologias

- Python 3.10+
- Streamlit
- Pandas / NumPy
- Matplotlib
- Scikit-learn

---

##  Como executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o app:
   ```bash
   streamlit run app.py
   ```

3. O Streamlit abrirá no navegador:
   ```
   http://localhost:8501
   ```

---

##  Estrutura do app

### 1. EDA
- Mostra amostra dos dados (`head()`)
- Estatísticas descritivas (`describe()`)
- Histograma de variáveis
- Matriz de correlação

### 2. Random Forest
- Divide dados em treino/teste
- Treina o modelo `RandomForestRegressor`
- Exibe MSE e R²
- Mostra importância das variáveis
- Gráfico de valores reais x previstos

### 3. K-Means
- Agrupa pacientes por semelhança de atributos
- Exibe tamanho e médias de cada grupo
- Gráfico 2D interativo para visualizar os clusters

---

##  Resultados

- O modelo de regressão obteve valores típicos de **R² ≈ 0.45–0.60**, coerentes com o dataset padronizado.
- O K-Means separou pacientes em grupos com diferentes perfis clínicos, destacando padrões de risco.

---

##  Estrutura de arquivos

```
 ml-diabetes
├── app.py               # código principal
├── requirements.txt     # dependências do projeto
├── README.md            # este documento
```

---

##  Autor

**Abraão Saraiva**  
Curso: Machine Learning 
