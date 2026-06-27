import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(
    page_title="Iris Dashboard - Ciencia de Datos",
    layout="wide"
)

st.title("🌸 Dashboard de Ciencia de Datos - Iris Dataset")
st.caption("EDA + Preprocesamiento + OLAP + Machine Learning con Streamlit")


@st.cache_data
def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    df["species"] = df["target"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    })

    return df, iris


df, iris = load_data()

feature_cols = iris.feature_names
species_list = ["setosa", "versicolor", "virginica"]


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙️ Controles")

show_raw = st.sidebar.checkbox(
    "Mostrar dataset completo filtrado",
    value=False
)

test_size = st.sidebar.slider(
    "Porcentaje de prueba",
    0.10,
    0.50,
    0.30,
    0.05
)

do_scaling = st.sidebar.checkbox(
    "Aplicar StandardScaler",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtro por especie")

species_filter = st.sidebar.multiselect(
    "Selecciona una o varias especies",
    options=species_list,
    default=species_list
)

df_f = df[df["species"].isin(species_filter)].copy()


# -------------------------------------------------
# UNIDAD 1
# -------------------------------------------------
with st.expander("✅ Unidad 1. Introducción", expanded=True):
    st.write("""
    Este dashboard integra el flujo básico de un proyecto de Ciencia de Datos:

    1. Comprensión del problema.
    2. Análisis exploratorio de datos.
    3. Preprocesamiento.
    4. Simulación OLAP.
    5. Modelado.
    6. Evaluación.
    """)

    st.write("📌 Problema: clasificar flores Iris en tres especies.")
    st.write(f"🔢 Dataset filtrado: **{df_f.shape[0]} filas × {df_f.shape[1]} columnas**")
    st.write("🎯 Variable objetivo: `species`")


# -------------------------------------------------
# UNIDAD 2 - EDA
# -------------------------------------------------
st.header("📌 Unidad 2. Análisis Exploratorio de Datos")

colA, colB = st.columns(2)

with colA:
    st.subheader("2.1 Vista previa")
    st.dataframe(df_f.head(10), use_container_width=True)

with colB:
    st.subheader("2.2 Tipos de datos")
    st.code(df_f.dtypes.to_string())

st.subheader("2.3 Estadística descriptiva")
st.dataframe(df_f[feature_cols].describe(), use_container_width=True)


st.subheader("2.4 Distribución de especies")

fig_class = px.histogram(
    df_f,
    x="species",
    color="species",
    title="Distribución de especies"
)

st.plotly_chart(fig_class, use_container_width=True)


st.subheader("2.5 Distribución por variable")

feature_dist = st.selectbox(
    "Selecciona una variable",
    feature_cols
)

nbins = st.slider(
    "Número de bins",
    5,
    50,
    20
)

left, right = st.columns(2)

with left:
    fig_hist = px.histogram(
        df_f,
        x=feature_dist,
        color="species",
        nbins=nbins,
        opacity=0.75,
        facet_col="species",
        title=f"Histograma de {feature_dist}"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    fig_box = px.box(
        df_f,
        x=feature_dist,
        y="species",
        color="species",
        points="all",
        title=f"Boxplot de {feature_dist}"
    )

    st.plotly_chart(fig_box, use_container_width=True)


st.subheader("2.6 Relación entre variables")

c1, c2 = st.columns(2)

feature_x = c1.selectbox(
    "Variable X",
    feature_cols,
    index=0
)

feature_y = c2.selectbox(
    "Variable Y",
    feature_cols,
    index=2
)

fig_scatter = px.scatter(
    df_f,
    x=feature_x,
    y=feature_y,
    color="species",
    title=f"{feature_x} vs {feature_y}"
)

st.plotly_chart(fig_scatter, use_container_width=True)


st.subheader("2.7 Matriz de correlación")

corr = df_f[feature_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Matriz de correlación"
)

st.plotly_chart(fig_corr, use_container_width=True)


# -------------------------------------------------
# UNIDAD 3 - PREPROCESAMIENTO
# -------------------------------------------------
st.header("📌 Unidad 3. Preprocesamiento")

col1, col2 = st.columns(2)

with col1:
    st.subheader("3.1 Revisión de valores nulos")

    st.dataframe(
        df_f.isnull().sum().to_frame("nulos"),
        use_container_width=True
    )

with col2:
    st.subheader("3.2 Selección de variables")

    selected_features = st.multiselect(
        "Variables independientes",
        feature_cols,
        default=feature_cols
    )


X = df_f[selected_features].copy()
y = df_f["species"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

if do_scaling:
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_test_proc = scaler.transform(X_test)

    st.success("✅ StandardScaler aplicado correctamente.")
else:
    X_train_proc = X_train.values
    X_test_proc = X_test.values

    st.warning("⚠️ El modelo se entrenará sin escalamiento.")


# -------------------------------------------------
# UNIDAD 4 - OLAP
# -------------------------------------------------
st.header("📌 Unidad 4. Simulación OLAP")

st.write("""
En esta sección se simula un análisis tipo OLAP usando agregaciones por especie.
Esto permite observar medidas como promedio, desviación estándar, mínimo y máximo.
""")

olap_feature = st.selectbox(
    "Variable para resumen OLAP",
    selected_features
)

rollup = df_f.groupby("species")[olap_feature].agg(
    ["mean", "std", "min", "max"]
).reset_index()

st.dataframe(rollup, use_container_width=True)

fig_olap = px.bar(
    rollup,
    x="species",
    y="mean",
    error_y="std",
    title=f"Promedio de {olap_feature} por especie"
)

st.plotly_chart(fig_olap, use_container_width=True)


# -------------------------------------------------
# UNIDAD 5 - MODELADO
# -------------------------------------------------
st.header("📌 Unidad 5. Modelado y Evaluación")

st.subheader("5.1 Modelo: Logistic Regression")

model = LogisticRegression(max_iter=5000)

model.fit(X_train_proc, y_train)

y_pred = model.predict(X_test_proc)

acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.3f}")


st.subheader("5.2 Matriz de confusión")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=species_list
)

fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=species_list,
    y=species_list,
    colorscale="Blues"
)

fig_cm.update_layout(
    title="Matriz de Confusión",
    xaxis_title="Predicción",
    yaxis_title="Clase real"
)

st.plotly_chart(fig_cm, use_container_width=True)


st.subheader("5.3 Reporte de clasificación")

st.code(
    classification_report(y_test, y_pred)
)


# -------------------------------------------------
# UNIDAD 6 - EXPERIMENTACIÓN
# -------------------------------------------------
st.header("📌 Unidad 6. Experimentación")

st.write("""
Prueba diferentes combinaciones de variables desde la sección de preprocesamiento
y observa cómo cambia el accuracy del modelo.
""")

st.markdown("""
Ejemplos de experimentos:

1. Usar solo `petal length (cm)`.
2. Usar `petal length (cm)` y `petal width (cm)`.
3. Usar todas las variables.
""")


# -------------------------------------------------
# DATASET COMPLETO
# -------------------------------------------------
if show_raw:
    st.header("📄 Dataset completo filtrado")
    st.dataframe(df_f, use_container_width=True)
