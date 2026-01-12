# app.py
import sys
!{sys.executable} -m pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

st.set_page_config(page_title="Análisis Estadístico de Ventas - Streamlit", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_data(file_path):
    # This function is modified to load from a given path directly
    df = pd.read_csv(file_path, parse_dates=["fecha"], infer_datetime_format=True)
    return df

def resumen_descriptivo(df):
    numeric = df.select_dtypes(include=[np.number])
    desc = numeric.describe().T
    return desc

def agregar_columnas_temporales(df):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["fecha_date"] = df["fecha"].dt.date
    df["hora"] = df["fecha"].dt.hour
    # Removed locale='es_ES' to avoid 'unsupported locale setting' error
    df["dia_semana"] = df["fecha"].dt.day_name()
    return df

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, nbins=30, title=f"Histograma de {column}")
    return fig

def plot_box(df, column, by=None):
    if by:
        fig = px.box(df, x=by, y=column, points="outliers", title=f"Boxplot de {column} por {by}")
    else:
        fig = px.box(df, y=column, points="outliers", title=f"Boxplot de {column}")
    return fig

def correlation_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    return fig

# ---------------------------
# UI: Sidebar - carga y filtros
# ---------------------------
st.sidebar.title("Carga y filtros")
# Directly load the DataFrame from the generated CSV path
# The st.file_uploader and df = load_data(uploaded_file) are commented out/modified
# to ensure df is always loaded when running in Colab without interactive UI
df = load_data("/content/drive/MyDrive/5to_ciclo/IDL1_data_analisis_de_la_informacion/Demo_ventas_tienda_conveniencia.csv")

st.sidebar.markdown("---")
st.sidebar.write("Si no subes archivo, el app mostrará instrucciones y ejemplos.")

# ---------------------------
# Main layout
# ---------------------------
st.title("Análisis de información con métodos estadísticos")
st.markdown("Este app demuestra técnicas descriptivas, inferenciales y predictivas aplicadas al dataset de ventas por hora. "
            "Carga tu archivo CSV con las columnas: **fecha, tienda, producto, turno, unidades_vendidas, precio_unitario, venta_total**.")

# Remove st.stop() since df will always be loaded now
# if df is None:
#     st.info("Sube el archivo CSV en la barra lateral para comenzar. Mientras tanto, aquí tienes una guía rápida de lo que hace la app.")
#     st.markdown("""
#     **Secciones del app**
#     - Análisis descriptivo: resumen numérico, histogramas, boxplots.
#     - Análisis inferencial: correlaciones, pruebas t/ANOVA, regresión simple.
#     - Forecasting simple: promedio móvil y regresión por tiempo.
#     """)
#     st.stop()

# ---------------------------
# Preprocesamiento
# ---------------------------
df = agregar_columnas_temporales(df)
st.sidebar.write("Rango de fechas detectado:")
st.sidebar.write(f"Desde **{df['fecha'].min()}** hasta **{df['fecha'].max()}**")

# Filtros interactivos
with st.sidebar.expander("Filtros interactivos"):
    tiendas = st.multiselect("Selecciona tiendas", options=sorted(df["tienda"].unique()), default=sorted(df["tienda"].unique()))
    productos = st.multiselect("Selecciona productos", options=sorted(df["producto"].unique()), default=sorted(df["producto"].unique()))
    fecha_min = st.date_input("Fecha inicio", value=df["fecha"].dt.date.min())
    fecha_max = st.date_input("Fecha fin", value=df["fecha"].dt.date.max())

# Aplicar filtros
mask = (df["tienda"].isin(tiendas)) & (df["producto"].isin(productos)) & (df["fecha"].dt.date >= fecha_min) & (df["fecha"].dt.date <= fecha_max)
df_f = df.loc[mask].copy()

st.header("1. Datos cargados y limpieza básica")
st.write("Número de registros después de aplicar filtros:", len(df_f))
st.dataframe(df_f.head(50))

# ---------------------------
# 2. Análisis descriptivo
# ---------------------------
st.markdown("---")
st.header("2. Análisis descriptivo")
st.markdown("**Objetivo:** resumir la información con medidas de tendencia central y dispersión. Estas medidas ayudan a entender el comportamiento típico y la variabilidad de las ventas.")

# Resumen numérico
st.subheader("Resumen numérico")
desc = resumen_descriptivo(df_f)
st.table(desc)

# Visualizaciones descriptivas
col1, col2 = st.columns(2)
with col1:
    st.subheader("Histograma de unidades vendidas")
    fig_hist = plot_histogram(df_f, "unidades_vendidas")
    st.plotly_chart(fig_hist, use_container_width=True)
with col2:
    st.subheader("Boxplot de venta_total por producto")
    fig_box = plot_box(df_f, "venta_total", by="producto")
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Distribución por turno")
turno_counts = df_f.groupby("turno")["venta_total"].sum().reset_index().sort_values("venta_total", ascending=False)
fig_turno = px.bar(turno_counts, x="turno", y="venta_total", title="Venta total por turno")
st.plotly_chart(fig_turno, use_container_width=True)

# ---------------------------
# 3. Análisis inferencial
# ---------------------------
st.markdown("---")
st.header("3. Análisis inferencial")
st.markdown("**Objetivo:** evaluar relaciones entre variables y contrastar hipótesis. Aquí aplicamos correlación, pruebas de hipótesis y regresión simple.")

# Correlación
st.subheader("Correlación entre variables numéricas")
fig_corr = correlation_heatmap(df_f)
st.pyplot(fig_corr)

# Scatter precio vs unidades
st.subheader("Relación precio_unitario vs unidades_vendidas")
fig_scatter = px.scatter(df_f, x="precio_unitario", y="unidades_vendidas", color="producto", trendline="ols",
                         title="Scatter precio_unitario vs unidades_vendidas con línea de regresión")
st.plotly_chart(fig_scatter, use_container_width=True)

# Prueba t o ANOVA
st.subheader("Comparación de medias entre grupos")
st.markdown("Selecciona una variable categórica para comparar la **venta_total** entre sus grupos usando ANOVA (si >2 grupos) o t-test (2 grupos).")
group_var = st.selectbox("Variable categórica", options=["tienda", "producto", "turno", "dia_semana"])
groups = df_f[group_var].unique()

if len(groups) == 2:
    g1 = df_f[df_f[group_var] == groups[0]]["venta_total"]
    g2 = df_f[df_f[group_var] == groups[1]]["venta_total"]
    tstat, pval = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
    st.write(f"t-statistic: **{tstat:.3f}**, p-value: **{pval:.4f}**")
    st.write("Interpretación: si p-value < 0.05, hay evidencia de diferencia significativa entre medias.")
else:
    # ANOVA
    samples = [df_f[df_f[group_var] == g]["venta_total"].dropna() for g in groups]
    fstat, pval = stats.f_oneway(*samples)
    st.write(f"ANOVA F-statistic: **{fstat:.3f}**, p-value: **{pval:.4f}**")
    st.write("Interpretación: si p-value < 0.05, al menos un grupo difiere significativamente en la media de venta_total.")

# Regresión lineal simple (ejemplo)
st.subheader("Regresión lineal simple: venta_total ~ unidades_vendidas")
X = df_f[["unidades_vendidas"]].fillna(0)
y = df_f["venta_total"].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
st.write("Coeficiente (slope):", float(reg.coef_[0]))
st.write("Intercept:", float(reg.intercept_))
st.write("R2 en test:", float(r2_score(y_test, y_pred)))
st.write("RMSE en test:", float(np.sqrt(mean_squared_error(y_test, y_pred))))

fig_reg = px.scatter(x=X_test["unidades_vendidas"], y=y_test, labels={"x":"unidades_vendidas", "y":"venta_total"},
                     title="Regresión lineal simple: venta_total vs unidades_vendidas")
fig_reg.add_traces(px.line(x=X_test["unidades_vendidas"].sort_values(), y=np.sort(y_pred)).data)
st.plotly_chart(fig_reg, use_container_width=True)

# ---------------------------
# 4. Análisis temporal y forecasting simple
# ---------------------------
st.markdown("---")
st.header("4. Análisis temporal y forecasting simple")
st.markdown("**Objetivo:** mostrar técnicas básicas de series temporales: agregación, promedio móvil y un forecast simple por regresión temporal.")

# Agregar serie temporal agregada por hora o día
freq = st.selectbox("Agregación temporal", options=["H", "D"], index=1, help="H = hora, D = día")
if freq == "H":
    ts = df_f.set_index("fecha").resample("H")["venta_total"].sum().reset_index()
else:
    ts = df_f.set_index("fecha").resample("D")["venta_total"].sum().reset_index()

st.subheader("Serie temporal agregada")
fig_ts = px.line(ts, x="fecha", y="venta_total", title="Serie temporal de venta_total")
st.plotly_chart(fig_ts, use_container_width=True)

# Promedio móvil
window = st.slider("Ventana promedio móvil (periodos)", min_value=2, max_value=30, value=7)
ts["ma"] = ts["venta_total"].rolling(window=window, min_periods=1).mean()
fig_ma = px.line(ts, x="fecha", y=["venta_total", "ma"], labels={"value":"venta_total / ma"})
fig_ma.update_layout(title=f"Venta total y promedio móvil (window={window})")
st.plotly_chart(fig_ma, use_container_width=True)

# Forecasting simple por regresión con tendencia y estacionalidad horaria (si freq=H)
st.subheader("Forecasting simple por regresión temporal")
st.markdown("Se crea una regresión lineal usando índice temporal y componentes cíclicos (hora del día) como features. Es un ejemplo pedagógico, no un modelo de producción.")

# Preparar features
ts_model = ts.copy()
ts_model = ts_model.dropna().reset_index(drop=True)
ts_model["t"] = np.arange(len(ts_model))
if freq == "H":
    ts_model["hour"] = ts_model["fecha"].dt.hour
    # codificar ciclo horario con sen/cos
    ts_model["hour_sin"] = np.sin(2 * np.pi * ts_model["hour"] / 24)
    ts_model["hour_cos"] = np.cos(2 * np.pi * ts_model["hour"] / 24)
    features = ["t", "hour_sin", "hour_cos"]
else:
    ts_model["dayofweek"] = ts_model["fecha"].dt.dayofweek
    ts_model["dow_sin"] = np.sin(2 * np.pi * ts_model["dayofweek"] / 7)
    ts_model["dow_cos"] = np.cos(2 * np.pi * ts_model["dayofweek"] / 7)
    features = ["t", "dow_sin", "dow_cos"]

X = ts_model[features]
y = ts_model["venta_total"]
model_time = LinearRegression().fit(X, y)

# Predecir horizon
horizon = st.number_input("Horizonte de forecast (periodos)", min_value=1, max_value=168, value=24)
last_t = ts_model["t"].iloc[-1]
future_pred_data = [] # Renamed to avoid conflict with global 'future' from Prophet
for i in range(1, horizon + 1):
    t_i = last_t + i
    if freq == "H":
        # calcular hora futura aproximada
        last_date = ts_model["fecha"].iloc[-1]
        future_date = last_date + pd.Timedelta(hours=i)
        hour = future_date.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        future_pred_data.append({"fecha": future_date, "t": t_i, "hour_sin": hour_sin, "hour_cos": hour_cos})
    else:
        last_date = ts_model["fecha"].iloc[-1]
        future_date = last_date + pd.Timedelta(days=i)
        dow = future_date.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        future_pred_data.append({"fecha": future_date, "t": t_i, "dow_sin": dow_sin, "dow_cos": dow_cos})

df_future_pred = pd.DataFrame(future_pred_data)
X_future = df_future_pred[features]
y_future_pred = model_time.predict(X_future)

# Mostrar forecast
df_forecast = pd.DataFrame({"fecha": df_future_pred["fecha"], "forecast": y_future_pred})
fig_forecast = px.line(pd.concat([ts[["fecha", "venta_total"]].rename(columns={"venta_total":"value"}),
                                 df_forecast.rename(columns={"forecast":"value"})]),
                       x="fecha", y="value", color_discrete_sequence=["#636EFA"])
# plot original and forecast separately for clarity
fig = px.line(ts, x="fecha", y="venta_total", title="Serie y Forecast")
fig.add_scatter(x=df_forecast["fecha"], y=df_forecast["forecast"], mode="lines", name="Forecast")
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Nota pedagógica:** este forecast usa regresión lineal con componentes cíclicos. Para producción se recomiendan modelos especializados (Prophet, SARIMA, modelos ML).")

# ---------------------------
# 5. Conclusiones y export
# ---------------------------
st.markdown("---")
st.header("5. Conclusiones y siguientes pasos")
st.markdown("""
**Qué aprendimos**
- El análisis descriptivo resume la información y detecta outliers y patrones.
- El análisis inferencial permite contrastar hipótesis y cuantificar relaciones.
- El análisis temporal y forecasting requiere agregar series y modelar tendencia y estacionalidad.
""")

st.subheader("Siguientes pasos recomendados")
st.write("- Validar calidad de datos (missing, duplicados, precios atípicos).")
st.write("- Probar modelos de series temporales especializados (Prophet, SARIMA).")
st.write("- Evaluar modelos de ML para predicción por tienda/producto.")
st.write("- Construir un dashboard con filtros y alertas para operaciones.")

st.info("Puedes descargar el dataset filtrado para seguir trabajando localmente.")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Descargar datos filtrados (CSV)", data=csv, file_name="ventas_filtradas.csv", mime="text/csv")
