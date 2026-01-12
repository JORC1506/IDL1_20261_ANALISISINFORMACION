import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Visualización de Ventas - Tiendas de Conveniencia")

#cargar el archivo
archivo = st.file_uploader("Sube el archivo CSV", type=["csv"])

"""
if archivo is not None:
    df = pd.read_csv(archivo)
    st.subheader("Vista previa de datos")
    st.dataframe(df)

    # validación básica
    columnas_necesarias = {'producto', 'turno', 'tienda', 'venta_total'}
    if not columnas_necesarias.issubset(df.columns):
        st.error("El archivo CSV debe contener las columnas: " + str(columnas_necesarias))
    else:
        #ventas de productos
        st.subheader("Ventas total por tipo de productos")
        ventas_producto = df.groupby('producto')['venta_total'].sum()

        fig1 = plt.figure()
        ventas_producto.plot(kind='bar', x='producto', y='venta_total')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        #ventas por turno
        st.subheader("Ventas total por turno")
        ventas_turno = df.groupby('turno')['venta_total'].sum()

        fig2 = plt.figure()
        ventas_turno.plot(kind='bar', x='turno', y='venta_totales')
        st.pyplot(fig2)

        #Ventas por tienda
        st.subheader("Ventas total por tienda")
        ventas_tienda = df.groupby('tienda')['venta_total'].sum()

        fig3 = plt.figure()
        ventas_tienda.plot(kind='bar', x='tienda', y='venta_totales')
        st.pyplot(fig3)

else:
    st.error("El archivo CSV debe contener las columnas: " + str(columnas_necesarias))

    
st.title("Correlación de Pearson - ventas")
"""

if archivo is not None:
    df = pd.read_csv(archivo)
    
    st.subheader("Variables numéricas disponibles")
    vars_numericas = df.select_dtypes(include='number').columns.tolist()
    st.write(vars_numericas)

    var_x = st.selectbox("Selecciona la primera variable", vars_numericas)
    var_y = st.selectbox("Selecciona la segunda variable", vars_numericas)

    if var_x != var_y:
        correlacion = df[var_x].corr(df[var_y], method='pearson')

        st.metric(
            label="Correlación entre {var_x} y {var_y}",
            value=round(correlacion, 3)
        )

        #Interpretación automática
        if abs(correlacion) >= 0.7:
            st.warning("Correlación fuerte")
        elif abs(correlacion) >= 0.4:
            st.info("Correlación moderada")
        elif abs(correlacion) >= 0.2:
            st.success("Correlación débil")
        else:
            st.write("Correlación nula o inexistente")

        st.info("Interpretación: {interpretacion}")

    else:
        st.error("Seleccione dos variables diferentes")
else:
    st.error("Sube un archivo CSV")
