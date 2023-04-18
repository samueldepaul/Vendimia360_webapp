# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
from pathlib import Path
import base64
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(r"Group 1header (2) (1).jpg")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

st.title("Página principal")
st.markdown(
    f'<div style="background-color:#FFBFBF; padding: 10px 25px; border-radius: 5px;"><h4 style="color:#320014; font-size: 16px;">Bienvenido a la página principal de Vendimia360. En esta sección, podrás tener acceso a información histórica de tus fincas. Podrás navegar entre diferentes visualizaciones para comprender mejor el comportamiento de tus viñedos en el pasado. Observa cómo ha evolucionado la producción de uvas, la cosecha en las diferentes zonas y otros datos relevantes que te ayudarán a tomar decisiones informadas para el futuro de tus cultivos. ¡Explora y descubre con Vendimia360!</h4></div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

st.write("""## Información sobre la producción""")

inicial = pd.read_csv("UH_2023_TRAIN.txt", sep = "|")
predicciones = pd.read_csv("Nearest Neighbors.txt", sep = "|", header = None)

inicial_2021 = inicial.iloc[:8526,:]         

final_rellenado = inicial
final_rellenado.iloc[8526:,10] = predicciones.iloc[:,6]

st.write("""Para empezar, te mostramos un gráfico de la evolución de la producción total de tus fincas a lo largo de las diferentes campañas.""")
         
## Gráfico de producción total por campaña
ts_prod_campaña = inicial_2021.groupby("CAMPAÑA")["PRODUCCION"].sum().reset_index()
from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 0, 0), (0.75, 0, 0.25), (0.5, 0, 0.5), (0.25, 0, 0.75), (0,0, 0)]
cmap = LinearSegmentedColormap.from_list('red_purple', colors)

import plotly.express as px
ts_prod_campaña = inicial_2021.groupby("CAMPAÑA")["PRODUCCION"].sum().reset_index()
colors = ["#ff0000", "#c9003f", "#96007f", "#6100bf", "#000000"]
colors_reverted =  colors[::-1]

fig = px.line(ts_prod_campaña, x='CAMPAÑA', y='PRODUCCION',
              labels={'CAMPAÑA': 'Campaña', 'PRODUCCION': 'Producción'},
              title='Producción total de los viñedos en cada campaña (en kg de uva)',
              color_discrete_sequence=colors)
fig.update_traces(line=dict(width=3))
fig.update_yaxes(range=[6500000, 11500000])
fig.update_layout(
    plot_bgcolor='#f9f9f9',  # Color de fondo del gráfico
    paper_bgcolor='#f9f9f9',  # Color de fondo del papel
    )
st.plotly_chart(fig)

###PRODUCCIÓN EN FUNCIÓN DE VARIABLE A ELEGIR
st.markdown("<hr>", unsafe_allow_html=True)
st.write("""A continuación puedes observar la producción de la campaña que desees agrupando en función de una variable en particular.
         Puedes escoger si deseas ver la producción total o la producción media de cada grupo.""")

# inicial_2021['ID_ZONA'] = inicial_2021['ID_ZONA'].astype(str)
# inicial_2021['ID_ESTACION'] = inicial_2021['ID_ESTACION'].astype(str)
# inicial_2021['VARIEDAD'] = inicial_2021['VARIEDAD'].astype(str)
# inicial_2021['MODO'] = inicial_2021['MODO'].astype(str)
# inicial_2021['TIPO'] = inicial_2021['TIPO'].astype(str)
# inicial_2021['COLOR'] = inicial_2021['COLOR'].astype(str)


group_variable = st.selectbox("Selecciona una variable en función de la que desees agrupar las viñas", ("Estación Meteorológica", "Zona", "Variedad", "Modo", "Tipo", "Color"))
if group_variable == "Estación Meteorológica":
    group_variable_name = "ID_ESTACION"
elif group_variable == "Zona":
    group_variable_name = "ID_ZONA"
elif group_variable == "Modo":
    group_variable_name = "MODO"
elif group_variable == "Tipo":
    group_variable_name = "TIPO"
elif group_variable == "Color":
    group_variable_name = "COLOR"
elif group_variable == "Variedad":
    group_variable_name = "VARIEDAD"
    
year = st.selectbox("Selecciona la campaña que deseas visualizar", (14,15,16,17,18,19,20,21))

opcion = st.radio("Seleccione una opción:", ("Producción media", "Producción total"))
if opcion == "Producción media":
    # Agrupar los datos por variable escogida y promediar la PRODUCCIÓN
    df = inicial_2021.groupby([group_variable_name, 'CAMPAÑA'])['PRODUCCION'].mean().reset_index()
elif opcion == "Producción total":
    # Agrupar los datos por variable escogida y sumar la PRODUCCIÓN
    df = inicial_2021.groupby([group_variable_name, 'CAMPAÑA'])['PRODUCCION'].sum().reset_index()

import plotly.express as px
df = df[df["CAMPAÑA"] == year]
# Establecer la paleta de colores
color_sequence = px.colors.sequential.Purples

df[group_variable_name] = pd.Categorical(df[group_variable_name], ordered=True)
df[group_variable_name] = df[group_variable_name].cat.codes

# Crear la figura
fig = px.bar(df, x=group_variable_name, y='PRODUCCION', color='PRODUCCION',
             color_continuous_scale=colors_reverted,
             labels={group_variable_name: group_variable, 'PRODUCCIÓN': 'Producción'},
             title='{} en función de {} para la campaña 20{} (en kg de uva)'.format(opcion, group_variable, year))

# Personalizar el diseño de la figura
fig.update_layout(
    plot_bgcolor='#f9f9f9',  # Color de fondo del gráfico
    paper_bgcolor='#f9f9f9',  # Color de fondo del papel
    xaxis=dict(title=dict(text=group_variable, font=dict(color='#444444', size=12))),
    yaxis=dict(title=dict(text='Producción', font=dict(color='#444444', size=12))),
    margin=dict(l=60, r=20, t=60, b=20),
    hovermode='x'
)
fig.update_xaxes(type='category')


# Mostrar el gráfico en la app de Streamlit
st.plotly_chart(fig)


st.write("""## Información sobre tus fincas""")

data = inicial_2021

import plotly.graph_objects as go

# Control deslizante para seleccionar el año
year = st.radio("Seleccione una opción", ('Todos los años', 'Sólo un año'))
if year == 'Sólo un año':
    selected_year = st.selectbox('Seleccione un año', sorted(data['CAMPAÑA'].unique()))
    data = data[data['CAMPAÑA'] == selected_year]

# Selección de la variable de interés
group_variable = st.selectbox("Selecciona una variable en función de la que desees agrupar las viñas", ("Estación Meteorológica", "Variedad", "Modo", "Tipo", "Color"), key=1)

if group_variable == "Estación Meteorológica":
    variable = "ID_ESTACION"
elif group_variable == "Modo":
    variable = "MODO"
elif group_variable == "Tipo":
    variable = "TIPO"
elif group_variable == "Color":
    variable = "COLOR"
elif group_variable == "Variedad":
    variable = "VARIEDAD"

# Resumen de las fincas en función de la variable seleccionada
summary = data[variable].value_counts()
percentages = round(summary/summary.sum()*100, 2)

# Visualización del resumen
st.write('### Distribución de las fincas en función de {}'.format(group_variable))
st.write('Total de fincas consideradas:', len(np.unique(data["ID_FINCA"])))

# División de la página en dos columnas
left_column, right_column = st.columns(2)

# Columna izquierda
with left_column:
    st.write('Cantidad de fincas por categoría:')
    st.write(summary)
    st.write('Porcentaje de fincas por categoría:')
    st.write(percentages)

# Columna derecha
with right_column:
    fig = go.Figure(data=[go.Pie(labels=percentages.index, values=percentages, 
                                 textinfo='label+percent',
                                 marker=dict(colors = ['#a81c43', '#d11141', '#e87b7c', '#f2a1a5', '#f9c6c9', '#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000', '#4d0000', '#1a0000'],
                                             line=dict(color='#000000', width=2)))])
    fig.update_layout(title='Resumen de las fincas en función de {}'.format(group_variable), 
                      font=dict(size=18), legend=dict(title=group_variable))
    fig.update_traces(hoverinfo='label+percent', 
                      textfont_size=20,
                      textposition='inside',
                      hole=0.4)
    fig.update_layout(height=700, width=500)
    st.plotly_chart(fig)


# dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

# classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Fores"))

# def get_dataset(dataset_name):
#     if dataset_name == "Iris":
#         data = datasets.load_iris()
#     elif dataset_name == "Breast Cancer":
#         data = datasets.load_breast_cancer()
#     elif dataset_name == "Wine":
#         data = datasets.load_wine()
    
#     X = data.data
#     y = data.target
    
#     return X, y

# X, y = get_dataset(dataset_name)
# st.write("Dataset Shape", X.shape)
# st.write("Number of classes", len(np.unique(y)))

# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == "KNN":
#         K = st.sidebar.slider("K", 1, 15)
#         params["K"] = K
#     return params

# params = add_parameter_ui(classifier_name)

# def get_classifier(clf_name, params):
#     if clf_name == "KNN":
#         clf = KNeighborsClassifier(n_neighbors = params["K"])
#     return clf

# clf = get_classifier(classifier_name, params)

# # Classification
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)

# st.write(f"classifier = {classifier_name}")
# st.write(f"accuracy = {acc}")

# # PLOT
# pca = PCA(2)
# X_projected = pca.fit_transform(X)

# x1 = X_projected[:,0]
# x2 = X_projected[:,1]

# fig = plt.figure()
# plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = cmap)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 1")
# plt.colorbar()

# st.pyplot(fig)





