# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:31:54 2023

@author: Samuel
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

st.title("Base de Datos")
st.markdown(
    f'<div style="background-color:#FFBFBF; padding: 10px 25px; border-radius: 5px;"><h4 style="color:#320014; font-size: 16px;">¡Bienvenido a la base de datos de tus viñas en VitiPredict! Aquí tendrás acceso a información detallada y actualizada de tus fincas. Podrás realizar consultas básicas, filtrar y ordenar los datos según tus necesidades, y visualizarlos de manera numérica. </h4></div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

inicial = pd.read_csv("UH_2023_TRAIN.txt", sep = "|")
predicciones = pd.read_csv("Nearest Neighbors.txt", sep = "|", header = None)

inicial_2021 = inicial.iloc[:8526,:]         

final_rellenado = inicial
final_rellenado.iloc[8526:,10] = predicciones.iloc[:,6]


# TABLA DE DATOS GENÉRICA
df = inicial_2021
# df['ID_ZONA'] = df['ID_ZONA'].astype(str)
# df['ID_ESTACION'] = df['ID_ESTACION'].astype(str)
# df['ALTITUD'] = df['ALTITUD'].astype(str)
# df['VARIEDAD'] = df['VARIEDAD'].astype(str)
# df['MODO'] = df['MODO'].astype(str)
# df['TIPO'] = df['TIPO'].astype(str)
# df['COLOR'] = df['COLOR'].astype(str)

# Obtener la lista de variables del dataset
variables = list(df.columns)

# Permitir al usuario elegir la variable por la que quiere filtrar
filtro_variable = st.selectbox('Selecciona la variable por la que quieres filtrar', variables)

# Permitir al usuario elegir el tipo de comparación que quiere hacer
operadores = {
    'igual a': lambda x, y: x == y,
    'distinto de': lambda x, y: x != y,
    'mayor que': lambda x, y: x > y,
    'mayor o igual que': lambda x, y: x >= y,
    'menor que': lambda x, y: x < y,
    'menor o igual que': lambda x, y: x <= y
}
operador = st.selectbox('Selecciona el tipo de comparación', list(operadores.keys()))

# Permitir al usuario elegir el valor concreto a filtrar
valor_filtro = st.text_input('Introduce el valor a filtrar')

# Filtrar el dataset
if st.button('Filtrar'):
    try:
        # Convertir el valor a filtrar al tipo de la variable correspondiente
        valor_filtro = df[filtro_variable].dtype.type(valor_filtro)

        # Obtener las filas que cumplen la condición
        filas = df.loc[operadores[operador](df[filtro_variable], valor_filtro)]

        # Mostrar el número de filas que cumplen la condición
        st.write('Se han encontrado {} filas que cumplen las condiciones de la query.'.format(len(filas)))

        # Mostrar una tabla con las filas que cumplen la condición
        st.write(filas)

    except ValueError:
        st.write('El valor introducido no es válido para la variable seleccionada.')


