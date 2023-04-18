# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:31:54 2023

@author: Samuel
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
import base64


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

st.title("Predicción")

st.markdown(
    f'<div style="background-color:#FFBFBF; padding: 10px 25px; border-radius: 5px;"><h4 style="color:#320014; font-size: 16px;">En esta sección, conocerás la producción esperada de tus fincas para la próxima campaña. Tienes la opción de introducir manualmente los datos de una finca, o de subir un archivo CSV con los datos de múltiples fincas.</h4></div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

forma_intro = st.selectbox("Selecciona cómo quieres introducir los datos:", ("Subir CSV", "Manualmente"))


# Leer archivo CSV como dataframe
train_original = pd.read_csv('train_original.csv')
df_train2 = pd.read_csv('df_train2.csv')

# Carga de los modelos para los 5 folds:
import pickle

# cargar características
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# cargar modelos
models = []
for i in range(1, 6):
    with open(f'model_fold_{i}.pkl', 'rb') as f:
        model = pickle.load(f)
        models.append(model)

if forma_intro == "Manualmente":
    manual_CAMPAÑA = 22
    manual_ID_FINCA = st.text_input('Introduce el ID de la Finca. (Valor entero)')
    manual_ID_ZONA = st.text_input('Introduce el ID de la Zona. (Valor entero)')
    manual_ID_ESTACION = st.text_input('Introduce el ID de la Estación Meteorológica. (Valor entero)')
    manual_ALTITUD = st.text_input('Introduce la altitud de la finca. (Valor entero o decimal)')
    manual_VARIEDAD = st.text_input('Introduce la variedad de uva que se cosecha. Debe de ser una de las siguientes opciones: {}'.format(np.unique(train_original["VARIEDAD"])))
    manual_MODO = st.text_input('Introduce el Modo de cultivo. (Toma valores 1 o 2)')
    manual_TIPO = st.text_input('Introduce el Tipo de cultivo. (Toma valores 0 o 1)')
    manual_COLOR = st.text_input('Introduce el Color de la uva. (Toma valores 0 o 1)')
    manual_SUPERFICIE = st.text_input('Introduce la Superfície de la plantación en hectáreas (Valor entero o decimal)')
    manual_PRODUCCION = np.nan

    if all(val is not '' for val in [manual_COLOR, manual_SUPERFICIE, manual_ID_FINCA, manual_ID_ZONA, manual_MODO, manual_TIPO, manual_ID_ESTACION, manual_ALTITUD]):
        # Crea un diccionario con los valores introducidos por el usuario
        user_input = {
            "CAMPAÑA": np.int(manual_CAMPAÑA),
            "ID_FINCA": np.int(manual_ID_FINCA),
            "ID_ZONA": np.int(manual_ID_ZONA),
            "ID_ESTACION": np.int(manual_ID_ESTACION),
            "ALTITUD": str(manual_ALTITUD),
            "VARIEDAD": np.int(manual_VARIEDAD),
            "MODO": np.int(manual_MODO),
            "TIPO": np.int(manual_TIPO),
            "COLOR": np.int(manual_COLOR),
            "SUPERFICIE": np.float(manual_SUPERFICIE),
            "PRODUCCION": np.float(manual_PRODUCCION)
        }
        
        # Crea un dataframe con los valores introducidos por el usuario
        test_manual = pd.DataFrame(user_input, index=[0])
        
        # Une los dataframes de entrenamiento y el de entrada del usuario
        df_train = pd.concat([train_original, test_manual], ignore_index=True)

######################################### TRANSFORMAR EL DF AL FORMATO CORRESPONDIENTE:    
    if st.button('Realizar predicciones'):
        ## Clave única
        df_train['ID_KEY_PROD']=df_train[['CAMPAÑA','ID_FINCA','VARIEDAD','MODO',
                                  'TIPO','COLOR','SUPERFICIE']].apply(lambda x : str(x['CAMPAÑA'])+'_'+str(x['ID_FINCA'])+'_'
                                                                      +str(x['VARIEDAD'])+'_'+str(x['MODO'])+'_'+str(x['TIPO'])+'_'
                                                                      +str(x['COLOR'])+'_'+str(x['SUPERFICIE']),axis=1)
        ## Imputación SUPERFICIE
        df_train2=df_train.copy()
        df_train_trs=df_train.copy()
        # Eliminando ceros en todos los periodos. ya que es imposible tener ceros como SUPERFICIE
        df_train_trs['SUPERFICIE2']=df_train_trs['SUPERFICIE'].copy()
        df_train_trs.loc[df_train_trs['SUPERFICIE2']==0,'SUPERFICIE2']=np.nan
        df_superficie_imput=df_train_trs[df_train_trs['CAMPAÑA'].isin([20,21,22])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','SUPERFICIE2']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).median().reset_index()
        df_superficie_imput.columns=['ID_FINCA', 'VARIEDAD', 'MODO', 'TIPO', 'COLOR', 'superficie_imp']
        df_train_trsa=df_train_trs[df_train_trs['CAMPAÑA'].isin([14,15,16,17,18,19])].copy()
        df_train_trsa=pd.merge(df_train_trsa,df_superficie_imput,how='left',on=['ID_FINCA', 'VARIEDAD', 'MODO', 'TIPO', 'COLOR'])
        df_train_trsb=df_train_trs[df_train_trs['CAMPAÑA'].isin([20,21,22])].copy()
        df_train_trsb['superficie_imp']=df_train_trsb['SUPERFICIE2'].copy()
        df_train_trs2=pd.concat([df_train_trsa,df_train_trsb])
        # Imputando valores nulos en SUPERFICIE. Por el promedio por campaña
        df_mean_super=df_train_trs2[['CAMPAÑA','superficie_imp']].groupby('CAMPAÑA').mean().reset_index()
        dic_mean_super={}
        for x,y in zip(df_mean_super['CAMPAÑA'],df_mean_super['superficie_imp']):
            dic_mean_super[x]=np.round(y,4)
        df_train_trs2.loc[df_train_trs2['superficie_imp'].isnull(),'superficie_imp']=df_train_trs2.loc[df_train_trs2['superficie_imp'].isnull(),['CAMPAÑA','superficie_imp']].apply(lambda x : dic_mean_super[x['CAMPAÑA']],axis=1)
        
        ## Transformación de ALTITUD a numérico
        df_train_trs2['lst_altitud']=df_train_trs2['ALTITUD'].apply(lambda x : str(x).split('-'))
        df_train_trs2['altitud_tr']=df_train_trs2['lst_altitud'].apply(lambda lst_alt : sum([float(x) for x in lst_alt])/len(lst_alt))
        del df_train_trs2['lst_altitud']
        # Imputando valores nulos en ALTITUD. Por el promedio por campaña
        df_mean_alt=df_train_trs2[['CAMPAÑA','altitud_tr']].groupby('CAMPAÑA').mean().reset_index()
        dic_mean_alt={}
        for x,y in zip(df_mean_alt['CAMPAÑA'],df_mean_alt['altitud_tr']):
            dic_mean_alt[x]=np.round(y,2)
        df_train_trs2.loc[df_train_trs2['altitud_tr'].isnull(),'altitud_tr']=df_train_trs2.loc[df_train_trs2['altitud_tr'].isnull(),['CAMPAÑA','altitud_tr']].apply(lambda x : dic_mean_alt[x['CAMPAÑA']],axis=1)
        
        ## Transformación de datos categóricos
        from sklearn import preprocessing
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, StratifiedKFold
        from math import sqrt
        import itertools
        categorical=['ID_ZONA','ID_ESTACION','VARIEDAD', 'MODO']
        for l in categorical:
            df_train_trs2[l+'_tr']=df_train_trs2[l].copy()
        for l in [x+'_tr' for x in categorical]:
            le = preprocessing.LabelEncoder()
            le.fit(df_train_trs2[l])
            df_train_trs2.loc[~df_train_trs2[l].isnull(),l]=le.transform(df_train_trs2.loc[~df_train_trs2[l].isnull(),l])
        df_train2=pd.merge(df_train2,df_train_trs2[['ID_KEY_PROD','superficie_imp','altitud_tr','ID_ZONA_tr','ID_ESTACION_tr','VARIEDAD_tr','MODO_tr']],how='left',on=['ID_KEY_PROD'])
        
        ## Agrupaciones de PRODUCCION para campañas t-1, t-2, t-3
        df_agrup_train1=df_train2.copy()
        df_agrup_train2=df_train2.copy()
        df_agrup_train3=df_train2.copy()
        df_agrup_train1['CAMPAÑA']=df_agrup_train1['CAMPAÑA']+1
        df_agrup_train2['CAMPAÑA']=df_agrup_train2['CAMPAÑA']+2
        df_agrup_train3['CAMPAÑA']=df_agrup_train3['CAMPAÑA']+3
        dic_nom_col={'ID_FINCA':'finca', 'ID_ZONA':'zona', 'ID_ESTACION':'esta',
                     'VARIEDAD':'varied', 'MODO':'modo', 'TIPO':'tipo', 'COLOR':'color','superficie_imp':'super'}
        train_agg={}
        train_agg['PRODUCCION']=['mean','var','min','max']
        def train_agrup_cols(col):
            if col=='ID_FINCA':
                df_agrup_col1=df_agrup_train1[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col2=df_agrup_train2[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col3=df_agrup_train3[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col1.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag1']
                df_agrup_col2.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag2']
                df_agrup_col3.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag3']
                return df_agrup_col1,df_agrup_col2,df_agrup_col3
            else:
                df_agrup_col1=df_agrup_train1[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col2=df_agrup_train2[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col3=df_agrup_train3[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col1.columns=['prod_'+dic_nom_col[col]+'_lag1_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col1.columns]
                df_agrup_col2.columns=['prod_'+dic_nom_col[col]+'_lag2_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col2.columns]
                df_agrup_col3.columns=['prod_'+dic_nom_col[col]+'_lag3_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col3.columns]
                return df_agrup_col1,df_agrup_col2,df_agrup_col3
        df_impxvaried_lag1=df_agrup_train1[df_agrup_train1['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag1={}
        for x in sorted(df_impxvaried_lag1['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag1['VARIEDAD'],df_impxvaried_lag1['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag1[x]=dic_temp
        
        df_impxvaried_lag2=df_agrup_train2[df_agrup_train2['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag2={}
        for x in sorted(df_impxvaried_lag2['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag2['VARIEDAD'],df_impxvaried_lag2['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag2[x]=dic_temp
        
        df_impxvaried_lag3=df_agrup_train3[df_agrup_train3['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag3={}
        for x in sorted(df_impxvaried_lag3['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag3['VARIEDAD'],df_impxvaried_lag3['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag3[x]=dic_temp
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[0],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1']=df_train2.loc[(df_train2['prod_finca_lag1'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[1],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag2'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag2']=df_train2.loc[(df_train2['prod_finca_lag2'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[2],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag3'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag3']=df_train2.loc[(df_train2['prod_finca_lag3'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[0],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[1],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag2_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag2_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag2_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[2],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag3_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag3_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag3_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
                    
        ## Agrupaciones de PRODUCCIÓN acumuladas 
        # t-1 y t-2
        def train_agrup_cols2(col):
            df_agrup_col1_2=pd.DataFrame()
            if col=='ID_FINCA':
                for camp in [16,17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][[col,'PRODUCCION']].groupby(col).mean().reset_index()
                    df_temp1.columns=[ col , 'prod_'+dic_nom_col[col]+'_lag1_2']
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_2=pd.concat([df_agrup_col1_2,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_2
            else:
                for camp in [16,17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][[col,'PRODUCCION']].groupby(col).agg(train_agg).reset_index()
                    df_temp1.columns=['prod_'+dic_nom_col[col]+'_lag1_2'+x[1] if x[0] not in [col] else x[0] for x in df_temp1.columns]
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_2=pd.concat([df_agrup_col1_2,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_2
        df_train2=pd.merge(df_train2,train_agrup_cols2('ID_FINCA'),how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1_2'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1_2']=df_train2.loc[(df_train2['prod_finca_lag1_2'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols2(col),how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_2'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_2'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_2'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)  
        #t-1, t-2 y t-3
        def train_agrup_cols3(col):
            df_agrup_col1_3=pd.DataFrame()
            if col=='ID_FINCA':
                for camp in [17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][[col,'PRODUCCION']].groupby(col).mean().reset_index()
                    df_temp1.columns=[ col , 'prod_'+dic_nom_col[col]+'_lag1_3']
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_3=pd.concat([df_agrup_col1_3,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_3
            else:
                for camp in [17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][[col,'PRODUCCION']].groupby(col).agg(train_agg).reset_index()
                    df_temp1.columns=['prod_'+dic_nom_col[col]+'_lag1_3'+x[1] if x[0] not in [col] else x[0] for x in df_temp1.columns]
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_3=pd.concat([df_agrup_col1_3,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_3
        df_train2=pd.merge(df_train2,train_agrup_cols3('ID_FINCA'),how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1_3'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1_3']=df_train2.loc[(df_train2['prod_finca_lag1_3'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols3(col),how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_3'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_3'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_3'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)  
        ## Agrupaciones de PRODUCCION para t-1, t-2, t-3 por llave de unicidad
        def train_agrup_completa():
            df_agrup_completo1=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp1.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1']
                df_temp1['CAMPAÑA']=camp
                df_agrup_completo1=pd.concat([df_agrup_completo1,df_temp1]).reset_index(drop=True)
                
            df_agrup_completo2=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp2=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-2])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp2.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','prod_completo_lag2']
                df_temp2['CAMPAÑA']=camp
                df_agrup_completo2=pd.concat([df_agrup_completo2,df_temp2]).reset_index(drop=True)
            
            df_agrup_completo3=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp3=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-3])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp3.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','prod_completo_lag3']
                df_temp3['CAMPAÑA']=camp
                df_agrup_completo3=pd.concat([df_agrup_completo3,df_temp3]).reset_index(drop=True)
            
            df_agrup_completo4=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp4=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp4.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1_2']
                df_temp4['CAMPAÑA']=camp
                df_agrup_completo4=pd.concat([df_agrup_completo4,df_temp4]).reset_index(drop=True)
            
            df_agrup_completo5=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp5=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp5.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1_3']
                df_temp5['CAMPAÑA']=camp
                df_agrup_completo5=pd.concat([df_agrup_completo5,df_temp5]).reset_index(drop=True)
            
            return df_agrup_completo1, df_agrup_completo2, df_agrup_completo3, df_agrup_completo4, df_agrup_completo5
        df_train2=pd.merge(df_train2,train_agrup_completa()[0],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1'] = df_train2.loc[(df_train2['prod_completo_lag1'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[1],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag2'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag2'] = df_train2.loc[(df_train2['prod_completo_lag2'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[2],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag3'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag3'] = df_train2.loc[(df_train2['prod_completo_lag3'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[3],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1_2'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1_2'] = df_train2.loc[(df_train2['prod_completo_lag1_2'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[4],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1_3'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1_3'] = df_train2.loc[(df_train2['prod_completo_lag1_3'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2['rat_prodcomp_superf_lag1']=df_train2['prod_completo_lag1']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag2']=df_train2['prod_completo_lag2']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag3']=df_train2['prod_completo_lag3']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag1_2']=df_train2['prod_completo_lag1_2']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag1_3']=df_train2['prod_completo_lag1_3']/df_train2['superficie_imp']
        
        ## Reducción de la memoria dataset
        def reduce_mem_usage(df):
            """ iterate through all the columns of a dataframe and modify the data type
                to reduce memory usage.        
            """
            start_mem = df.memory_usage().sum() / 1024**2
            for col in df.columns:
                col_type = df[col].dtype
                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)  
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
            end_mem = df.memory_usage().sum() / 1024**2
            return df
        df_train2=reduce_mem_usage(df_train2)
        target='PRODUCCION'
        df_train_pt4=df_train2.copy()
        df_train_pt4=df_train_pt4[df_train_pt4['CAMPAÑA'].isin([17,18,19,20,21,22])].reset_index(drop=True)
        ss = StandardScaler()
        df_scaled = pd.DataFrame(ss.fit_transform(df_train_pt4[features]),columns = features)
        df_scaled=pd.concat([df_train_pt4[['ID_KEY_PROD','CAMPAÑA','PRODUCCION']],df_scaled],axis=1)

        # seleccionar características
        test = df_scaled[df_scaled["CAMPAÑA"] == 22].reset_index(drop = True)
        train = df_scaled[df_scaled["CAMPAÑA"] < 22].reset_index(drop = True)
        
        # hacer predicciones
        predictions = []
        for model in models:
            prediction = model.predict(test[features])
            predictions.append(prediction)
            
        # tomar el promedio de las predicciones de todos los modelos
        test["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
        test_manual["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
        predictions_mean = pd.DataFrame(np.abs(np.mean(predictions, axis=0)))

        st.write("## Resultados:")
        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)
        prod_ant = np.round(np.mean(train.groupby(["CAMPAÑA"]).mean()["PRODUCCION"]),2)
        prod_fut = np.round(np.mean(test.groupby(["CAMPAÑA"]).sum()["predicted_tot"]),2)
        incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
        col1.metric("Producción media de todas tus fincas", "{} kg".format(prod_ant))
        col2.metric("Producción esperada próxima campaña finca introducida", "{} kg".format(prod_fut), "{}%".format(incremento))
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        grouped_modo = train_original[train_original["CAMPAÑA"] < 22].groupby(["MODO"]).mean()
        prod_ant = np.round(np.mean(grouped_modo[grouped_modo.index == np.int(manual_MODO)]["PRODUCCION"]),2)
        prod_fut = np.round(np.mean(test_manual.groupby(["MODO"]).sum()["predicted_tot"]),2)
        incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
        col1.metric("Producción media fincas con mismo modo de cultivo", "{} kg".format(prod_ant))
        col2.metric("Producción esperada próxima campaña finca introducida", "{} kg".format(prod_fut), "{}%".format(incremento))
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        grouped_tipo = train_original[train_original["CAMPAÑA"] < 22].groupby(["TIPO"]).mean()
        prod_ant = np.round(np.mean(grouped_tipo[grouped_tipo.index == np.int(manual_TIPO)]["PRODUCCION"]),2)
        prod_fut = np.round(np.mean(test_manual.groupby(["TIPO"]).sum()["predicted_tot"]),2)
        incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
        col1.metric("Producción media fincas con mismo tipo de cultivo", "{} kg".format(prod_ant))
        col2.metric("Producción esperada próxima campaña finca introducida", "{} kg".format(prod_fut), "{}%".format(incremento))
        

        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        grouped_color = train_original[train_original["CAMPAÑA"] < 22].groupby(["COLOR"]).mean()
        prod_ant = np.round(np.mean(grouped_color[grouped_color.index == np.int(manual_COLOR)]["PRODUCCION"]),2)
        prod_fut = np.round(np.mean(test_manual.groupby(["COLOR"]).sum()["predicted_tot"]),2)
        incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
        col1.metric("Producción media de las fincas con uva del mismo color", "{} kg".format(prod_ant))
        col2.metric("Producción esperada próxima campaña finca introducida", "{} kg".format(prod_fut), "{}%".format(incremento))
        
        # [AÑADIR ALGO MÁS???]

elif forma_intro == "Subir CSV":
    st.write("##### Limitaciones:")
    st.write("El archivo subido debe de tener como mínimo las variables: {}".format(train_original.columns.values))
    st.write("(Se puede subir por ejemplo un csv que contenga únicamente las observaciones de la campaña 22 de UH_2023_TRAIN.txt)")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")
    if uploaded_file is not None:
        test_subido = pd.read_csv(uploaded_file)
        df_train = pd.concat([train_original, test_subido])    
        
        st.write("## Resultados:")
        
        ######################################### TRANSFORMAR EL DF AL FORMATO CORRESPONDIENTE:    
        ## Clave única
        df_train['ID_KEY_PROD']=df_train[['CAMPAÑA','ID_FINCA','VARIEDAD','MODO',
                                  'TIPO','COLOR','SUPERFICIE']].apply(lambda x : str(x['CAMPAÑA'])+'_'+str(x['ID_FINCA'])+'_'
                                                                      +str(x['VARIEDAD'])+'_'+str(x['MODO'])+'_'+str(x['TIPO'])+'_'
                                                                      +str(x['COLOR'])+'_'+str(x['SUPERFICIE']),axis=1)
        ## Imputación SUPERFICIE
        df_train2=df_train.copy()
        df_train_trs=df_train.copy()
        # Eliminando ceros en todos los periodos. ya que es imposible tener ceros como SUPERFICIE
        df_train_trs['SUPERFICIE2']=df_train_trs['SUPERFICIE'].copy()
        df_train_trs.loc[df_train_trs['SUPERFICIE2']==0,'SUPERFICIE2']=np.nan
        df_superficie_imput=df_train_trs[df_train_trs['CAMPAÑA'].isin([20,21,22])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','SUPERFICIE2']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).median().reset_index()
        df_superficie_imput.columns=['ID_FINCA', 'VARIEDAD', 'MODO', 'TIPO', 'COLOR', 'superficie_imp']
        df_train_trsa=df_train_trs[df_train_trs['CAMPAÑA'].isin([14,15,16,17,18,19])].copy()
        df_train_trsa=pd.merge(df_train_trsa,df_superficie_imput,how='left',on=['ID_FINCA', 'VARIEDAD', 'MODO', 'TIPO', 'COLOR'])
        df_train_trsb=df_train_trs[df_train_trs['CAMPAÑA'].isin([20,21,22])].copy()
        df_train_trsb['superficie_imp']=df_train_trsb['SUPERFICIE2'].copy()
        df_train_trs2=pd.concat([df_train_trsa,df_train_trsb])
        # Imputando valores nulos en SUPERFICIE. Por el promedio por campaña
        df_mean_super=df_train_trs2[['CAMPAÑA','superficie_imp']].groupby('CAMPAÑA').mean().reset_index()
        dic_mean_super={}
        for x,y in zip(df_mean_super['CAMPAÑA'],df_mean_super['superficie_imp']):
            dic_mean_super[x]=np.round(y,4)
        df_train_trs2.loc[df_train_trs2['superficie_imp'].isnull(),'superficie_imp']=df_train_trs2.loc[df_train_trs2['superficie_imp'].isnull(),['CAMPAÑA','superficie_imp']].apply(lambda x : dic_mean_super[x['CAMPAÑA']],axis=1)
        
        ## Transformación de ALTITUD a numérico
        df_train_trs2['lst_altitud']=df_train_trs2['ALTITUD'].apply(lambda x : str(x).split('-'))
        df_train_trs2['altitud_tr']=df_train_trs2['lst_altitud'].apply(lambda lst_alt : sum([float(x) for x in lst_alt])/len(lst_alt))
        del df_train_trs2['lst_altitud']
        # Imputando valores nulos en ALTITUD. Por el promedio por campaña
        df_mean_alt=df_train_trs2[['CAMPAÑA','altitud_tr']].groupby('CAMPAÑA').mean().reset_index()
        dic_mean_alt={}
        for x,y in zip(df_mean_alt['CAMPAÑA'],df_mean_alt['altitud_tr']):
            dic_mean_alt[x]=np.round(y,2)
        df_train_trs2.loc[df_train_trs2['altitud_tr'].isnull(),'altitud_tr']=df_train_trs2.loc[df_train_trs2['altitud_tr'].isnull(),['CAMPAÑA','altitud_tr']].apply(lambda x : dic_mean_alt[x['CAMPAÑA']],axis=1)
        
        ## Transformación de datos categóricos
        from sklearn import preprocessing
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, StratifiedKFold
        from math import sqrt
        import itertools
        categorical=['ID_ZONA','ID_ESTACION','VARIEDAD', 'MODO']
        for l in categorical:
            df_train_trs2[l+'_tr']=df_train_trs2[l].copy()
        for l in [x+'_tr' for x in categorical]:
            le = preprocessing.LabelEncoder()
            le.fit(list(df_train_trs2[l].dropna()))
            df_train_trs2.loc[~df_train_trs2[l].isnull(),l]=le.transform(df_train_trs2.loc[~df_train_trs2[l].isnull(),l])
        df_train2=pd.merge(df_train2,df_train_trs2[['ID_KEY_PROD','superficie_imp','altitud_tr','ID_ZONA_tr','ID_ESTACION_tr','VARIEDAD_tr','MODO_tr']],how='left',on=['ID_KEY_PROD'])
        
        ## Agrupaciones de PRODUCCION para campañas t-1, t-2, t-3
        df_agrup_train1=df_train2.copy()
        df_agrup_train2=df_train2.copy()
        df_agrup_train3=df_train2.copy()
        df_agrup_train1['CAMPAÑA']=df_agrup_train1['CAMPAÑA']+1
        df_agrup_train2['CAMPAÑA']=df_agrup_train2['CAMPAÑA']+2
        df_agrup_train3['CAMPAÑA']=df_agrup_train3['CAMPAÑA']+3
        dic_nom_col={'ID_FINCA':'finca', 'ID_ZONA':'zona', 'ID_ESTACION':'esta',
                     'VARIEDAD':'varied', 'MODO':'modo', 'TIPO':'tipo', 'COLOR':'color','superficie_imp':'super'}
        train_agg={}
        train_agg['PRODUCCION']=['mean','var','min','max']
        def train_agrup_cols(col):
            if col=='ID_FINCA':
                df_agrup_col1=df_agrup_train1[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col2=df_agrup_train2[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col3=df_agrup_train3[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).mean().reset_index()
                df_agrup_col1.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag1']
                df_agrup_col2.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag2']
                df_agrup_col3.columns=['CAMPAÑA', col , 'prod_'+dic_nom_col[col]+'_lag3']
                return df_agrup_col1,df_agrup_col2,df_agrup_col3
            else:
                df_agrup_col1=df_agrup_train1[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col2=df_agrup_train2[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col3=df_agrup_train3[['CAMPAÑA',col,'PRODUCCION']].groupby(['CAMPAÑA',col]).agg(train_agg).reset_index()
                df_agrup_col1.columns=['prod_'+dic_nom_col[col]+'_lag1_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col1.columns]
                df_agrup_col2.columns=['prod_'+dic_nom_col[col]+'_lag2_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col2.columns]
                df_agrup_col3.columns=['prod_'+dic_nom_col[col]+'_lag3_'+x[1] if x[0] not in ['CAMPAÑA',col] else x[0] for x in df_agrup_col3.columns]
                return df_agrup_col1,df_agrup_col2,df_agrup_col3
        df_impxvaried_lag1=df_agrup_train1[df_agrup_train1['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag1={}
        for x in sorted(df_impxvaried_lag1['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag1['VARIEDAD'],df_impxvaried_lag1['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag1[x]=dic_temp
        
        df_impxvaried_lag2=df_agrup_train2[df_agrup_train2['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag2={}
        for x in sorted(df_impxvaried_lag2['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag2['VARIEDAD'],df_impxvaried_lag2['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag2[x]=dic_temp
        
        df_impxvaried_lag3=df_agrup_train3[df_agrup_train3['CAMPAÑA']<23][['CAMPAÑA','VARIEDAD','PRODUCCION']].groupby(['CAMPAÑA','VARIEDAD']).mean().reset_index()
        dic_impxvaried_lag3={}
        for x in sorted(df_impxvaried_lag3['CAMPAÑA'].unique()):
            dic_temp={}
            for y,z in zip(df_impxvaried_lag3['VARIEDAD'],df_impxvaried_lag3['PRODUCCION']):
                dic_temp[y]=z
            dic_impxvaried_lag3[x]=dic_temp
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[0],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1']=df_train2.loc[(df_train2['prod_finca_lag1'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[1],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag2'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag2']=df_train2.loc[(df_train2['prod_finca_lag2'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2=pd.merge(df_train2,train_agrup_cols('ID_FINCA')[2],how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag3'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag3']=df_train2.loc[(df_train2['prod_finca_lag3'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[0],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[1],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag2_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag2_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag2_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols(col)[2],how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag3_'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag3_'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag3_'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
                    
        ## Agrupaciones de PRODUCCIÓN acumuladas 
        # t-1 y t-2
        def train_agrup_cols2(col):
            df_agrup_col1_2=pd.DataFrame()
            if col=='ID_FINCA':
                for camp in [16,17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][[col,'PRODUCCION']].groupby(col).mean().reset_index()
                    df_temp1.columns=[ col , 'prod_'+dic_nom_col[col]+'_lag1_2']
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_2=pd.concat([df_agrup_col1_2,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_2
            else:
                for camp in [16,17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][[col,'PRODUCCION']].groupby(col).agg(train_agg).reset_index()
                    df_temp1.columns=['prod_'+dic_nom_col[col]+'_lag1_2'+x[1] if x[0] not in [col] else x[0] for x in df_temp1.columns]
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_2=pd.concat([df_agrup_col1_2,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_2
        df_train2=pd.merge(df_train2,train_agrup_cols2('ID_FINCA'),how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1_2'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1_2']=df_train2.loc[(df_train2['prod_finca_lag1_2'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols2(col),how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_2'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_2'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_2'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)  
        #t-1, t-2 y t-3
        def train_agrup_cols3(col):
            df_agrup_col1_3=pd.DataFrame()
            if col=='ID_FINCA':
                for camp in [17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][[col,'PRODUCCION']].groupby(col).mean().reset_index()
                    df_temp1.columns=[ col , 'prod_'+dic_nom_col[col]+'_lag1_3']
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_3=pd.concat([df_agrup_col1_3,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_3
            else:
                for camp in [17,18,19,20,21,22]:
                    df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][[col,'PRODUCCION']].groupby(col).agg(train_agg).reset_index()
                    df_temp1.columns=['prod_'+dic_nom_col[col]+'_lag1_3'+x[1] if x[0] not in [col] else x[0] for x in df_temp1.columns]
                    df_temp1['CAMPAÑA']=camp
                    df_agrup_col1_3=pd.concat([df_agrup_col1_3,df_temp1]).reset_index(drop=True)
                return df_agrup_col1_3
        df_train2=pd.merge(df_train2,train_agrup_cols3('ID_FINCA'),how='left',on=['CAMPAÑA','ID_FINCA'])
        df_train2.loc[(df_train2['prod_finca_lag1_3'].isnull()) 
                & (df_train2['CAMPAÑA']>16),'prod_finca_lag1_3']=df_train2.loc[(df_train2['prod_finca_lag1_3'].isnull()) 
                           & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        for col in ['ID_ZONA','ID_ESTACION','VARIEDAD']:
            df_train2=pd.merge(df_train2,train_agrup_cols3(col),how='left',on=['CAMPAÑA',col])
            for estad in ['mean','var','min','max']:
                df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_3'+estad].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_'+dic_nom_col[col]+'_lag1_3'+estad] = df_train2.loc[(df_train2['prod_'+dic_nom_col[col]+'_lag1_3'+estad].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)  
        ## Agrupaciones de PRODUCCION para t-1, t-2, t-3 por llave de unicidad
        def train_agrup_completa():
            df_agrup_completo1=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp1=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp1.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1']
                df_temp1['CAMPAÑA']=camp
                df_agrup_completo1=pd.concat([df_agrup_completo1,df_temp1]).reset_index(drop=True)
                
            df_agrup_completo2=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp2=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-2])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp2.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','prod_completo_lag2']
                df_temp2['CAMPAÑA']=camp
                df_agrup_completo2=pd.concat([df_agrup_completo2,df_temp2]).reset_index(drop=True)
            
            df_agrup_completo3=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp3=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-3])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp3.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','prod_completo_lag3']
                df_temp3['CAMPAÑA']=camp
                df_agrup_completo3=pd.concat([df_agrup_completo3,df_temp3]).reset_index(drop=True)
            
            df_agrup_completo4=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp4=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp4.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1_2']
                df_temp4['CAMPAÑA']=camp
                df_agrup_completo4=pd.concat([df_agrup_completo4,df_temp4]).reset_index(drop=True)
            
            df_agrup_completo5=pd.DataFrame()
            for camp in [17,18,19,20,21,22]:
                df_temp5=df_train_trs2[df_train_trs2['CAMPAÑA'].isin([camp-1,camp-2,camp-3])][['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR','PRODUCCION']].groupby(['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR']).mean().reset_index()
                df_temp5.columns=['ID_FINCA','VARIEDAD','MODO','TIPO','COLOR', 'prod_completo_lag1_3']
                df_temp5['CAMPAÑA']=camp
                df_agrup_completo5=pd.concat([df_agrup_completo5,df_temp5]).reset_index(drop=True)
            
            return df_agrup_completo1, df_agrup_completo2, df_agrup_completo3, df_agrup_completo4, df_agrup_completo5
        df_train2=pd.merge(df_train2,train_agrup_completa()[0],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1'] = df_train2.loc[(df_train2['prod_completo_lag1'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag1[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[1],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag2'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag2'] = df_train2.loc[(df_train2['prod_completo_lag2'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[2],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag3'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag3'] = df_train2.loc[(df_train2['prod_completo_lag3'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[3],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1_2'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1_2'] = df_train2.loc[(df_train2['prod_completo_lag1_2'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag2[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        
        df_train2=pd.merge(df_train2,train_agrup_completa()[4],how='left',on=['CAMPAÑA','ID_FINCA','VARIEDAD','MODO','TIPO','COLOR'])
        df_train2.loc[(df_train2['prod_completo_lag1_3'].isnull()) 
                        & (df_train2['CAMPAÑA']>16),'prod_completo_lag1_3'] = df_train2.loc[(df_train2['prod_completo_lag1_3'].isnull()) 
                                   & (df_train2['CAMPAÑA']>16),['CAMPAÑA','VARIEDAD']].apply(lambda x : dic_impxvaried_lag3[x['CAMPAÑA']][x['VARIEDAD']],axis=1)
        df_train2['rat_prodcomp_superf_lag1']=df_train2['prod_completo_lag1']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag2']=df_train2['prod_completo_lag2']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag3']=df_train2['prod_completo_lag3']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag1_2']=df_train2['prod_completo_lag1_2']/df_train2['superficie_imp']
        df_train2['rat_prodcomp_superf_lag1_3']=df_train2['prod_completo_lag1_3']/df_train2['superficie_imp']
        
        ## Reducción de la memoria dataset
        def reduce_mem_usage(df):
            """ iterate through all the columns of a dataframe and modify the data type
                to reduce memory usage.        
            """
            start_mem = df.memory_usage().sum() / 1024**2
            for col in df.columns:
                col_type = df[col].dtype
                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)  
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
            end_mem = df.memory_usage().sum() / 1024**2
            return df
        df_train2=reduce_mem_usage(df_train2)
        target='PRODUCCION'
        df_train_pt4=df_train2.copy()
        df_train_pt4=df_train_pt4[df_train_pt4['CAMPAÑA'].isin([17,18,19,20,21,22])].reset_index(drop=True)
        ss = StandardScaler()
        df_scaled = pd.DataFrame(ss.fit_transform(df_train_pt4[features]),columns = features)
        df_scaled=pd.concat([df_train_pt4[['ID_KEY_PROD','CAMPAÑA','PRODUCCION']],df_scaled],axis=1)

        # seleccionar características
        test = df_scaled[df_scaled["CAMPAÑA"] == 22].reset_index(drop = True)
        train = df_scaled[df_scaled["CAMPAÑA"] < 22].reset_index(drop = True)
        
        # hacer predicciones
        predictions = []
        for model in models:
            prediction = model.predict(test[features])
            predictions.append(prediction)
            
        # tomar el promedio de las predicciones de todos los modelos
        test["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
        test_subido["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
        predictions_mean = pd.DataFrame(np.abs(np.mean(predictions, axis=0)))

        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)
        prod_ant = np.round(np.mean(train.groupby(["CAMPAÑA"]).sum()["PRODUCCION"]),2)
        prod_fut = np.round(np.mean(test.groupby(["CAMPAÑA"]).sum()["predicted_tot"]),2)
        incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
        col1.metric("Producción media campañas pasadas", "{} kg".format(prod_ant))
        col2.metric("Producción esperada próxima campaña", "{} kg".format(prod_fut), "{}%".format(incremento))

        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("### Mostrar resultados individuales:")

        valor_filtro = st.text_input('A continuación obtendrás las predicciones (en kg) de la producción de cada una de las observaciones del CSV que has introducido. Escoge cuántas predicciones quieres mostrar:')
        
        if valor_filtro != '':
            valor_filtro = np.int(valor_filtro)
            ####### ELEGIR CUÁNTOS RESULTADOS MOSTRAR
            for i in range(valor_filtro):
                st.write("Se prevé que la finca {} produzca {}kg de uva de variedad {}, color {} y modo {}.".format(test_subido["ID_FINCA"][i], np.round(test_subido["predicted_tot"][i], 2), test_subido["VARIEDAD"][i], test_subido["COLOR"][i], test_subido["MODO"][i]))
            if predictions_mean.shape[0]>valor_filtro:
                st.write("[...]")
                st.write("Se han mostrado {} de {} predicciones.".format(valor_filtro,predictions_mean.shape[0]))
            else:
                st.write("Se han mostrado las {} predicciones".format(test_subido.shape[0]))
            ##################################################
       
        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("### Gráficos de la distribución de la producción pasada y predicha:")
        # Añadimos la columna indicando si son datos reales o predicción
        train_original['tipo_dato'] = 'Datos Reales'
        test_subido['tipo_dato'] = 'Predicción'
        test_subido['PRODUCCION'] = test_subido['predicted_tot']
        
        # Concatenamos ambos dataframes
        df_train_pt4['tipo_dato'] = np.where(df_train_pt4['CAMPAÑA'] == 22, 'Predicción', 'Datos Reales')
        
        train = df_train2[df_train2["CAMPAÑA"] < 22].reset_index(drop = True).copy()
        test = df_train2[df_train2["CAMPAÑA"] == 22].reset_index(drop = True).copy()
        test['PRODUCCION'] = test_subido['predicted_tot']
               
        import plotly.express as px
        import plotly.graph_objects as go
                
        # Define los colores para la escala de color
        colors = ['#7e5bb5', '#c43d2e']
        
        # Define los nombres de las columnas que pueden usarse como variable categórica
        opciones = ['MODO', 'COLOR', 'TIPO']
        
        # Añade un desplegable en Streamlit para que el usuario pueda elegir la variable categórica
        opcion = st.selectbox('Agrupar por:', opciones)
        
        train = train.sort_values(by = ["CAMPAÑA", opcion]).reset_index(drop = True)
        test = test.sort_values(by = ["CAMPAÑA", opcion]).reset_index(drop = True)
        
        # Crea las figuras
        fig_train = px.bar(train, x="CAMPAÑA", y="PRODUCCION", color=opcion,
                           barmode="stack", title="Producción en función de " + opcion,
                           color_continuous_scale=colors)
        fig_train.update_layout(height=600, width=600, 
                                margin=dict(l=50, r=50, t=80, b=50),
                                showlegend=True, legend=None,
                                barmode="stack",
                                xaxis_title="CAMPAÑA", yaxis_title="Producción")
        fig_train.update_traces(marker=dict(line=dict(width=0, color='Black')),
                                opacity=0.8)
        
        fig_test = px.bar(test, x="CAMPAÑA", y="PRODUCCION", color=opcion,
                          barmode="stack", title="",
                          color_continuous_scale=colors)
        fig_test.update_layout(height=600, width=200, 
                               margin=dict(l=20, r=50, t=80, b=50),
                               showlegend=True, legend=dict(orientation="h", y=1.1),
                               barmode="stack",
                               plot_bgcolor='#F8D7DA', # Reddish background color
                               xaxis_title="", yaxis_title="Predicción de la producción")
        fig_test.update_traces(marker=dict(line=dict(width=0, color='Black')),
                               opacity=0.8)
        
        # Muestra las figuras en Streamlit
        col1, col2 = st.columns((7, 3))
        with col1:
            st.plotly_chart(fig_train)
        with col2:
            st.plotly_chart(fig_test)
         
        st.markdown("<hr>", unsafe_allow_html=True)      

        st.write("### Gráfico de la producción prevista y los topes de la Denominación de Origen.")
        st.write("En el siguiente gráfico podemos observar si se prevé que muchas de las fincas superen, o no, los límites de producción establecidos por la Denominación de Orígen (DO). En este caso, la Denominación de Origen de Valencia establece que el límite de producción para las fincas de uva blanca es de 12000kg/Ha y para las fincas de uva tinta 9100kg/Ha")
        ## Gráfico para comprobar los topes de la DO
        test = df_train2[df_train2["CAMPAÑA"] == 22].reset_index(drop = True).copy()
        test['PRODUCCION'] = test_subido['predicted_tot']
        test["pred_prod_ha"] = test["PRODUCCION"]/test["superficie_imp"]
        test = test[test["superficie_imp"]>1]
        
        x_jitter = np.random.normal(0, 0.1, size=len(test['COLOR']))
                
        fig = px.scatter(test, x=test['COLOR']+x_jitter, y='pred_prod_ha', color='CAMPAÑA',
                         color_continuous_scale=["#c43d2e", "#c43d2e"], opacity=0.7)
        
        fig.add_trace(go.Scatter(x=test['COLOR'], y=[9100]*(len(test))*2, name='Límite uva tinta', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=test['COLOR'], y=[12000]*(len(test))*2, name='Límite uva blanca', line=dict(color='green')))
        
        fig.update_layout(title='Gráfico para comprobar los topes de la DO Valencia',
                          xaxis_title='Color',
                          yaxis_title='Producción por hectárea',
                          showlegend=True,
                          legend=dict(yanchor="top", y=1.05, xanchor="left", x=0.01),
                          margin=dict(l=50, r=50, t=80, b=50),
                          hovermode='closest',
                          coloraxis_showscale=False, #Quita la barra lateral
                          yaxis=dict(range=[-1000, 14000])) #Ajusta el límite del eje y
        
        fig.update_traces(marker=dict(size=6, line=dict(width=1, color='Black')),
                          hovertemplate='<b>CAMPAÑA %{customdata}</b><br>' +
                                        'Color: %{x:.2f}<br>' +
                                        'Producción por hectárea: %{y:.2f}<br>' +
                                        '<extra></extra>',
                          customdata=test['CAMPAÑA'])
        
        st.plotly_chart(fig)



# if forma_intro == "Manualmente":
#     st.write("hola")
# elif forma_intro == "Subir CSV":
#     if (uploaded_file is not None):
#         # seleccionar características
#         test = df_scaled[df_scaled["CAMPAÑA"] == 22].reset_index(drop = True)
#         train = df_scaled[df_scaled["CAMPAÑA"] < 22].reset_index(drop = True)
        
#         # hacer predicciones
#         predictions = []
#         for model in models:
#             prediction = model.predict(test[features])
#             predictions.append(prediction)
            
#         # tomar el promedio de las predicciones de todos los modelos
#         test["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
#         test_subido["predicted_tot"] = np.abs(np.mean(predictions, axis=0))
#         predictions_mean = pd.DataFrame(np.abs(np.mean(predictions, axis=0)))
        
# # Dividir la página en dos columnas
# col1, col2 = st.columns(2)
# prod_ant = np.round(np.mean(train.groupby(["CAMPAÑA"]).sum()["PRODUCCION"]),2)
# prod_fut = np.round(np.mean(test.groupby(["CAMPAÑA"]).sum()["predicted_tot"]),2)
# incremento = np.round((prod_fut - prod_ant) / prod_ant * 100, 2)
# col1.metric("Producción media campañas pasadas", "{} kg".format(prod_ant))
# col2.metric("Producción esperada próxima campaña", "{} kg".format(prod_fut), "{}%".format(incremento))

# if forma_intro == "Manualmente":
#     st.write("hola")
# elif forma_intro == "Subir CSV":
#     if uploaded_file is not None:
#         valor_filtro = st.text_input('A continuación obtendrás las predicciones (en kg) de la producción de cada una de las observaciones del CSV que has introducido. Escoge cuántas predicciones quieres mostrar:')
        
#         if valor_filtro != '':
#             valor_filtro = np.int(valor_filtro)
#             ####### ELEGIR CUÁNTOS RESULTADOS MOSTRAR
#             for i in range(valor_filtro):
#                 st.write("Se prevé que la finca {} produzca {}kg de uva de variedad {}, color {} y modo {}.".format(test_subido["ID_FINCA"][i], np.round(test_subido["predicted_tot"][i], 2), test_subido["VARIEDAD"][i], test_subido["COLOR"][i], test_subido["MODO"][i]))
#             if predictions_mean.shape[0]>valor_filtro:
#                 st.write("[...]")
#                 st.write("Se han mostrado {} de {} predicciones.".format(valor_filtro,predictions_mean.shape[0]))
#             else:
#                 st.write("Se han mostrado las {} predicciones".format(test_subido.shape[0]))
#             ##################################################


# import plotly.graph_objs as go
# import plotly.offline as pyo

# train["MODO"] = df_train2[df_train2["CAMPAÑA"]<22]["MODO"].reset_index(drop=True).copy()
# test["MODO"] = df_train2[df_train2["CAMPAÑA"] == 22]["MODO"].reset_index(drop=True).copy()

# # Plot 1
# trace1 = go.Bar(
#     x=train.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack().index,
#     y=train.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack()[['D', 'T']].sum(axis=1),
#     name='D',
#     marker=dict(color='#78C15A')
# )

# trace2 = go.Bar(
#     x=train.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack().index,
#     y=train.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack()[['R', 'S']].sum(axis=1),
#     name='S',
#     marker=dict(color='#900C3F')
# )

# data1 = [trace1, trace2]

# layout1 = go.Layout(
#     title='Producción en función de MODO',
#     xaxis=dict(title='CAMPAÑA'),
#     yaxis=dict(title='Producción'),
#     barmode='stack',
#     paper_bgcolor='#D1E9D1',  # Greenish background color
#     plot_bgcolor='#D1E9D1',  # Greenish background color
# )

# fig1 = go.Figure(data=data1, layout=layout1)

# # Plot 2
# trace3 = go.Bar(
#     x=test.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack().index,
#     y=test.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack()[['D', 'T']].sum(axis=1),
#     name='D',
#     marker=dict(color='#78C15A')
# )

# trace4 = go.Bar(
#     x=test.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack().index,
#     y=test.groupby(['CAMPAÑA', 'MODO'])['PRODUCCION'].sum().unstack()[['R', 'S']].sum(axis=1),
#     name='S',
#     marker=dict(color='#900C3F')
# )

# data2 = [trace3, trace4]

# layout2 = go.Layout(
#     xaxis=dict(title='CAMPAÑA'),
#     yaxis=dict(title='Predicción de la producción'),
#     barmode='stack',
#     paper_bgcolor='#F8D7DA',  # Reddish background color
#     plot_bgcolor='#F8D7DA',  # Reddish background color
# )

# fig2 = go.Figure(data=data2, layout=layout2)

# fig = go.FigureWidget(fig1)

# pyo.iplot([fig1, fig2])
