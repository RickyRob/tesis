import pandas as pd
import yfinance as yf
import numpy as np
import xlwings as xw
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go

# Funcion de bienvenida

def bienvenida():
    print('\n')
    print('####################################################')
    print('################# RICKY INVESTING ##################')
    print('####################################################')
    print('\n')

# Funcion leer data

### Esta funcion importa los datos de Yahoo Finance y retorna un dataframe limpio con solo los
#Datos de cierre ajustado

def data(ticker:str,f:str):
    df = yf.download(ticker,end=f,interval='1m')
    df.drop(['Open','High','Low','Close','Volume'], axis = 1, inplace=True)
    df.dropna(inplace=True)
    return df

"""
Esta parte del código descarga los datos de 7 días con periocidad diaria
para calcular los precios de apertura y cierre de un día
"""

def data2(ticker:str,s:str,f:str):
    df2 = yf.download(ticker,start=s,end=f)
    df2.reset_index(inplace=True)
    aper = df2['Close'][:-1]
    aper = aper.reset_index()
    aper.drop('index',axis=1,inplace=True)

    cier = df2['Open'][1:]
    cier = cier.reset_index()
    cier.drop('index',axis=1,inplace=True)

    fecha = df2['Date'][1:]
    fecha = fecha.reset_index()
    fecha.drop('index',axis=1,inplace=True)

    df_tab = pd.DataFrame()
    df_tab['Fecha'] = fecha
    df_tab['Cierre d-1'] = aper
    df_tab['Apertura'] = cier

    df_tab['Diff'] = df_tab['Cierre d-1'] - df_tab['Apertura']
    df_tab['Per'] = (df_tab['Diff']/df_tab['Cierre d-1'])*100

    #aper
    #df2.drop(['Open','High','Low','Close','Volume'], axis = 1, inplace=True)
    #df2.dropna(inplace=True)
    return df_tab

### GRAFICA DE CAMBIO DE PRECIO 

def grafico1(d1):
    v1 = np.round(d1.iloc[-1,-1],2)
    v2 = np.round(d1.iloc[-2,-1],2)
    r = (1 - (v2/v1))
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = v1,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Impulso", 'font': {'size': 30}},
    delta = {'reference': r, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [-3, 3], 'tickwidth': 5, 'tickcolor': "darkblue",'ticklen':5},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [-3, 0], 'color': 'cyan'},
            {'range': [0, 1.5], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': v1}}))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    fig.show()

## GRAFICA DE IMPULSO
def grafico1(d1):
    v1 = np.round(d1.iloc[-1,-1],2)
    v2 = np.round(d1.iloc[-2,-1],2)
    r = (1 - (v2/v1))
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = v1,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Impulso", 'font': {'size': 30}},
    delta = {'reference': r, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [-3, 3], 'tickwidth': 5, 'tickcolor': "darkblue",'ticklen':5},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [-3, 0], 'color': 'cyan'},
            {'range': [0, 1.5], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': v1}}))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    fig.show()
