import pandas as pd
import yfinance as yf
import numpy as np
import xlwings as xw
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


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

def voldia(df,ticker,df2):
    f = lambda x: list(x.index.date)
    l1 = f(df)
    l2 = list(map(lambda x : x.day, l1))
    df['day'] = l2
    r = df['Adj Close'].diff()/df['Adj Close']
    df['rend'] = r
    #df['Vol'] = np.var(df['rend'])
    print(df.head())
    #print(np.var(df['rend']))
    #agru = list(df.groupby('day'))
    #vola = map(lambda x : np.var(x['rend']),agru)
    #print(list(vola).reset_index)

    g = pd.DataFrame(list(df.groupby('day')))
    #pl1,pl2,pl3,pl4,pl5 = df.groupby('day').plot(y='Adj Close',title=f'Precios por minuto de: {ticker}')
    #gra = df.groupby('day').plot(y='Adj Close',title=f'Precios por minuto de: {ticker}')
    #vol1,vol2,vol3,vol4,vol5 = df.groupby('day').plot(y='rend',title=f'Rendimientos por minuto de: {ticker}')
    print('################')
    fig, (ax3,ax4,ax5) = plt.subplots(3, 1)
    fig.suptitle(f'Precios por minuto de: {ticker}')
    pl1 = g[1][0]
    pl2 = g[1][1]
    pl3 = g[1][2]
    pl4 = g[1][3]
    pl5 = g[1][4]
    #ax1.plot(pl1['Adj Close'])
    #ax2.plot(pl2['Adj Close'])
    ax3.plot(pl3['Adj Close'])
    ax4.plot(pl4['Adj Close'])
    ax5.plot(pl5['Adj Close'])
    #########print('################')
    fig2, (a3,a4,a5) = plt.subplots(3, 1)
    fig2.suptitle(f'Rendimientos por minuto de: {ticker}')
    #a1.plot(pl1['rend'])
    #a2.plot(pl2['rend'])
    a3.plot(pl3['rend'])
    a4.plot(pl4['rend'])
    a5.plot(pl5['rend'])
    ###########################
    var1 = np.sqrt(np.var(pl1['rend']))*100
    var2 = np.sqrt(np.var(pl2['rend']))*100
    var3 = np.sqrt(np.var(pl3['rend']))*100
    var4 = np.sqrt(np.var(pl4['rend']))*100
    var5 = np.sqrt(np.var(pl5['rend']))*100
    des = [var1,var2,var3,var4,var5]
    fig3, (d1,d2) = plt.subplots(2, 1)
    fig3.suptitle(f'{ticker}')
    d1.plot(des,marker='o')
    d1.set_ylabel('Volatilidad diaria %')
    d2.plot(df2['Per'], marker='o')
    d2.set_ylabel('Cierre - Apertura $')

    # Sacando los minimos y maximos del rendimiento diario
    min1 = np.min(np.min(pl1['rend']))
    max1 = np.min(np.max(pl1['rend']))
    min2 = np.min(np.min(pl2['rend']))
    max2 = np.min(np.max(pl2['rend']))
    min3 = np.min(np.min(pl3['rend']))
    max3 = np.max(np.max(pl3['rend']))
    min4 = np.max(np.min(pl4['rend']))
    max4 = np.max(np.max(pl4['rend']))
    min5 = np.max(np.min(pl5['rend']))
    max5 = np.max(np.max(pl5['rend']))

    # Creando el df de salida para las redes neuronales

    df3 = df2.copy()
    df3.drop(['Cierre d-1','Apertura','Per'],axis=1, inplace=True)
    df3.drop([0],axis=0,inplace=True)
    minimos = [min1,min2,min3,min4,min5]
    maximos = [max1,max2,max3,max4,max5]
    df3['renMin'] = minimos
    df3['renMax'] = maximos
    df3['volProm'] = des

    print(df3.head())
    
    
    plt.show()
    return df3

def red(df3):
    print("###############################")
    print("###############################")
    print("###############################")
    print('Deseas hacer una predicción de la volatilidad???')
    pre = input('Respuesta Y/N: ').upper()
    if pre == 'Y':
        promMin = df3['renMin'].mean()
        promMax = df3['renMax'].mean()
        #features = ['Diff','renMin','renMax']
        features = ['Diff','renMax']
        target = ['volProm']
        x = df3[features]
        y = df3[target]
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        scale = preprocessing.StandardScaler()
        scale.fit(X_train)
        X_train = scale.transform(X_train)
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)
        X_test = scale.transform(X_test)
        y_predict = regressor.predict(X_test)
        #y_result = y_predict - Y_test
        #s = regressor.score(X_test, Y_test)
        #print(y_result)
        #print(y_predict.shape)
        #print(s)
        #print(y_predict)
        varia = float(input('Valor de la variación: '))
        #varia = float(varia)
        prueba = [[varia,promMax]]
        prueba = scale.transform(prueba)
        prediccion = regressor.predict(prueba)
        prediccion = prediccion[0][0]
        print(f'Tu predicción es: {prediccion}')
        #r = regressor.predict(3,promMin,promMax)
        #print(r)
    else:
        print('No has querido predecir')
    return prediccion
    

## GRAFICA DE IMPULSO
def graficofinal(pr):
    r = pr
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = pr,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Impulso", 'font': {'size': 30}},
    delta = {'reference': r, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [-1, 1], 'tickwidth': 5, 'tickcolor': "darkblue",'ticklen':5},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [-1, 0], 'color': 'cyan'},
            {'range': [0, 0.75], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': pr}}))

    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    fig.show()

