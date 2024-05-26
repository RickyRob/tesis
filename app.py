### Este es el archivo madre del proyecto 01
from modules.funciones import *
from datetime import datetime
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')
bienvenida()

t = input('Nombre del ticket: ').upper()
hoy= datetime.now()
df = data(ticker=t,f=hoy)

df_tab = data2(ticker=t,s=hoy+timedelta(days=-10),f=hoy)

print(df_tab.tail())
#grafico1(d1=df_tab)
df3 = voldia(df,ticker=t,df2=df_tab)

prediccion = red(df3=df3)

graficofinal(pr=prediccion)

