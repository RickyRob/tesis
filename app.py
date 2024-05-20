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
print(np.round(df_tab.iloc[-2,-1],2))
print(np.round(df_tab.iloc[-1,-1],2))

grafico1(d1=df_tab)
