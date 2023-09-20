import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("demand_inventory.csv")
print(data.head())

data.drop(['Unnamed: 0'],axis=1,inplace=True)

#kysynnän kuvaaja
kysyntä = px.line(data, x='Date', y='Demand', title='Kysyntä')
kysyntä.show()

#varaston kuvaaja
varasto = px.line(data, x='Date', y='Inventory', title='Varastot')
varasto.show()

#kysynnän ennustaminen
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
time_series = data.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()

#ACF ja PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

#ennustus malli seuraavat 15 päivää (SARIMAX koska kysynnässä vähän kausittaisuutta)
_order = (1,1,1)
seasonal_order = (1,1,1,2)#2 koska datassa vain kahden kuukauden tiedot
malli = SARIMAX(time_series, order=_order, seasonal_order=seasonal_order)
malli_fit = malli.fit(disp=False)

ennuste = malli_fit.predict(len(time_series), len(time_series)+15-1)
ennusteet = ennuste.astype(int)
print(ennusteet)

#varaston optimointi
tulevat_paivat = pd.date_range(start=time_series.index[-1]+pd.DateOffset(days=1), periods=15, freq='D')

ennustettu_kysyntä = pd.Series(ennusteet, index=tulevat_paivat)

ap_varasto = 5500

varaston_taydennysaika = 1

palvelutaso = 0.95

#optimaalinen tilausmäärä Newsvendorin kaavalla
z = np.abs(np.percentile(ennustettu_kysyntä, 100 * (1-palvelutaso)))
tilausmaara = np.ceil(ennustettu_kysyntä.mean()+z).astype(int)

tilauspiste = ennustettu_kysyntä.mean()*varaston_taydennysaika + z

varmuusvarasto = tilauspiste - ennustettu_kysyntä.mean() * varaston_taydennysaika

#kokonaiskustannukset
pitokustannukset = 0.1
varastokustannukset = 10
kokonais_pitokustannukset = pitokustannukset * (ap_varasto + 0.5 * tilausmaara)
kokonais_varastokustannukset = varastokustannukset * np.maximum(0, ennustettu_kysyntä.mean()*varaston_taydennysaika-ap_varasto)

kokonaiskustannukset = kokonais_pitokustannukset + kokonais_varastokustannukset

print('Optimaalinen tilausmäärä:', tilausmaara)
print("Tilauspiste:", tilauspiste)
print("Varmuusvarasto:", varmuusvarasto)
print("Kokonaiskustannukset", kokonaiskustannukset)
