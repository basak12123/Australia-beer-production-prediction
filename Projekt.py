import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import arma_order_select_ic
import pyreadr
import plotly.graph_objects as go
import matplotlib.dates as mdates
from dm_test import dm_test
from scipy import stats

# Ustawienia stylu podobne do ggplot2


plt.rc(
    "figure",
    autolayout=True,
    figsize=(11.69,8.27),
    titlesize=10,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
sns.set_theme(style='whitegrid',
              context='paper',
              palette='dark')



def ts_plot(ts):
    # Dodanie kolumny z kwartałem do hovertemplate
    ts['quarter'] = ts.iloc[:, 0].dt.to_period('Q').astype(str)

    fig = go.Figure()

    # Dodanie linii z markerami i szablonu hover
    fig.add_trace(go.Scatter(
        x=ts.iloc[:, 0],
        y=ts.iloc[:, 1],
        mode='lines+markers',  # Linie z markerami
        line=dict(color='royalblue', width=3),
        marker=dict(size=6, color='darkblue', symbol='circle'),  # Styl markerów
        hovertemplate=(
            "Data: %{x}<br>"
            "Produkcja: %{y}<br>"
            "Kwartał: %{customdata}<extra></extra>"
        ),  # Własny szablon z kwartałem
        customdata=ts['quarter']  # Przekazanie informacji o kwartale
    ))

    # Konfiguracja osi X (tylko lata)
    fig.update_xaxes(
        dtick="M12",  # Podziałka co 12 miesięcy (rok)
        tickformat="%Y",  # Format: Rok
        tickangle=0,  # Prostoliniowe etykiety
        showgrid=True,  # Widoczność siatki
        gridcolor='lightgrey',
        gridwidth=0.5  # Cieńsza siatka
    )

    # Konfiguracja osi Y
    fig.update_yaxes(
        showgrid=True,  # Widoczność siatki
        gridcolor='whitesmoke',  # Delikatniejszy kolor siatki
        gridwidth=0.5  # Cieńsza siatka
    )

    # Dodanie pionowych linii (początek roku)
    start_years = ts.iloc[:, 0].dt.to_period('Y').unique()  # Ekstrakcja unikalnych lat
    for year in start_years:
        fig.add_vline(
            x=str(year) + "-01-01",  # Początek roku
            line_width=0.5,
            line_dash="dash",
            line_color="grey",
        )

    # Tytuły i motyw
    fig.update_layout(
        title="Produkcja piwa w Australii",
        xaxis_title="Data",
        yaxis_title="Hektolitry",
        template="plotly_white",
        title_font=dict(size=20, family='Arial', color='darkblue'),
    )

    # Dodanie suwaka z tłem
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, bgcolor='lightgrey')  # Tło suwaka
        )
    )

    # Wyświetlenie wykresu
    fig.show()



pd.set_option('display.max_columns', None)

data = sm.datasets.get_rdataset("ausbeer", package="fpp2").data

#print(data.columns)
# Mamy dwie kolumny: time i value




# Funkcja do mapowania kwartałów na miesiące
def convert_to_datetime(row):
    year, quarter = str(row).split('.')
    quarter_to_month = {'0': '01', '25': '04', '5': '07', '75': '10'}
    month = quarter_to_month[quarter]
    return f"{year}-{month}-01"

data['time'] = pd.to_datetime(data['time'].apply(convert_to_datetime))
data = data.loc[data['time'] >= pd.Timestamp('2000-01-01')]
#ts_plot(data)

#print(data.value.describe())
'''
Średnia wartość: 415
STD value: 37
MIN: 374
MAX: 506
Dane w milionach hektolitrów? Potwierdzić
'''

####### Wykres danych
'''
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='time', y='value', color='darkblue', marker='o')
plt.title('Produkcja piwa w Australii')
plt.xlabel('Lata')
plt.ylabel('Megalitry')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Znaczniki co rok
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
#plt.savefig('wykres1.png', format='png', bbox_inches='tight')
#plt.show()
'''

# Dekompozycja, przyjęty model addytywny z okresem 4
# Odnośnie filtrów w średniej kroczącej. Nadanie 5*[1/5] bądź 4*[1/4] daje gorsze wyniki niż domyślny, zaimplementowany
# w funkcji: filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period

decomposition = seasonal_decompose(data['value'], model='additive', period=4, extrapolate_trend='freq')
decompositionm = seasonal_decompose(data['value'], model='multiplicative', period=4, extrapolate_trend='freq')

### Porównanie szumu uzyskanego za pomocą dekompozycji addytywnej vs multiplikatywnej
'''
fig, ax = plt.subplots(2,1, sharex=True)
fig.suptitle('Porównanie szumu M vs A')
sns.lineplot(decomposition.resid,ax=ax[0], color='black')
ax[0].set_title('A')
sns.lineplot(decompositionm.resid,ax=ax[1], color='black')
ax[1].set_title('M')
plt.tight_layout()
plt.show()
'''

### Bardzo podobne
'''
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('Porównanie ACF')
plot_acf(decomposition.resid, ax=ax[0], color='darkgreen', lags=41)
ax[0].set_title('Dekompozycja addytywna')
plot_acf(decompositionm.resid, ax=ax[1], color='darkgreen', lags=41)
ax[1].set_title('Dekompozycja multiplikatywna')
plt.tight_layout()
plt.show()
'''


### Test Ljung-Boxa / losowości reszt
LBTestA = sm.stats.acorr_ljungbox(decomposition.resid - np.mean(decomposition.resid), auto_lag=True)
LBTestM = sm.stats.acorr_ljungbox(decompositionm.resid - np.mean(decompositionm.resid), auto_lag=True)
#print(LBTestA)
#print(LBTestM)


### Test Shapiro - Wilka / normalności reszt
SWTestA = stats.shapiro(decomposition.resid)
SWTestM = stats.shapiro(decompositionm.resid)
#print(SWTestA)
#print(SWTestM)



### WNIOSEK: Model addytywny (?)
SSM1 = np.sum((decomposition.trend + decomposition.seasonal - data.value.mean())**2)
SSM1M = np.sum((decomposition.trend + decomposition.seasonal - data.value.mean())**2)
SST = np.sum((data.value - data.value.mean())**2)

R1 = SSM1/SST
#print(f'R^2={R1}')


### BOX-COX i regresja liniowa

def estimate_season(x, n=42):
    g_avg = [x[::4].mean(), x[1::4].mean(), x[2::4].mean(), x[3::4].mean()]
    season = np.tile(g_avg, n // len(g_avg) + 1)[:n]
    return season

logdata = np.log(data.value)
t = np.arange(42)
reg = sm.OLS(logdata, sm.add_constant(t)).fit()
log_trend = reg.fittedvalues
x = logdata - reg.fittedvalues
log_season = estimate_season(x)
log_noise = logdata - log_trend - log_season

reg_trend = np.exp(log_trend)
reg_season = np.exp(log_season)
reg_noise = np.exp(log_noise)

LBTestReg = sm.stats.acorr_ljungbox(reg_noise - np.mean(reg_noise), auto_lag=True)
SWTestReg = stats.shapiro(reg_noise)
SSM2 = np.sum((reg_trend*reg_season - data.value.mean())**2)
R2 = SSM2/SST
'''
print(f'Transformacja Box-Cox i estymacja trendu regresją liniową:'
      f'R^2 = {R2}\n'
      f'LJUNG-BOX = {LBTestReg}\n'
      f'SHAPIRO-WILK = {SWTestReg}\n')
'''



### model wielomianowy
k = [2,3,4,5,6, 7, 8]
coef_polyreg = {d: np.polyfit(t, logdata, d, full=True)[0] for d in k}
polyreg = {d: {'trend': np.exp(np.polyval(coef_polyreg[d], t)),
               'season': np.exp(estimate_season(logdata - np.polyval(coef_polyreg[d], t))),
               'noise': np.exp(logdata - np.polyval(coef_polyreg[d], t) - estimate_season(logdata - np.polyval(coef_polyreg[d], t)))} for d in k}

for d in k:
    LBTest = sm.stats.acorr_ljungbox(polyreg[d]['noise'] - np.mean(polyreg[d]['noise']), auto_lag=True)
    SWTest = stats.shapiro(polyreg[d]['noise'])
    SSM = np.sum((polyreg[d]['trend'] * polyreg[d]['season'] - data.value.mean()) ** 2)
    R = SSM / SST
    print(f'Transformacja Box-Cox i estymacja trendu regresją wielomianową dla stopnia {d} \n:'
          f'R^2 = {R}\n'
          f'LJUNG-BOX = \n{LBTest}\n'
          f'SHAPIRO-WILK = \n{SWTest}\n')


### Wielomian stopnia 3 uzyskuje najlepsze wyniki
d = 3
fig, ax = plt.subplots(4,1, sharex=False)
fig.suptitle('Dekompozycja Beer Production')
sns.lineplot(data['value'],ax=ax[0], color='black')
ax[0].set_title('Pierwotny szereg')
sns.lineplot(polyreg[d]['trend'],ax=ax[1], color='darkred')
ax[1].set_title('Składowa trendu')
sns.lineplot(polyreg[d]['season'],ax=ax[2], color='darkgreen')
ax[2].set_title('Składowa sezonowa')
sns.lineplot(polyreg[d]['noise'], ax=ax[3], color='darkblue')
ax[3].set_title('Szum')
plt.savefig('wykres2.png', format='png', bbox_inches='tight')
plt.tight_layout()
plt.show()


### regresja nielokalna
'''
lowess_reg = sm.nonparametric.lowess
k = [3/4, 2/3, 1/4, 1/3, 1/6, 1/8]

localreg_trend = {d: lowess_reg(data.value, t, frac=d)[:,1] for d in k}
localreg_season = {d: estimate_season(data.value - localreg_trend[d]) for d in k}
localreg_noise = {d: data.value - localreg_trend[d] - localreg_season[d] for d in k}

for d in k:
    LBTest = sm.stats.acorr_ljungbox(localreg_noise[d] - np.mean(localreg_noise[d]), auto_lag=True)
    SWTest = stats.shapiro(localreg_noise[d])
    SSM = np.sum((localreg_trend[d] + localreg_season[d] - data.value.mean()) ** 2)
    R = SSM / SST
    print(f'Estymacja trendu regresją lokalną dla stopnia {d} \n:'
          f'R^2 = {R}\n'
          f'LJUNG-BOX = \n{LBTest}\n'
          f'SHAPIRO-WILK = \n{SWTest}\n')
'''
### eksponencjalna?


