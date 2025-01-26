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
from scipy.special import  inv_boxcox


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
data = data.loc[data['time'] >= pd.Timestamp('1990-01-01')]
#ts_plot(data)


#print(data.value.describe())
'''
Liczba: 82
Średnia wartość: 438
STD value: 47
MIN: 374
MAX: 599
MEDIAN: 422
Dane w milionach hektolitrów? Potwierdzić
'''
n=82

####### Wykres danych

plt.figure(figsize=(8, 5))
sns.lineplot(data=data, x='time', y='value', color='darkblue', marker='o')
plt.title('Produkcja piwa w Australii', loc='center')
plt.xlabel('Lata')
plt.ylabel('Megalitry')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Znaczniki co rok
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('wykres1.png', format='png', bbox_inches='tight')
plt.show()


def mean_outlier(data, n=2, indexes=[3, 7, 59]):
    result = data.copy()
    means = np.convolve(data, np.ones(n) / n, mode='valid')
    result.iloc[indexes] = means[indexes]
    return result

# Dekompozycja, przyjęty model addytywny z okresem 4
# Odnośnie filtrów w średniej kroczącej. Nadanie 5*[1/5] bądź 4*[1/4] daje gorsze wyniki niż domyślny, zaimplementowany
# w funkcji: filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period

decomposition = seasonal_decompose(data['value'], model='additive', period=4, extrapolate_trend='freq')
decompositionm = seasonal_decompose(data['value'], model='multiplicative', period=4, extrapolate_trend='freq')

### Po analizie outliers
decomposition_resid = mean_outlier(decomposition.resid)

'''
### Porównanie szumu uzyskanego za pomocą dekompozycji addytywnej vs multiplikatywnej
fig, ax = plt.subplots(2,1, sharex=True)
fig.suptitle('Porównanie szumu M vs A')
sns.lineplot(decomposition.resid,ax=ax[0], color='black')
ax[0].set_title('A')
sns.lineplot(decompositionm.resid,ax=ax[1], color='black')
ax[1].set_title('M')
plt.tight_layout()
plt.show()
'''

'''
### Dekompozycja
fig, ax = plt.subplots(4,1, sharex=False, figsize=(8,5))
fig.suptitle('Dekompozycja Beer Production')
sns.lineplot(data['value'],ax=ax[0], color='black')
sns.lineplot(decomposition.trend + decomposition.seasonal, ax=ax[0], color='red', alpha=0.5)
ax[0].set_title('Pierwotny szereg')
sns.lineplot(decomposition.trend,ax=ax[1], color='darkred')
ax[1].set_title('Składowa trendu')
sns.lineplot(decomposition.seasonal,ax=ax[2], color='darkgreen')
ax[2].set_title('Składowa sezonowa')
sns.lineplot(decomposition_resid, ax=ax[3], color='darkblue')
ax[3].set_title('Szum')
plt.tight_layout()
plt.show()



### Bardzo podobne
### ACF
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
fig.suptitle('Porównanie ACF')
plot_acf(decomposition_resid, ax=ax[0], color='darkgreen', lags=41)
ax[0].set_title('Dekompozycja addytywna')
plot_acf(decompositionm.resid, ax=ax[1], color='darkgreen', lags=41)
ax[1].set_title('Dekompozycja multiplikatywna')
plt.tight_layout()
plt.show()

### PACF
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
fig.suptitle('Porównanie ACF')
plot_pacf(decomposition_resid, ax=ax[0], color='darkgreen', lags=41)
ax[0].set_title('Dekompozycja addytywna')
plot_pacf(decompositionm.resid, ax=ax[1], color='darkgreen', lags=41)
ax[1].set_title('Dekompozycja multiplikatywna')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8,5))
sm.qqplot(decomposition_resid, line='45', fit=True, ax=ax)
plt.suptitle('Wykres qqplot dla reszt')
plt.tight_layout()
#plt.savefig('wykres5.png', format='png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(np.arange(n), decomposition_resid)
ax.scatter([3,7,59], decomposition.resid.reset_index().iloc[[3,7,59],1], color='red')
plt.suptitle(f'Wartości dla których Z > 2.9')
plt.tight_layout()
#plt.savefig('wykres6.png', format='png', bbox_inches='tight')
plt.show()
'''


### Test Ljung-Boxa / losowości reszt
LBTestA = sm.stats.acorr_ljungbox(decomposition_resid, lags=[4, 8, 12])
LBTestM = sm.stats.acorr_ljungbox(decompositionm.resid, lags=[4, 8, 12])

### Test Shapiro - Wilka / normalności reszt
SWTestA = stats.shapiro(decomposition_resid)
SWTestM = stats.shapiro(decompositionm.resid)

SSM1 = np.sum((decomposition.trend + decomposition.seasonal - data.value.mean())**2)
SSM1M = np.sum((decompositionm.trend*decompositionm.seasonal - data.value.mean())**2)
SST = np.sum((data.value - data.value.mean())**2)
R1 = SSM1/SST


print(f'Estymacja średnią kroczącą zakładając model addytywny:'
      f'R^2 = {R1}\n'
      f'LJUNG-BOX = {LBTestA}\n'
      f'SHAPIRO-WILK = {SWTestA}\n')

print(f'Estymacja średnią kroczącą zakładając model multiplikatywny:'
      f'R^2 = {SSM1M/SST}\n'
      f'LJUNG-BOX = {LBTestM}\n'
      f'SHAPIRO-WILK = {SWTestM}\n')



### BOX-COX i regresja liniowa

def estimate_season(x, n=82):
    g_avg = [x[::4].mean(), x[1::4].mean(), x[2::4].mean(), x[3::4].mean()]
    season = np.tile(g_avg, n // len(g_avg) + 1)[:n]
    return season


lambda1 = stats.boxcox(data.value)
#print(f'Nakładamy tranformacje box-cox z lambda =: {lambda1[1]}')
boxdata = lambda1[0]
t = np.arange(n)

reg = sm.OLS(boxdata, sm.add_constant(t)).fit()
reg_trend = inv_boxcox(reg.fittedvalues, lambda1[1])
reg_season = estimate_season(data.value - reg_trend)
reg_noise_og = data.value - reg_trend - reg_season

#Po analizie outliers
reg_noise = mean_outlier(reg_noise_og, 2)

LBTestReg = sm.stats.acorr_ljungbox(reg_noise, lags=[4, 8, 12])
SWTestReg = stats.shapiro(reg_noise)
SSM2 = np.sum((reg_trend+reg_season - data.value.mean())**2)
R2 = SSM2/SST

'''
fig, ax = plt.subplots(4,1, sharex=False, figsize=(8,5))
fig.suptitle('Dekompozycja Beer Production')
sns.lineplot(x=t,y=data['value'],ax=ax[0], color='black')
sns.lineplot(reg_trend + reg_season, ax=ax[0], color='red', alpha=0.5)
ax[0].set_title('Pierwotny szereg')
sns.lineplot(reg_trend,ax=ax[1], color='darkred')
ax[1].set_title('Składowa trendu')
sns.lineplot(reg_season,ax=ax[2], color='darkgreen')
ax[2].set_title('Składowa sezonowa')
sns.lineplot(reg_noise, ax=ax[3], color='darkblue')
ax[3].set_title('Szum')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
fig.suptitle('Porównanie ACF')
plot_acf(reg_noise, ax=ax[0], color='darkgreen')
ax[0].set_title('ACF')
plot_pacf(reg_noise, ax=ax[1], color='darkgreen')
ax[1].set_title('PACF')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8,5))
sm.qqplot(reg_noise, line='45', fit=True, ax=ax)
plt.suptitle('Wykres qqplot dla reszt')
plt.tight_layout()
#plt.savefig('wykres5.png', format='png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(np.arange(n), reg_noise)
ax.scatter([3,7,59], reg_noise_og.reset_index().iloc[[3,7,59],1], color='red')
plt.suptitle(f'Wartości dla których Z > 2.9')
plt.tight_layout()
#plt.savefig('wykres6.png', format='png', bbox_inches='tight')
plt.show()
'''


print(f'Transformacja Box-Cox i estymacja trendu regresją liniową:'
      f'R^2 = {R2}\n'
      f'LJUNG-BOX = {LBTestReg}\n'
      f'SHAPIRO-WILK = {SWTestReg}\n')



### model wielomianowy
k = [2,3,4,5]
coef_polyreg = {d: np.polyfit(t, boxdata, d, full=True)[0] for d in k}
polyreg = {d: {'trend': inv_boxcox(np.polyval(coef_polyreg[d], t), lambda1[1]),
               'season': estimate_season(data.value - inv_boxcox(np.polyval(coef_polyreg[d], t), lambda1[1])),
               'noise': data.value -
                        inv_boxcox(np.polyval(coef_polyreg[d], t), lambda1[1]) -
                        estimate_season(data.value - inv_boxcox(np.polyval(coef_polyreg[d], t), lambda1[1]))} for d in k}

### Po analize outliers



for d in k:
    polyreg[d]['s_noise'] = mean_outlier(polyreg[d]['noise'], 2)


for d in k:
    LBTest = sm.stats.acorr_ljungbox(polyreg[d]['s_noise'], lags=[4,8,12])
    SWTest = stats.shapiro(polyreg[d]['s_noise'])
    SSM = np.sum((polyreg[d]['trend'] + polyreg[d]['season'] - data.value.mean()) ** 2)
    R = SSM / SST
    print(f'Transformacja Box-Cox i estymacja trendu regresją wielomianową dla stopnia {d} \n:'
          f'R^2 = {R}\n'
          f'LJUNG-BOX = \n{LBTest}\n'
          f'SHAPIRO-WILK = \n{SWTest}\n')
    '''
    fig, ax = plt.subplots(4, 1, sharex=False, figsize=(8,5))
    fig.suptitle(f'Dekompozycja Beer Production dla wielomianu stopnia {d}')
    sns.lineplot(x=t, y=data['value'], ax=ax[0], color='black')
    sns.lineplot(polyreg[d]['trend'] + polyreg[d]['season'], ax=ax[0], color='red', alpha=0.5)
    ax[0].set_title('Pierwotny szereg')
    sns.lineplot(polyreg[d]['trend'], ax=ax[1], color='darkred')
    ax[1].set_title('Składowa trendu')
    sns.lineplot(polyreg[d]['season'], ax=ax[2], color='darkgreen')
    ax[2].set_title('Składowa sezonowa')
    sns.lineplot(polyreg[d]['s_noise'], ax=ax[3], color='darkblue')
    ax[3].set_title('Szum')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    fig.suptitle('Porównanie ACF')
    plot_acf(polyreg[d]['s_noise'], ax=ax[0], color='darkgreen')
    ax[0].set_title('ACF')
    plot_pacf(polyreg[d]['s_noise'], ax=ax[1], color='darkgreen')
    ax[1].set_title('PACF')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    sm.qqplot(polyreg[d]['s_noise'], line='45', fit=True, ax=ax)
    plt.suptitle('Wykres qqplot dla reszt')
    plt.tight_layout()
    #plt.savefig('wykres5.png', format='png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(np.arange(n), polyreg[d]['s_noise'])
    ax.scatter([3,7,59], polyreg[d]['noise'].reset_index().iloc[[3,7,59],1], color='red')
    plt.suptitle(f'Wartości dla których Z > 2.9')
    plt.tight_layout()
    #plt.savefig('wykres6.png', format='png', bbox_inches='tight')
    plt.show()
    '''

d=2


fig, ax = plt.subplots(3, 1, sharex=False, figsize=(8,5))
plt.title(f'Dekompozycja Beer Production dla wielomianu stopnia {d}', loc='center')
sns.lineplot(polyreg[d]['trend'], ax=ax[0], color='darkred')
ax[0].set_title('Składowa trendu')
sns.lineplot(polyreg[d]['season'], ax=ax[1], color='darkgreen')
ax[1].set_title('Składowa sezonowa')
sns.lineplot(polyreg[d]['s_noise'], ax=ax[2], color='darkblue')
ax[2].set_title('Szum')
plt.tight_layout()
plt.savefig('wykres2.png', format='png', bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(sharex=False, figsize=(8,5))
sns.lineplot(x=t, y=data['value'], color='black', label='values')
sns.lineplot(polyreg[d]['trend'] + polyreg[d]['season'], color='red', alpha=0.8, label='model')
plt.title('Porównanie wartości przybliżonych modelem z prawdziwymi', loc='center')
plt.tight_layout()
plt.savefig('wykres3.png', format='png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
plt.title('Porównanie reszt', loc='center')
plot_acf(polyreg[d]['noise'], ax=ax[0], color='darkblue')
ax[0].set_title('ACF')
plot_pacf(polyreg[d]['noise'], ax=ax[1], color='darkblue')
ax[1].set_title('PACF')
plt.tight_layout()
plt.savefig('wykres4.png', format='png', bbox_inches='tight')
plt.show()


'''
### Eksport danych
decompose_poly2 = pd.DataFrame({'trend': polyreg[2]['trend'], 'season':polyreg[2]['season'], 'noise':polyreg[2]['noise'], 's_noise':polyreg[2]['s_noise']})
#decompose_poly3 = pd.DataFrame({'trend': polyreg[3]['trend'], 'season':polyreg[3]['season'], 'noise':polyreg[3]['noise']})
decompose_poly2.to_csv('decompose_poly2.csv')
#decompose_poly3.to_csv('decompose_poly3.csv')
'''


### regresja nielokalna

lowess_reg = sm.nonparametric.lowess
k = [2/3, 1/4, 1/3, 1/6, 1/8]

localreg_trend = {d: lowess_reg(data.value, t, frac=d)[:,1] for d in k}
localreg_season = {d: estimate_season(data.value - localreg_trend[d]) for d in k}
localreg_noise = {d: data.value - localreg_trend[d] - localreg_season[d] for d in k}
localreg_s_noise = {d: mean_outlier(localreg_noise[d]) for d in k}

for d in k:
    LBTest = sm.stats.acorr_ljungbox(localreg_s_noise[d], lags=[4,8,9,12])
    SWTest = stats.shapiro(localreg_s_noise[d])
    SSM = np.sum((localreg_trend[d] + localreg_season[d] - data.value.mean()) ** 2)
    R = SSM / SST
    print(f'Estymacja trendu regresją lokalną dla stopnia {d} \n:'
          f'R^2 = {R}\n'
          f'LJUNG-BOX = \n{LBTest}\n'
          f'SHAPIRO-WILK = \n{SWTest}\n')
    '''
    fig, ax = plt.subplots(4, 1, sharex=False)
    fig.suptitle(f'Dekompozycja Beer Production dla regresji nielokalnej z parametrem{d}')
    sns.lineplot(x=t, y=data['value'], ax=ax[0], color='black')
    sns.lineplot(localreg_trend[d] + localreg_season[d], ax=ax[0], color='red', alpha=0.5)
    ax[0].set_title('Pierwotny szereg')
    sns.lineplot(localreg_trend[d], ax=ax[1], color='darkred')
    ax[1].set_title('Składowa trendu')
    sns.lineplot(localreg_season[d], ax=ax[2], color='darkgreen')
    ax[2].set_title('Składowa sezonowa')
    sns.lineplot(localreg_s_noise[d], ax=ax[3], color='darkblue')
    ax[3].set_title('Szum')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Porównanie ACF')
    plot_acf(localreg_s_noise[d], ax=ax[0], color='darkgreen')
    ax[0].set_title('ACF')
    plot_pacf(localreg_s_noise[d], ax=ax[1], color='darkgreen')
    ax[1].set_title('PACF')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    sm.qqplot(localreg_s_noise[d], line='45', fit=True, ax=ax)
    plt.suptitle('Wykres qqplot dla reszt')
    plt.tight_layout()
    #plt.savefig('wykres5.png', format='png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(np.arange(n), localreg_s_noise[d])
    ax.scatter([3,7,59], localreg_noise[d].reset_index().iloc[[3,7,59],1], color='red')
    plt.suptitle(f'Wartości dla których Z > 2.9')
    plt.tight_layout()
    #plt.savefig('wykres6.png', format='png', bbox_inches='tight')
    plt.show()
    '''

### eksponencjalna?

'''
def lag(data, d):
    x = data
    for _ in range(d):
        x = np.diff(x, n=1)
    return x

lag_storage = {d: lag(data.value, d) for d in range(1, 13)}
'''

'''
# Tworzenie wykresów w układzie 3x4 (3 wiersze, 4 kolumny)
fig, ax = plt.subplots(3, 4, figsize=(15, 10))  # 3 wiersze, 4 kolumny (dla 12 lagów)

# Iteracja przez lag_storage, aby przypisać każdy d do odpowiedniego podwykresu
for i, (d, lagged_values) in enumerate(lag_storage.items()):
    row = i // 4  # Indeks wiersza
    col = i % 4   # Indeks kolumny
    ax[row, col].plot(t[-len(lagged_values):], lagged_values, marker='o', color='darkblue')
    ax[row, col].set_title(f'Lag d={d}')
    ax[row, col].set_xlabel('Lata')
    ax[row, col].set_ylabel('Megalitry')

# Dopasowanie układu i wyświetlenie wykresu
plt.tight_layout()
plt.show()
'''

'''
for d in lag_storage.keys():
    plt.figure(figsize=(15, 5))
    plt.plot(lag_storage[d], label=d)
    plt.suptitle(f'{d}')
    plt.show()
'''


#result = arma_order_select_ic(polyreg[2]['noise'], max_ar=8, max_ma=8, ic='aic', trend='c')
#print(result)