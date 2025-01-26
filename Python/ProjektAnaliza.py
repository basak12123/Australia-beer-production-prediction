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
from ProjektDekompozycja import data, polyreg
from scipy.stats import percentileofscore
from scipy.stats import t

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


###################################################################################################


ddata = pd.DataFrame(polyreg[2])
ddata.reset_index(drop=True)
ddata['value'] = data.value
n = len(ddata.value)



outlier_value = 2.9
z_scores = np.abs((ddata.noise - ddata.noise.mean()) / ddata.noise.std())
outliers = ddata.noise[z_scores > outlier_value]
indices = np.arange(n)[z_scores > outlier_value]
print(f'Wartości odstające (Z-Score): {outliers}')
print(f'Indeksy wartości odstających (Z-Score): {indices}')

def iqr_outlier_detection(data):
    Q1 = np.percentile(data, 25)  # Pierwszy kwartyl
    Q3 = np.percentile(data, 75)  # Trzeci kwartyl
    IQR = Q3 - Q1  # Rozstęp międzykwartylowy

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]

    return outliers, indices

outliers, indices = iqr_outlier_detection(ddata.noise)
print("Wartości odstające (IQR):", outliers)
print("Indeksy wartości odstających (IQR):", indices)


def grubs_test(data, alpha=0.05):
    """
    Wykrywa wartości odstające za pomocą testu Grubbsa.

    Parametry:
        data (list lub numpy array): Dane wejściowe.
        alpha (float): Poziom istotności (domyślnie 0.05).

    Zwraca:
        outlier (float lub None): Wartość odstająca (jeśli istnieje).
        index (int lub None): Indeks wartości odstającej w oryginalnych danych (jeśli istnieje).
    """
    n = len(data)
    if n < 3:
        raise ValueError("Test Grubbsa wymaga co najmniej 3 obserwacji.")

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    G = max(abs(data - mean) / std_dev)
    critical_value = (n - 1) / np.sqrt(n) * np.sqrt(
        t.ppf(1 - alpha / (2 * n), n - 2) ** 2 / (n - 2 + t.ppf(1 - alpha / (2 * n), n - 2) ** 2))

    if G > critical_value:
        outlier = data[np.argmax(abs(data - mean))]
        index = np.argmax(abs(data - mean))
        return outlier, index
    else:
        return None, None

outlier, index = grubs_test(list(ddata.noise))
if outlier is not None:
    print("Wartość odstająca (Grubbsa):", outlier)
    print("Indeks wartości odstającej (Grubbsa):", index)
else:
    print("Brak wartości odstających (Grubbsa).")


def mean_outlier(indexes, n, data=ddata.noise):
    result = data.copy()
    means = np.convolve(data, np.ones(n) / n, mode='valid')
    result.iloc[indices] = means[indices]
    return result

noise = mean_outlier(indices, 2)

#print(ddata.noise.iloc[indices])



fig, ax = plt.subplots(figsize=(8,7))
sm.qqplot(noise, line='45', fit=True, ax=ax)
plt.title('Wykres QQ dla szeregu z uśrednionymi obserwacjami odstającymi', loc='center')
plt.tight_layout()
ax.set_ylabel('Kwantyle próbkowe')
ax.set_xlabel('Kwantyle teoretyczne')
plt.savefig('wykres5.png', format='png', bbox_inches='tight')
plt.show()


# Na potrzeby wykresu
reduced = pd.DataFrame({'time': data.time, 'value': noise})

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(reduced.time, reduced.value)
ax.scatter(reduced.time.iloc[indices], outliers, color='red')
# Dodanie pionowych czerwonych kropkowanych linii
for time in reduced.time.iloc[indices]:
    ax.axvline(x=time, color='red', linestyle='--', linewidth=0.8, alpha=0.3)

# Dodanie ticks na osi X
ax.set_xticks(reduced.time.iloc[indices])
ax.set_xticklabels(reduced.time.iloc[indices].dt.strftime('%Y-%m'), rotation=45, fontsize=8)
plt.title(r'Szereg z uśrednionymi obserwacjami odstającymi', loc='center')
plt.tight_layout()
plt.savefig('wykres6.png', format='png', bbox_inches='tight')
plt.show()
