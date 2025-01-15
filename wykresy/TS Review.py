import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
data = sm.datasets.get_rdataset("ausbeer", package="fpp2").data
def convert_to_datetime(row):
    year, quarter = str(row).split('.')
    quarter_to_month = {'0': '01', '25': '04', '5': '07', '75': '10'}
    month = quarter_to_month[quarter]
    return f"{year}-{month}-01"

data['time'] = pd.to_datetime(data['time'].apply(convert_to_datetime))
data = data.loc[data['time'] >= pd.Timestamp('2000-01-01')]

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

ts_plot(data)