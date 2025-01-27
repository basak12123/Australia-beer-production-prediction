library(readr)
library(forecast)
library(tseries)

# Wczytanie danych
file_path <- "/Users/filip/repositories/time-series-project/data/cut_data.csv"
data <- read_csv(file_path)

# Konwersja kolumny 'time' na format daty
data$time <- as.Date(data$time, format = "%Y-%m-%d")

# Utworzenie obiektu serii czasowej
ts_data <- ts(data$value, start = c(2000, 1), frequency = 4)  # Zakładamy dane kwartalne

# Wizualizacja serii czasowej
plot(ts_data, main = "Seria czasowa", ylab = "Wartość", xlab = "Czas")

# Sprawdzenie stacjonarności (test Dickeya-Fullera)
adf_test <- adf.test(ts_data)
print(adf_test)

# Dopasowanie modelu SARIMA
# Automatyczny dobór parametrów
sarima_model <- auto.arima(ts_data, seasonal = TRUE)

# Podsumowanie modelu
summary(sarima_model)

# Wizualizacja dopasowania
checkresiduals(sarima_model)

# Prognoza na kolejne 4 okresy
forecast_values <- forecast(sarima_model, h = 4)

# Wizualizacja prognozy
plot(forecast_values, main = "Prognoza SARIMA", ylab = "Wartość", xlab = "Czas")
