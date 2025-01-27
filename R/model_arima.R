# Załadowanie danych
library(readr)
library(forecast)
library(tseries)
library(ggplot2)
library(dplyr)

# Wczytanie danych
file_path <- "/Users/filip/repositories/time-series-project/data/cut_data.csv"
data <- read_csv(file_path)

# Konwersja kolumny 'time' na format daty
data$time <- as.Date(data$time, format = "%Y-%m-%d")

# Utworzenie obiektu serii czasowej
ts_data <- ts(data$value, start = c(2000, 1), frequency = 4)  # Zakładamy dane kwartalne

# Podział na zbiór treningowy i testowy
train_length <- floor(0.8 * length(ts_data))
ts_train <- window(ts_data, end = c(2000 + (train_length - 1) %/% 4, (train_length - 1) %% 4 + 1))
ts_test <- window(ts_data, start = c(2000 + train_length %/% 4, train_length %% 4 + 1))

# Wizualizacja serii czasowej
plot(ts_data, main = "Seria czasowa", ylab = "Wartość", xlab = "Czas")
abline(v = time(ts_data)[train_length], col = "red", lty = 2)  # Linia podziału na zbiory

# Sprawdzenie stacjonarności (test Dickeya-Fullera)
adf_test <- adf.test(ts_train)
print(adf_test)

# Dopasowanie modelu SARIMA
# Automatyczny dobór parametrów
sarima_model <- auto.arima(ts_train, seasonal = TRUE)

# Podsumowanie modelu
summary(sarima_model)

# Prognoza na podstawie zbioru treningowego
forecast_values <- forecast(sarima_model, h = length(ts_test))

# Wizualizacja prognozy
plot(forecast_values, main = "Prognoza SARIMA", ylab = "Wartość", xlab = "Czas")
lines(ts_test, col = "red", lwd = 2)  # Dodanie rzeczywistych wartości z zbioru testowego
legend("topleft", legend = c("Prognoza", "Wartości rzeczywiste"), col = c("blue", "red"), lwd = 2, bty = "n")


# Obliczenie jakości predykcji
accuracy_metrics <- accuracy(forecast_values, ts_test)
print(accuracy_metrics)

# Wizualizacja rezydualiów
checkresiduals(sarima_model)

mse <- mean((ts_test - forecast_values$mean)^2, na.rm = TRUE)
print(paste("MSE:", mse))

# Wizualizacja rezydualiów
checkresiduals(sarima_model)

residuals <- residuals(sarima_model)
qqnorm(residuals, main = "Q-Q Plot Residuals")
qqline(residuals, col = "red", lwd = 2)
shapiro.test(residuals)

