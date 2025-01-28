# Załadowanie danych
library(readr)
library(forecast)
library(tseries)
library(ggplot2)
library(dplyr)

file_path <- "/Users/filip/repositories/time-series-project/data/cut_data.csv"
data <- read_csv(file_path)

data$time <- as.Date(data$time, format = "%Y-%m-%d")

ts_data <- ts(data$value, start = c(2000, 1), frequency = 4)

train_length <- floor(0.8 * length(ts_data))
ts_train <- window(ts_data, end = c(2000 + (train_length - 1) %/% 4, (train_length - 1) %% 4))
ts_test <- window(ts_data, start = c(2000 + train_length %/% 4, train_length %% 4 + 1))

plot(ts_data, main = "Seria czasowa", ylab = "Wartość", xlab = "Czas")
abline(v = time(ts_data)[train_length], col = "red", lty = 2) 

adf_test <- adf.test(ts_train)
print(adf_test)

sarima_model <- auto.arima(ts_train, seasonal = TRUE)

summary(sarima_model)

forecast_values <- forecast(sarima_model, h = length(ts_test))

plot(forecast_values, main = "Prognoza SARIMA", ylab = "Wartość", xlab = "Czas")
lines(ts_test, col = "red", lwd = 2)  # Dodanie rzeczywistych wartości z zbioru testowego
legend("topleft", legend = c("Prognoza", "Wartości rzeczywiste"), col = c("blue", "red"), lwd = 2, bty = "n")



accuracy_metrics <- accuracy(forecast_values, ts_test)
print(accuracy_metrics)


checkresiduals(sarima_model)

mse <- mean((ts_test - forecast_values$mean)^2, na.rm = TRUE)
print(paste("MSE:", mse))


checkresiduals(sarima_model)

residuals <- residuals(sarima_model)
qqnorm(residuals, main = "Q-Q Plot Residuals")
qqline(residuals, col = "red", lwd = 2)
shapiro.test(residuals)








forecast_horizon <- (2023 - 2000 + 1) * 4 - length(ts_train)
forecast_values <- forecast(sarima_model, h = 64)


forecast_df <- data.frame(
  time = time(forecast_values$mean),
  forecast = as.numeric(forecast_values$mean),
  lower = as.numeric(forecast_values$lower[, 2]),
  upper = as.numeric(forecast_values$upper[, 2])
)

actual_df <- data.frame(
  time = time(ts_data),
  actual = as.numeric(ts_data)
)


ggplot() +
  geom_line(data = actual_df, aes(x = time, y = actual), color = "red", size = 1) +
  geom_line(data = forecast_df, aes(x = time, y = forecast), color = "blue", size = 1, linetype = "solid") +
  geom_ribbon(data = forecast_df, aes(x = time, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  labs(title = "Prognoza SARIMA do 2023 roku", x = "Czas", y = "Wartość") +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("Prognoza" = "blue", "Rzeczywiste" = "red")) +
  guides(color = guide_legend(title = NULL))


write.csv(forecast_df, file = "forecast_output.csv", row.names = FALSE)
