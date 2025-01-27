library(MASS)
library(forecast)
library(Metrics)
library(ggplot2)

beer <- read.csv("/Users/filip/repositories/time-series-project/data/Beer_to_2010.csv")

beer_dec2 <- read.csv("/Users/filip/repositories/time-series-project/data/decompose_poly2.csv")
beer_dec3 <- read.csv("/Users/filip/repositories/time-series-project/data/decompose_poly3.csv")

# 1
beer <- beer[137:length(beer$X), c(2,3)]

beer.ts <- ts(beer$value, frequency = 4)

bcbeer <- boxcox(beer.ts~as.numeric(1:length(beer$value)), lambda = seq(-7, 7, 1/100))
lambda1 <- bcbeer$x[which.max(bcbeer$y)]

var(beer$value)

beer.bc <- BoxCox(beer.ts, lambda1)

dec_ad <- decompose(beer.bc, "additive")

plot(dec_ad)

shapiro.test(dec_ad$random)


mod1 <- ar(dec_ad$random[3:(length(beer.ts)-2)], method = "yule-walker")

summary(mod1)
mod1$var.pred

predict(beer.ts)


Box.test(residuals(mod1), lag = 10, type = "Ljung-Box")

forecast_values <- forecast(mod1, h = 12)
plot(forecast_values)



# 2

beer_dec2.ts <- ts(beer_dec2$noise, frequency = 4)
mod2 <- ar(beer_dec2.ts, method = "yule-walker")

forecast_values <- forecast(mod2, h = 12)
plot(forecast_values)

train <- beer_dec2[1:72, ]
test <- beer_dec2[73:82, ]

mod2.1 <- ar(train$noise, method = "yule-walker")

forecast_test <- forecast(mod2.1, h = length(test$noise))

ar_forecast <- predict(mod2.1, n.ahead = length(test$noise))

forecast_values <- ts(ar_forecast$pred)

pred <- c(ar_forecast$pred)

mae_value <- mae(test$noise, pred)
rmse_value <- rmse(test$noise, pred) 


approx_error <- abs(test$noise - pred) / (test$noise + test$trend + test$season)
mean(approx_error)

data2 <- data.frame(Predykcja = pred + test$trend + test$season, 
                   Rzeczywiste = beer$value[73:82]) 

mse2 <- mean((data2$Predykcja - data2$Rzeczywiste)^2)

approx_error <- abs(data2$Predykcja - data2$Rzeczywiste) / data2$Rzeczywiste
mean(approx_error)

ggplot(data2, aes(x = 1:length(pred))) +
  geom_line(aes(y = Predykcja, color = "Predykcja"), size = 1) +
  geom_line(aes(y = Rzeczywiste, color = "Rzeczywiste"), size = 1) +
  labs(
    title = "Porównanie Predykcji i Wartości Rzeczywistych dla wielomianu rzędu 2",
    x = "Indeks",
    y = "Wartość",
    color = "Legenda"
  ) +
  scale_color_manual(values = c("Predykcja" = "blue", "Rzeczywiste" = "red")) +
  theme_minimal() +
  theme(
    legend.position = "right" 
  )

plot.ts(beer_dec2$trend + beer_dec2$season + beer_dec2$noise)
lines(x = 73:82, data2$Predykcja, col = "red")

qqnorm(beer_dec2$noise)
qqline(beer_dec2$noise)

# 3
ts.plot(beer_dec2$noise)

beer_dec3.ts <- ts(beer_dec3$noise, frequency = 4)
mod3 <- ar(beer_dec3.ts, method = "yule-walker")

forecast_values <- forecast(mod3, h = 12)
plot(forecast_values)

train2 <- beer_dec3[1:72, ]
test2 <- beer_dec3[73:82, ]

mod3.1 <- ar(train$noise, method = "yule-walker")

forecast_test2 <- forecast(mod3.1, h = length(test2$noise))
ar_forecast2 <- predict(mod3.1, n.ahead = length(test2$noise))
forecast_values2 <- ts(ar_forecast$pred)
pred2 <- c(ar_forecast$pred)

data3 <- data.frame(Predykcja = pred2 + test2$trend + test2$season, 
                    Rzeczywiste = beer$value[73:82]) 

approx_error2 <- abs(data3$Predykcja - data3$Rzeczywiste) / data3$Rzeczywiste
mean(approx_error2)

mse3 <- mean((data3$Predykcja - data3$Rzeczywiste)^2)


ggplot(data3, aes(x = 1:length(pred2))) +
  geom_line(aes(y = Predykcja, color = "Predykcja"), size = 1) +
  geom_line(aes(y = Rzeczywiste, color = "Rzeczywiste"), size = 1) +
  labs(
    title = "Porównanie Predykcji i Wartości Rzeczywistych dla wielomianu rzędu 3",
    x = "Indeks",
    y = "Wartość",
    color = "Legenda"
  ) +
  scale_color_manual(values = c("Predykcja" = "blue", "Rzeczywiste" = "red")) +
  theme_minimal() +
  theme(
    legend.position = "right" 
  )




