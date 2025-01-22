library(MASS)
library(forecast)
library(tseries)
library(astsa)
library(knitr)
library(gridExtra)
library(grid)

beer <- read.csv("/Users/filip/repositories/time-series-project/data/Beer_to_2010.csv")
beer <- read.csv("/Users/filip/repositories/time-series-project/data/cut_data.csv")
beer <- beer$value

plot.ts(beer)
beer.ts <- ts(beer, frequency = 4)

acf(beer.ts, lag.max =10, main = "")
pacf(beer.ts, lag.max=10, main = "")

seasonplot(beer.ts, 4, col=rainbow(3), year.labels = TRUE, main = "Seasonal~Plot")


bcbeer <- boxcox(beer.ts~as.numeric(1:length(beer)))
lambda1 <- bcbeer$x[which.max(bcbeer$y)]

beer.tr <- ((beer.ts)^(-2) - 1)/(-2)

ts.plot(beer.ts, ylab = "Production in Megalitres",xlab = "Time(Quarterly)", main = "Box Cox Transformed Data")

var(beer.ts)
var(beer.tr)


# Differencing Seasonality at lag 4
beerdiff4 <- diff(beer.ts, lag =4)
var(beerdiff4)
ts.plot(beerdiff4, ylab = "Differenced At Lag 4 ")
abline(lm(beerdiff4~as.numeric(1:length( beerdiff4))), col ="red")

# Differencing at Lag 1
beerdiff4diff1 <- diff(beerdiff4, lag =1)
var(beerdiff4diff1)
ts.plot(beerdiff4diff1, ylab = "Differenced At Lag 4, Lag 1")
abline(lm(beerdiff4diff1~as.numeric(1:length( beerdiff4diff1))), col ="red")


# Differenced at lag 4 PACF AND ACF
par(mfrow=c(1,2))
acf(beerdiff4, lag.max = 24, main = "")
pacf(beerdiff4, lag.max = 24, main = "")

# DIFFERENCED at lag 4 and lag 1 PACF AND ACF
par(mfrow=c(1,2))
acf(beerdiff4diff1, lag.max=24, main = "")
pacf(beerdiff4diff1, lag.max=24, main = "")


