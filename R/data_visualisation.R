library(ggplot2)
library(dplyr)
library(forecast)


beer_prod <- read.csv("Beer_to_2010.csv")

BP_ts <- ts(beer_prod$value, frequency = 12)

bp_ts_2010 <- ts(beer_prod$value, frequency = 4)
ggplot(beer_prod, aes(x = `time`, y = `value`)) +
  geom_line() +
  theme_minimal()

ts.plot(bp_qt$value)

Acf(beer_prod$value, lag.max = 40)
Pacf(beer_prod$value, lag.max = 40)


BP_spencer <- stats::filter(beer_prod$value, (1/320) * c(-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 21, 3, -5, -6, -3), sides = 2)

ts.plot(BP_spencer)

BP_tslm <- tslm(BP_ts ~ trend + season)


dec_BP_ad <- decompose(BP_ts, "additive")
dec_BP_mp <- decompose(BP_ts, "multiplicative")

plot(dec_BP_ad)
plot(dec_BP_mp)


bp_ad_2010 <- decompose(bp_ts_2010, "additive")
bp_mp_2010 <- decompose(bp_ts_2010, "multiplicative")
plot(bp_ad_2010)
plot(bp_mp_2010)
