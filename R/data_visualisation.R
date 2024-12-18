library(ggplot2)

beer_prod <- read.csv("monthly-beer-production-in-austr.csv", col.names = c("month", "production"))
beer_prod$month <- as.Date(paste0(beer_prod$month, "-01"))

ggplot(beer_prod, aes(x = `month`, y = `production`)) +
  geom_line() +
  theme_minimal()

