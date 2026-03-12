# AI-assisted stock prediction task in R
# First pass: intentionally imperfect AI-style workflow
# Checked pass: human-reviewed correction

library(readr)
library(dplyr)
library(randomForest)

# Load data
df <- read_csv("finance-charts-apple.csv", show_col_types = FALSE)

# Basic preprocessing
df <- df %>%
  mutate(Date = as.Date(Date)) %>%
  arrange(Date) %>%
  mutate(
    return_t = AAPL.Close / lag(AAPL.Close) - 1,
    return_lag1 = lag(return_t),
    ma5 = zoo::rollmean(AAPL.Close, 5, fill = NA, align = "right"),
    vol5 = zoo::rollapply(return_t, 5, sd, fill = NA, align = "right")
  )

# -----------------------------
# FIRST PASS: intentionally imperfect AI workflow
# Mistake 1: predicts SAME-DAY direction, not NEXT-DAY direction
# Mistake 2: includes same-day price information in predictors
# Mistake 3: random split for time series data
# -----------------------------
bad <- df %>%
  filter(!is.na(return_t), !is.na(return_lag1), !is.na(ma5), !is.na(vol5)) %>%
  mutate(target_same_day_up = ifelse(AAPL.Close > AAPL.Open, 1, 0))

bad_features <- c(
  "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume",
  "mavg", "dn", "up", "return_t", "return_lag1", "ma5", "vol5"
)

set.seed(42)
idx_bad <- sample(seq_len(nrow(bad)), size = floor(0.7 * nrow(bad)))
train_bad <- bad[idx_bad, ]
test_bad  <- bad[-idx_bad, ]

rf_bad <- randomForest(
  x = train_bad[, bad_features],
  y = as.factor(train_bad$target_same_day_up),
  ntree = 200
)

pred_bad <- predict(rf_bad, newdata = test_bad[, bad_features])
acc_bad <- mean(pred_bad == as.factor(test_bad$target_same_day_up))
cm_bad <- table(
  Actual = test_bad$target_same_day_up,
  Predicted = pred_bad
)

cat("First-pass AI workflow accuracy:", round(acc_bad, 3), "\n")
print(cm_bad)

# -----------------------------
# CHECKED / CORRECTED WORKFLOW
# Goal: predict NEXT-DAY direction using only information available by day t
# Fix 1: target is next-day up/down
# Fix 2: only lagged or current-day-available features
# Fix 3: chronological split instead of random split
# -----------------------------
good <- df %>%
  mutate(
    target_next_day_up = ifelse(lead(AAPL.Close) > AAPL.Close, 1, 0),
    ma5_lag1 = lag(ma5),
    vol5_lag1 = lag(vol5)
  ) %>%
  filter(!is.na(return_lag1), !is.na(ma5_lag1), !is.na(vol5_lag1), !is.na(AAPL.Volume))

good_features <- c("return_lag1", "ma5_lag1", "vol5_lag1", "AAPL.Volume")

split <- floor(0.7 * nrow(good))
train_good <- good[1:split, ]
test_good  <- good[(split + 1):nrow(good), ]

rf_good <- randomForest(
  x = train_good[, good_features],
  y = as.factor(train_good$target_next_day_up),
  ntree = 200
)

pred_good <- predict(rf_good, newdata = test_good[, good_features])
acc_good <- mean(pred_good == as.factor(test_good$target_next_day_up))
cm_good <- table(
  Actual = test_good$target_next_day_up,
  Predicted = pred_good
)

cat("Checked workflow accuracy:", round(acc_good, 3), "\n")
print(cm_good)

# Simple comparison
results <- data.frame(
  workflow = c("First-pass AI attempt", "Checked/corrected attempt"),
  accuracy = c(acc_bad, acc_good)
)
print(results)
