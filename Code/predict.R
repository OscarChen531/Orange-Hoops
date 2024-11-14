# Load necessary libraries
library(caret)
library(pROC)
library(dplyr)

# Load and filter data
performance <- read.csv("Data/PBP2324.csv", sep = ",", header = TRUE)

# Define clutch time: last 5 minutes of the second half with score difference between -5 and 5
clutch_data <- performance %>%
  filter(
    half == 2,
    secs_remaining <= 300,
    score_diff >= -5 & score_diff <= 5,
    shot_team == "Purdue"  # Filter for shots made by Purdue
  )

# Drop rows with any NA values in relevant columns for modeling
clutch_data <- clutch_data %>%
  drop_na(secs_remaining, score_diff, shot_outcome, shooter, three_pt, free_throw, possession_before, possession_after, win_prob)

# Create shot success as the outcome variable
clutch_data$shot_success <- ifelse(clutch_data$shot_outcome == "made", 1, 0)

# Ensure outcome variable and shooter are factors
clutch_data$shot_success <- factor(clutch_data$shot_success, levels = c(0, 1), labels = c("Fail", "Success"))
clutch_data$shooter <- as.factor(clutch_data$shooter)

# Remove rows with missing values
clutch_data <- na.omit(clutch_data)

# Check class distribution
class_dist <- table(clutch_data$shot_success)
if (any(class_dist == 0)) {
  stop("Error: One of the classes (Fail or Success) has no records after filtering.")
}

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(clutch_data$shot_success, p = .8, list = FALSE)
train_data <- clutch_data[trainIndex, ]
test_data <- clutch_data[-trainIndex, ]

# Define the formula for the RF model
# Specify predictor formula using the selected features
predictors <- shot_success ~ score_diff + secs_remaining + attendance + naive_win_prob +
  home_favored_by + total_line + home_time_out_remaining +
  away_time_out_remaining + win_prob + 
  shooter + referees + arena

# Random Forest
set.seed(123)
rf_model <- train(predictors, data = train_data, method = "rf", tuneLength = 5)
rf_metrics <- calculate_metrics(rf_model, test_data)

# Calculate probabilities for each player in the test data
test_data$predicted_prob <- predict(rf_model, test_data, type = "prob")[, "Success"]

# Choose the player with the highest predicted probability in each game situation
best_player_per_situation <- test_data %>%
  group_by(score_diff) %>%  # Replace `score_diff` with the relevant identifier for each situation
  filter(predicted_prob == max(predicted_prob)) %>%
  select(shooter, predicted_prob) %>%
  ungroup()

print(best_player_per_situation)