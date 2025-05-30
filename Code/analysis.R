# Plot the shot success rate per shooter
ggplot(clutch_data, aes(x = shooter, fill = as.factor(shot_success))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Shot Success Rate by Shooter", x = "Shooter", y = "Shot Success Rate (%)", fill = "Shot Outcome") +
  theme_minimal()

# Plot the distribution of three-point attempts and free throws
ggplot(clutch_data, aes(x = three_pt, fill = as.factor(shot_success))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Three-Point Attempt Success Rate", x = "Three-Point Attempt", y = "Shot Success Rate (%)", fill = "Shot Outcome") +
  theme_minimal()

ggplot(clutch_data, aes(x = free_throw, fill = as.factor(shot_success))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Free Throw Success Rate", x = "Free Throw", y = "Shot Success Rate (%)", fill = "Shot Outcome") +
  theme_minimal()

# Plot shot success over time remaining
ggplot(clutch_data, aes(x = secs_remaining, y = shot_success, color = shooter)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Shot Success by Time Remaining", x = "Seconds Remaining", y = "Shot Success Probability") +
  theme_minimal()

# Plot shot success by score difference
ggplot(clutch_data, aes(x = score_diff, y = shot_success, color = shooter)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Shot Success by Score Difference", x = "Score Difference", y = "Shot Success Probability") +
  theme_minimal()

# Plot shot success by win probability
ggplot(clutch_data, aes(x = win_prob, y = shot_success, color = shooter)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Shot Success by Win Probability", x = "Win Probability", y = "Shot Success Probability") +
  theme_minimal()

# Shot success by possession before the shot
ggplot(clutch_data, aes(x = possession_before, y = shot_success, color = shooter)) +
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
  labs(title = "Shot Success by Possession Before Shot", x = "Possession Before", y = "Shot Success Probability") +
  theme_minimal()

# Prepare the dataset for feature plot
# Convert shot_success to a factor if it’s not already
clutch_data$shot_success <- as.factor(clutch_data$shot_success)

# Select predictors and the response variable
predictors <- clutch_data %>%
  select(score_diff, secs_remaining, attendance, naive_win_prob, 
         home_favored_by, total_line, home_time_out_remaining, 
         away_time_out_remaining, win_prob)

# Create the feature plot
featurePlot(
  x = predictors,
  y = clutch_data$shot_success,
  plot = "density",
  scales = list(x = list(relation = "free"), y = list(relation = "free")),
  auto.key = list(columns = 2),
  main = "Feature Plot of Predictors by Shot Success"
)

# Select predictors and the response variable
predictors <- clutch_data %>%
  select(score_diff, secs_remaining, attendance, naive_win_prob, 
         home_favored_by, total_line, home_time_out_remaining, 
         away_time_out_remaining, win_prob)

predictors <- na.omit(predictors)

# Compute the correlation matrix
cor_matrix <- cor(predictors, use = "complete.obs")

library(ggcorrplot)

# Plot using ggcorrplot
ggcorrplot(cor_matrix, lab = TRUE, colors = c("blue", "white", "red"))

# Load necessary libraries
library(caret)
library(pROC)

# Prepare Data
# Filter out rows where shooter is Caleb Furst
clutch_data <- subset(clutch_data, shooter != "Caleb Furst")

# Ensure outcome variable and shooter are factors, rename levels for clarity
clutch_data$shot_success <- factor(clutch_data$shot_success, levels = c(0, 1), labels = c("Fail", "Success"))
clutch_data$shooter <- as.factor(clutch_data$shooter)
clutch_data$referees <- as.factor(clutch_data$referees)
clutch_data$arena <- as.factor(clutch_data$arena)


# Remove any rows with missing values in the entire dataset
clutch_data <- na.omit(clutch_data)

# Check class balance to ensure both classes are present
if (any(table(clutch_data$shot_success) == 0)) {
  stop("Error: One of the classes (Fail or Success) has no records in the dataset.")
}

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(clutch_data$shot_success, p = .8, list = FALSE)
train_data <- clutch_data[trainIndex, ]
test_data <- clutch_data[-trainIndex, ]

# Apply up-sampling on training data to balance classes if necessary
train_data <- upSample(x = train_data[, -which(names(train_data) == "shot_success")],
                       y = train_data$shot_success)
names(train_data)[ncol(train_data)] <- "shot_success"  # Rename outcome column

# Define a function to calculate metrics for a model
calculate_metrics <- function(model, test_data, positive_class = "Success") {
  pred <- predict(model, test_data)
  pred_prob <- predict(model, test_data, type = "prob")[, positive_class]
  cm <- confusionMatrix(pred, test_data$shot_success)
  auc_value <- auc(test_data$shot_success, pred_prob)
  f1 <- 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) / 
    (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"])
  
  list(Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
       Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"],
       AUC = auc_value, F1_Score = f1)
}

# Specify predictor formula using the selected features
predictors <- shot_success ~ score_diff + secs_remaining + attendance + naive_win_prob +
  home_favored_by + total_line + home_time_out_remaining +
  away_time_out_remaining + win_prob + 
  shooter + referees + arena

# Train Models and Calculate Metrics

# Logistic Regression
set.seed(123)
logistic_model <- train(predictors, data = train_data, method = "glm", family = "binomial")
logistic_metrics <- calculate_metrics(logistic_model, test_data)

# Ridge Regression
set.seed(123)
ridge_model <- train(predictors, data = train_data, method = "glmnet",
                     tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 0.1, by = 0.01)),
                     trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
ridge_metrics <- calculate_metrics(ridge_model, test_data)

# Support Vector Machine (SVM) with Radial Basis Function Kernel
set.seed(123)
svm_model <- train(predictors, data = train_data, method = "svmRadial", tuneLength = 10,
                   trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
svm_metrics <- calculate_metrics(svm_model, test_data)

# Random Forest
set.seed(123)
rf_model <- train(predictors, data = train_data, method = "rf", tuneLength = 5)
rf_metrics <- calculate_metrics(rf_model, test_data)

# Gradient Boosting Machine (GBM)
set.seed(123)
gbm_model <- train(predictors, data = train_data, method = "gbm", verbose = FALSE, tuneLength = 5,
                   trControl = trainControl(method = "cv", number = 5))
gbm_metrics <- calculate_metrics(gbm_model, test_data)

# K-Nearest Neighbors (KNN)
set.seed(123)
knn_model <- train(predictors, data = train_data, method = "knn", tuneLength = 5)
knn_metrics <- calculate_metrics(knn_model, test_data)

# Compile Results
results <- data.frame(
  Model = c("Logistic Regression", "Ridge Regression", "SVM", "Random Forest", "GBM", "KNN"),
  Accuracy = c(logistic_metrics$Accuracy, ridge_metrics$Accuracy, svm_metrics$Accuracy,
               rf_metrics$Accuracy, gbm_metrics$Accuracy, knn_metrics$Accuracy),
  Kappa = c(logistic_metrics$Kappa, ridge_metrics$Kappa, svm_metrics$Kappa,
            rf_metrics$Kappa, gbm_metrics$Kappa, knn_metrics$Kappa),
  Sensitivity = c(logistic_metrics$Sensitivity, ridge_metrics$Sensitivity, svm_metrics$Sensitivity,
                  rf_metrics$Sensitivity, gbm_metrics$Sensitivity, knn_metrics$Sensitivity),
  Specificity = c(logistic_metrics$Specificity, ridge_metrics$Specificity, svm_metrics$Specificity,
                  rf_metrics$Specificity, gbm_metrics$Specificity, knn_metrics$Specificity),
  AUC = c(logistic_metrics$AUC, ridge_metrics$AUC, svm_metrics$AUC,
          rf_metrics$AUC, gbm_metrics$AUC, knn_metrics$AUC),
  F1_Score = c(logistic_metrics$F1_Score, ridge_metrics$F1_Score, svm_metrics$F1_Score,
               rf_metrics$F1_Score, gbm_metrics$F1_Score, knn_metrics$F1_Score)
)

print(results)


# Normalize metrics for comparison
results$Accuracy_norm <- results$Accuracy / max(results$Accuracy)
results$AUC_norm <- results$AUC / max(results$AUC)
results$F1_Score_norm <- results$F1_Score / max(results$F1_Score)

# Assign new weights
weight_accuracy <- 0.25
weight_auc <- 0.25
weight_f1 <- 0.20
weight_sensitivity <- 0.15
weight_specificity <- 0.10
weight_kappa <- 0.05

# Calculate the new composite score
results$Composite_Score <- (results$Accuracy_norm * weight_accuracy) + 
  (results$AUC_norm * weight_auc) + 
  (results$F1_Score_norm * weight_f1) +
  (results$Sensitivity * weight_sensitivity) +
  (results$Specificity * weight_specificity) +
  (results$Kappa * weight_kappa)

# Sort by the new composite score
results <- results[order(-results$Composite_Score), ]
print("Ranked Models by New Composite Score:")
print(results)

# Select the best model based on highest composite score
best_model <- results[1, ]
print("Best Model:")
print(best_model)
