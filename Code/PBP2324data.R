library(glmnet)
library(caret)
library(Matrix)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(readr)
library(Metrics)
library(pROC)
library(pls)  # For PLSR and PCR models


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

# Display data structure
str(clutch_data)
