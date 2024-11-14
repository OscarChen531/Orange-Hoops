# Orange Hoops Data Science Contest: Data-Driven Decision Making for Clutch-Time Shot Selection


## Overview

This project leverages NCAA Men's Basketball data from the 2023-2024 season to analyze and predict the best player for taking high-stakes shots in clutch game moments. Using machine learning, we focused on identifying which Purdue University player has the highest likelihood of shot success in the final five minutes of close games. By modeling game conditions such as score difference, time remaining, and win probability, the project aims to inform strategic decisions for optimal shot selection under pressure.

## Authors
- **Wei-Chieh Chen**
- **Peiya Qi**
- **Yash Gaikwad**

## Abstract

The analysis centers on data-driven shot selection during critical moments. The Random Forest model emerged as the best-performing predictor, with Zach Edey identified as the most reliable player for clutch shots. This approach aims to enhance decision-making by offering actionable insights into player performance under high-pressure conditions.

## Project Structure

- **Data Pre-Processing & Exploratory Data Analysis**: Filtering and analyzing game data focusing on clutch-time scenarios. Key visualizations included:
  - Shot success rate per player
  - Shot success based on shot type (three-point, free throw)
  - Shot success relative to time, score difference, and possession scenarios
  - Correlation analysis to identify relationships among predictive features.

- **Model Training and Evaluation**: We trained multiple models, including:
  - Logistic Regression
  - Ridge Regression
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - Gradient Boosting Machine (GBM)
  - k-Nearest Neighbors (kNN)

- **Optimal Model Selection**: Based on a composite scoring method that weighted accuracy, AUC, F1 Score, sensitivity, and other metrics, the Random Forest model was selected as the best-performing model for this project.

- **Player Selection Analysis**: The model provided predicted success probabilities for each Purdue player under various score differentials, with Zach Edey showing consistently high probabilities for successful shots in clutch moments.

## Results

The Random Forest model achieved strong performance across metrics, with Zach Edey identified as the optimal player for high-stakes shots based on his consistent performance in different game scenarios.

## Conclusion

This project demonstrates how machine learning can aid in sports decision-making by providing data-backed insights for strategic shot selection. By identifying high-probability players for critical moments, the model supports coaches in making informed, impactful decisions in close-game situations.

## Requirements

To replicate this analysis, you will need the following R packages:
- `tidyverse`
- `caret`
- `randomForest`
- `e1071`
- `glmnet`
- `gbm`

## How to Use

1. Clone the repository.
2. Pre-process the dataset by running `data_preprocessing.R` to filter clutch-time data.
3. Run `model_training.R` to train and evaluate different models.
4. Use `player_analysis.R` to compute player-specific probabilities for clutch shots.
5. Visualize results with `eda_visualization.R`.

## License

This project is licensed under the MIT License.

---

For more detailed information, refer to the full project documentation.
