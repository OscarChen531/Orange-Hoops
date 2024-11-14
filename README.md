# Orange Hoops Data Science Contest

This project uses NCAA Men’s Basketball data from the 2023-2024 season to recommend the optimal player for Purdue Men’s Basketball to take the final shot in clutch game situations. Through machine learning and data analysis, the project identifies **Zach Edey** as the player with the highest success probability under high-pressure conditions. The analysis is based on key game variables and situational predictors, including time remaining, score difference, and win probability.

## Project Overview
In basketball, deciding who should take the last shot in a close game is critical. This project aims to provide a data-driven solution by identifying the Purdue player most likely to succeed in clutch moments, defined as the last five minutes of the game with a score differential within five points.

Six machine learning models were evaluated for their predictive power in shot success, with the **Support Vector Machine (SVM)** model achieving the best performance. The SVM model was used to calculate each player’s likelihood of making a successful shot under specific game conditions, leading to the selection of Zach Edey as the primary recommendation.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Data Description
This project uses NCAA Men’s Basketball Division 1 data from the 2023-2024 season. The data includes:
- **Game Situational Variables**: Score difference, time remaining, win probability, possession factors, and other game context.
- **Player Performance Data**: Shot success rates under varying conditions, broken down by player and shot type (e.g., three-point attempts, free throws).

The target variable for modeling is `shot_success`, defined as either "Success" (made shot) or "Fail" (missed shot).

## Exploratory Data Analysis
The exploratory data analysis provides insights into factors affecting shot success under clutch conditions. Key visualizations include:
- **Shot Success by Shooter**: Compares shot success rates by player.
- **Three-Point and Free Throw Success Rates**: Assesses players’ effectiveness across shot types.
- **Shot Success Over Time Remaining**: Shows how success probability changes as time diminishes.
- **Score Difference and Win Probability**: Explores shot success based on the game’s competitiveness and pre-shot possession situations.
- **Feature Plot**: Displays density plots of predictors like score difference and time remaining, showing conditions favoring shot success.

## Modeling Approach
The following models were trained and evaluated for shot success prediction:
1. **Logistic Regression** and **Ridge Regression**: Baseline models with limited accuracy.
2. **Support Vector Machine (SVM)**: The highest-performing model with an accuracy of 66.7%.
3. **Random Forest and Gradient Boosting Machine (GBM)**: Ensemble models with accuracy close to SVM.
4. **k-Nearest Neighbors (kNN)**: A non-linear model with lower performance.

Models were assessed based on **Accuracy**, **Kappa**, **Sensitivity**, **Specificity**, **AUC**, and **F1 Score**. A composite score, derived from normalized and weighted metrics, determined that **SVM** was the optimal model for player recommendations.

## Results
The SVM model’s predictions consistently identified **Zach Edey** as the Purdue player most likely to make successful shots in clutch scenarios. Edey’s high probability scores across various game conditions validated him as the optimal choice for high-stakes shots, providing an objective, data-backed recommendation for critical game decisions.

## Conclusion
This project demonstrates that data-driven approaches can enhance decision-making in sports by identifying players with the highest probability of success in clutch moments. Zach Edey’s selection as the optimal player for game-winning shots is grounded in machine learning analysis, showing how sports analytics can provide valuable, actionable insights.

## Usage
To replicate or use this analysis:
1. **Prepare the Data**: Ensure NCAA game data is structured to include shot outcomes, game context, and player information.
2. **Run the Analysis**: Train and evaluate models using the provided code files.
3. **Interpret Results**: Use the SVM model output to determine the optimal player for high-pressure situations.

## Dependencies
- `R`: The project uses R for data manipulation, visualization, and model training.
- `caret`: For training machine learning models.
- `ggplot2`: For creating visualizations.
- `dplyr`: For data wrangling and manipulation.
- `pROC`: For calculating AUC scores in model evaluation.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
