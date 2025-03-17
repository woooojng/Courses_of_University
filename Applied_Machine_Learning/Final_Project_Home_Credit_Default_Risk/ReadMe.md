# Home Credit Default Risk Project

The course project is based on the [Home Credit Default Risk (HCDR) Kaggle Competition](https://github.com/woooojng/courses/tree/main/Applied%20Maching%20Learning/Final_Project).


The aim of this final project on the Home Credit Default Risk dataset is to develop a predictive
model that accurately predicts whether a client will default on a loan. We got the following results for each Phase.

- Phase2: EDA work(A visualization of each of the input and target features (looking at the distribution, and the central tendencies as captured by the mean, median etc.), A visualization of the correlation analysis, A graphic summary of the missing value analysis) and base pipeline work for logistic regression model.

- Phase3: To address the imbalance issue identified in the Home Credit Default Risk dataset, Synthetic Minority Over-sampling Technique (SMOTE) was applied. Both logistic regression and gradient boosting models, after hyperparameter tuning, feature selection, and SMOTE, demonstrated high accuracy for 'No Default' predictions across training, validation, and test sets. The inclusion of SMOTE improved the 'Default' prediction capabilities of both models compared to those without SMOTE, leading to improved classification performance. Models were evaluated based on logistic regression (LR) and gradient boosting (GB) with hyperparameter tuning and feature selection. Models incorporating feature selection alongside hyperparameter tuning showed slightly improved Test AUC scores compared to those without feature selection. Models applying hyperparameter tuning, feature selection, and SMOTE showed substantial improvement in F1 scores on the test set, with the gradient boosting model achieving the highest Test F1 Score and Test AUC among all models.

- Phase4: In this phase, a neural network was implemented and tested with a multitask loss function. The neural network aimed to achieve better predictions compared to the previous phases where feature engineering, hyperparameter tuning, and SMOTE were employed for logistic and gradient boosting models. The test accuracy of the neural network was 91.89%, slightly lower than the gradient boosting test accuracy of 93.16%.
