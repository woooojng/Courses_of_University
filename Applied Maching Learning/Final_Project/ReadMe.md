# Home Credit Default Risk Project

The course project is based on the [Home Credit Default Risk (HCDR) Kaggle Competition](https://github.com/woooojng/courses/tree/main/Applied%20Maching%20Learning/Final_Project).


The aim of this final project on the Home Credit Default Risk dataset is to develop a predictive
model that accurately predicts whether a client will default on a loan. Gradient Boosting models, after hyperparameter tuning and feature selection in the pipeline, outperform baseline Logistic Regression models in accuracy, AUC score, Precision-Recall curve, and confusion matrix evaluations. To address the imbalance issue of our target value, we applied Synthetic Minority Over-sampling Technique (SMOTE). The inclusion of SMOTE further improves the predictive capabilities of logistic regression and gradient boosting models, particularly for 'Default' predictions, across training, validation, and test sets. In the next phase, we introduced a neural network into our pipeline, incorporating a multitask loss function, aiming for improved predictive performance compared to phases 2 and 3, which focused on feature engineering, hyperparameter tuning, and SMOTE utilization with logistic and gradient boosting models. The test accuracy of the neural network reached 91.89$\%$, slightly below the gradient boosting test accuracy of 93.16$\%$.
