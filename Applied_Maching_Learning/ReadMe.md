# P 556: APPLIED MACHINE LEARNING Course in Computer Science Department

### Final Project : Home Credit Default Risk Project
The course project is based on the Home Credit Default Risk (HCDR) Kaggle Competition.
The aim of this final project on the Home Credit Default Risk dataset is to develop a predictive model that accurately predicts whether a client will default on a loan. We got the following results for each Phase.

#### On the Phase3 out of 4 phases, I worked as a leader for this team. The following is the goal and achievement of our team on this phase3.
Phase3: To address the imbalance issue identified in the Home Credit Default Risk dataset,
- Synthetic Minority Over-sampling Technique (SMOTE) was applied.
- Both logistic regression and gradient boosting models, after hyperparameter tuning, feature selection, and SMOTE, demonstrated high accuracy for 'No Default' predictions across training, validation, and test sets.
- The inclusion of SMOTE improved the 'Default' prediction capabilities of both models(logistic regression (LR) and gradient boosting (GB)) compared to those without SMOTE
- SMOTE improved classification performance. Models were evaluated based on logistic regression (LR) and gradient boosting (GB) with hyperparameter tuning and feature selection.
- Models incorporating feature selection alongside hyperparameter tuning showed slightly improved Test AUC scores compared to those without feature selection.
- Models applying hyperparameter tuning, feature selection, and SMOTE showed substantial improvement in F1 scores on the test set, with the gradient boosting model achieving the highest Test F1 Score and Test AUC among all models.

### HW 02 - KNN: classification, regression + EDA + ML pipelines
Outcomes and Goals:
- Describe the K-nearest-neighbors (KNN) algorithm for both regression and classification
- Articulate the algorithmic steps for training and testing for KNN (regression/classification)
- Understand image processing and classification tasks and apply KNN to these
- Master how hyperparameter selection can be accomplished via crossfold validation
- Describe implementations that make for efficient KNN learning and prediction
- Understand how to be Pythonic when coding up KNN (no FOR loops if possible)
- Implementing KNN and weighted KNN as classes

### HW 03 - Optimization theory
Outcomes and Goals:
- Define Optimization Theory, Constrained Optimization (uni/multivariate), Unconstrained optimization (uni/multivariate), Convex optimization (uni/multivariate)
- Depict the taxonomy of multivariate optimization problems 
- Given a problem, either univariate or multivariate, find the optimal solution (aka root-finding of the gradient function) using
- Gradient descent
- Frame machine learning as an optimization problem
- Explain why machine learners are sometimes known as root finders
- Distinguish between analytical and numerical optimization approaches to solving optimization problems
- Given a problem, either univariate or multivariate, find the optimal solution (aka root finding) using the Newton-Raphson
- Gradient descent by approximating the gradient numerically

### HW 06 - Probabilistic Approaches to Machine Learning
Outcomes and Goal
- Define probability basics including the following: Probability Axioms, Conditional probabilities, Product Rule, Chain Rule, and Bayes Rule.
- Derive Bayes Rule from scratch.
- Define a Bayesian network
- Define independence and conditional independence as it relates to Bayesian Networks
- Define and derive Naïve Bayes for different flavors:
- Multinomial Naïve Bayes vs. Bernoulli Naïve Bayes
- Discrete input variables vs. Continuous input variables 
- Laplace smoothing
- On paper, learn a multinomial Naïve Bayes model from data for different types of problems (e.g., spam detection) 
- Evaluate the learned multinomial Naïve Bayes model using metrics such as accuracy, and confusion matrices.
- Learn to process textual data using  SKlearn CountVectorizer (i.e., go from raw text to document by term matrix)
- Be able to configure a text classification pipeline (where the core classifier is a multinomial naive Bayes classifier)
- These goals are optional, for learners who want to understand the theory and implementation details: 
- Derive the closed form solution to maximum likelihood estimate for Naive Bayes
- Implement theNaive Bayes learning algorithm  (real variables, and nominal variables)
- Multinomial naive bayes
- Bernoulli  naive Bayes
- Evaluate classification results using metrics such as accuracy, and confusion matrices.  precision & recall, and F1 score s:

### HW 07 - Classification: logistic/softmax regression
Outcomes and Goals:
- Describe and apply basic operations on vectors (review)
- define a hyperplane
- define and calculate the distance from a point to a hyperplane
- binary classification (binomial logistic regression)
- multiclass (multinomial logistic regression)
- regularized logistic regression models (lasso, ridge, elastic net)
- Score a test case for a binomial or multinomial logistic regression model:
- Calculate perpendicular distances
- Make all distances positive by exponentiation
- Convert these to probabilities via normalization
- Express logistic regression in graph and matrix form
- Define logistic loss and cross-entropy (CXE)
- Define and derive the gradient for bi/multinomial logistic regression objective functions
- These goals are optional, for learners who want to understand the theory and implementation details. 
- Define and write code to calculate perpendicular distance to a separating hyperplane
- Implement bi/multinomial logistic regression learning algorithm from from scratch
- Extend it to penalized logistic regression
- Derive the logistic regression gradient from scratch
- Approximate the gradient numerically

### HW 10 - Perceptron and Support Vector machines
Outcomes and Goals:
- Describe and apply basic operations on vectors (review)
- Heaviside step function
- Sgn function
- Learn how to build  linear classifiers for Multinomial/Multiclass problems (as opposed to binary class problems)
- Define a perceptron model for the following:
- binary classification (binomial perceptron)
- multiclass (multinomial perceptron)
- Define and write code to calculate perpendicular distance to a separating hyperplane
- Define the margin for a training example
- Score a test case for a binomial or a multinomial perceptron model:
- Calculate perpendicular distances
- Classification rule binomial/multinomial 
- Express perceptron in graph and matrix form
- Define 0-1 loss, perceptron loss
- Define the unifying loss framework for classification and regression based machine learning
- Write and apply the gradient for bi/multinomial perceptron objective functions
- Approximate the gradient numerically
- Support vector machine (SVM)
- Define an SVM, support vectors
- Sub-gradient descent algorithm for SVM (stretch goal!)
- These goals are optional, for learners who want to understand the theory and implementation details. 
- Implement bi/multinomial perceptron learning algorithm from scratch
- Extend it to a penalized perceptron
- Derive the bi/multinomial perceptron  gradient from scratch

### HW 11 - Neural Networks, Multi-Layer Perceptrons (MLP), deep learning in PyTorch
Outcomes and Goals:
- What is a  neural network
- What is deep learning
- What is the difference or not between neural networks and deep learning
- Enumerate applications of neural networks
- Describe  multi-layer perceptrons
- Provide an executive summary of PyTorch
- Define a tensor and do tensor algebra in PyTorch
- Develop and train a linear model (for regression or classification) in  PyTorch 
- Develop and train a multi-layer perceptron for regression or classification in  PyTorch 
- Explain at a high level how machine learned models are just equations that can be programmatically expressed as expressions
- Explain what is autodiff and how it fits into learning a linear model
- These goals are optional for learners who want to understand the theory and implementation details. 
- Implement multi-layer perceptron in PyTorch from scratch
- Extend it to a penalized multi-layer perceptron
