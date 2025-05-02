# ML_Project

Random Forest Classifier from Scratch
Project Overview
This project implements a Random Forest Classifier from scratch using Python and NumPy. The objective is to manually build the logic behind a Random Forest model without relying on machine learning libraries like Scikit-learn for the core implementation.

The dataset used is the Bank Marketing dataset from the UCI Machine Learning Repository, which contains various client and campaign-related features to predict whether a customer subscribed to a term deposit.

Work Completed
Data Preprocessing:

Loaded and cleaned the dataset by dropping rows containing 'unknown' values.

Encoded all categorical columns manually into numeric values.

Mapped the target column (subscribed) to binary values: 'yes' as 1 and 'no' as 0.

Decision Tree Implementation:

Created a DecisionTree class supporting both Gini impurity and Entropy as criteria.

Added parameters for max_depth, min_samples_split, and min_samples_leaf.

Built the tree recursively with logic to find the best splits based on the selected criterion.

Random Forest Implementation:

Created a RandomForest class that builds multiple decision trees on bootstrap samples.

Implemented majority voting to combine the predictions from all trees.

Included parameters for:

Number of trees (n_trees)

Maximum tree depth

Minimum samples per split

Minimum samples per leaf

Criterion (Gini or Entropy)

Model Training and Evaluation:

Split the dataset into training and testing sets (80% train, 20% test).

Trained the custom Random Forest on the training data.

Made predictions and printed the first 10 predicted vs. true labels.

Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix from Scikit-learn’s metrics module.

Scikit-learn Comparison
To benchmark and validate the custom implementation, I added two new code blocks that:

Train Scikit-learn’s Random Forest:

Used RandomForestClassifier from Scikit-learn with the same parameters (n_estimators=10, max_depth=10) for fair comparison.

Trained on the same X_train and y_train data.

Made predictions on the test set.

Evaluate and Compare:

Printed out the accuracy, confusion matrix, and classification report for both:

Scikit-learn’s Random Forest

Custom-built Random Forest

This comparison helped verify that the custom model performs reasonably well when compared to Scikit-learn’s highly optimized version.
