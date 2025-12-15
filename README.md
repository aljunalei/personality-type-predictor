🧠 Personality Type Predictor

📖 Overview

The Personality Type Predictor is a machine learning project designed to classify individuals into one of the 16 MBTI (Myers-Briggs Type Indicator) personality types (e.g., INTP, ESFJ, ENFP).

By analyzing demographic details and psychometric scores, the model identifies patterns in user behavior to predict their likely personality profile with an accuracy of approximately 80%.

🚀 How It Works

The system takes in specific user attributes and processes them through a trained Multinomial Logistic Regression model.

🔹 Input Data Features

The model analyzes the following 8 features:

Age (Numerical)

Gender (Categorical: Male/Female)

Education (Categorical Level)

Introversion Score (Psychometric)

Sensing Score (Psychometric)

Thinking Score (Psychometric)

Judging Score (Psychometric)

Interest (Categorical: e.g., Technology, Arts, Sports)

🔹 The Pipeline

Data Preprocessing: - Categorical variables (Gender, Interest) are encoded.

Numerical features are normalized using StandardScaler to ensure optimal model performance.

Model Training: - Trained using Multinomial Logistic Regression with the saga solver for efficiency on large datasets.

Optimized via hyperparameter tuning (regularization C=0.5).

Prediction: - The model outputs the specific MBTI classification (e.g., INFP).

📊 Performance & Visualization

Accuracy: Achieved ~79.77% accuracy on the test dataset.

Analysis: Utilized Seaborn and Matplotlib to visualize correlation matrices and confusion matrices, identifying key relationships between psychometric scores and personality outcomes.

🛠️ Tech Stack

Language: Python

Machine Learning: Scikit-learn (Logistic Regression, StandardScaler, Train-Test Split)

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Statistical Analysis: Statsmodels

💻 Usage Example

import numpy as np

# Sample Input: [Age, Gender, Education, Introversion, Sensing, Thinking, Judging, Interest]
new_sample = np.array([[21, 1, 0, 4, 5, 6, 3, 3]])

# Preprocess and Predict
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)

print(f'Predicted Personality: {prediction[0]}')
# Output: Predicted Personality: INFP
****
