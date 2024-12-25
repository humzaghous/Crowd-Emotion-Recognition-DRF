import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "your_dataset_file.csv"  # Replace with the actual path or URL of your dataset
df = pd.read_csv(url)

# Assuming you have a threshold for rainfall classification, e.g., 500 mm
threshold = 500
df['Rainfall_Class'] = df['ANNUAL'] > threshold

# Extract features (X) and target variable (y)
features = df.drop(columns=["States/UTs", "YEAR", "ANNUAL", "Rainfall_Class"])
target = df["Rainfall_Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features (optional but can be helpful for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {'Algorithm': [], 'Precision': [], 'Recall': [], 'F1 Score': [],
           'MAE': [], 'MSE': [], 'RMSE': [], 'TPR': [], 'FPR': []}


# Function to calculate True Positive Rate (TPR) and False Positive Rate (FPR)
def calculate_tpr_fpr(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return tpr, fpr


# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

results['Algorithm'].append('Random Forest')
results['Precision'].append(precision_score(y_test, rf_predictions))
results['Recall'].append(recall_score(y_test, rf_predictions))
results['F1 Score'].append(f1_score(y_test, rf_predictions))
results['MAE'].append(mean_absolute_error(y_test, rf_predictions))
results['MSE'].append(mean_squared_error(y_test, rf_predictions))
results['RMSE'].append(mean_squared_error(y_test, rf_predictions, squared=False))
tpr, fpr = calculate_tpr_fpr(y_test, rf_predictions)
results['TPR'].append(tpr)
results['FPR'].append(fpr)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

results['Algorithm'].append('KNN')
results['Precision'].append(precision_score(y_test, knn_predictions))
results['Recall'].append(recall_score(y_test, knn_predictions))
results['F1 Score'].append(f1_score(y_test, knn_predictions))
results['MAE'].append(mean_absolute_error(y_test, knn_predictions))
results['MSE'].append(mean_squared_error(y_test, knn_predictions))
results['RMSE'].append(mean_squared_error(y_test, knn_predictions, squared=False))
tpr, fpr = calculate_tpr_fpr(y_test, knn_predictions)
results['TPR'].append(tpr)
results['FPR'].append(fpr)

# SVM
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

results['Algorithm'].append('SVM')
results['Precision'].append(precision_score(y_test, svm_predictions))
results['Recall'].append(recall_score(y_test, svm_predictions))
results['F1 Score'].append(f1_score(y_test, svm_predictions))
results['MAE'].append(mean_absolute_error(y_test, svm_predictions))
results['MSE'].append(mean_squared_error(y_test, svm_predictions))
results['RMSE'].append(mean_squared_error(y_test, svm_predictions, squared=False))
tpr, fpr = calculate_tpr_fpr(y_test, svm_predictions)
results['TPR'].append(tpr)
results['FPR'].append(fpr)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_predictions = nb_model.predict(X_test_scaled)

results['Algorithm'].append('Naïve Bayes')
results['Precision'].append(precision_score(y_test, nb_predictions))
results['Recall'].append(recall_score(y_test, nb_predictions))
results['F1 Score'].append(f1_score(y_test, nb_predictions))
results['MAE'].append(mean_absolute_error(y_test, nb_predictions))
results['MSE'].append(mean_squared_error(y_test, nb_predictions))
results['RMSE'].append(mean_squared_error(y_test, nb_predictions, squared=False))
tpr, fpr = calculate_tpr_fpr(y_test, nb_predictions)
results['TPR'].append(tpr)
results['FPR'].append(fpr)

# Decision Trees
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)

results['Algorithm'].append('Decision Trees')
results['Precision'].append(precision_score(y_test, dt_predictions))
results['Recall'].append(recall_score(y_test, dt_predictions))
results['F1 Score'].append(f1_score(y_test, dt_predictions))
results['MAE'].append(mean_absolute_error(y_test, dt_predictions))
results['MSE'].append(mean_squared_error(y_test, dt_predictions))
results['RMSE'].append(mean_squared_error(y_test, dt_predictions, squared=False))
tpr, fpr = calculate_tpr_fpr(y_test, dt_predictions)
results['TPR'].append(tpr)
results['FPR'].append(fpr)

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results)

# Display the results
print(results_df)
