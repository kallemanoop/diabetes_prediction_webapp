import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


file_path = r'C:\Users\kalle\Desktop\Anoop-Projects\Kailash\diabetes.csv'
diabetes_data = pd.read_csv(file_path)


missing_values = diabetes_data.isnull().sum()


X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
  

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=25)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)


accuracy_rf = accuracy_score(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
joblib.dump(rf_model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# print("Random Forest Model Accuracy:", accuracy_rf)
# print("Classification Report:\n", class_report_rf)
print("Model and Scaler saved successfully!")