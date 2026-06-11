import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
df = pd.read_csv('data/transactions.csv')
print("Full dataset shape:", df.shape)
print(df['isFraud'].value_counts())
fraud = df[df['isFraud'] == 1]
normal = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
df = pd.concat([fraud, normal])
print("After sampling:", df.shape)
print(df.head())
print(df['isFraud'].value_counts())  
print(df.isnull().sum())
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
df = pd.get_dummies(df, columns=['type'])   # type column to numbers
print(df.head())
print(df.shape)
X = df.drop('isFraud', axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Before SMOTE:", y_train.value_counts())
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_res.value_counts())
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
import pickle
pickle.dump(rf, open('model.pkl', 'wb'))  # Save Random Forest model
print("Model saved successfully!")