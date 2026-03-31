"""
Save models for deployment.
Run this script to train and save the model + preprocessor.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

print("Loading data...")
train_df = pd.read_csv('fraudTrain.csv/fraudTrain.csv')
train_df.drop('Unnamed: 0', axis=1, inplace=True)

train_df['dob'] = pd.to_datetime(train_df['dob'])
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['age'] = (train_df['trans_date_trans_time'] - train_df['dob']).dt.days // 365
train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
train_df['day_of_week'] = train_df['trans_date_trans_time'].dt.dayofweek
train_df['month'] = train_df['trans_date_trans_time'].dt.month
train_df['distance'] = np.sqrt((train_df['lat']-train_df['merch_lat'])**2 + (train_df['long']-train_df['merch_long'])**2)
train_df['log_amt'] = np.log1p(train_df['amt'])

drop_cols = ['trans_date_trans_time','cc_num','merchant','first','last',
             'street','city','state','zip','dob','trans_num','job']

X = train_df.drop(columns=['is_fraud']+drop_cols)
y = train_df['is_fraud']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

print("Sampling and training...")
X_train, _, y_train, _ = train_test_split(X, y, train_size=100000, stratify=y, random_state=42)

X_processed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_processed, y_train)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_smote, y_smote)

joblib.dump(model, 'best_ml_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(categorical_cols, 'categorical_cols.pkl')
joblib.dump(numerical_cols, 'numerical_cols.pkl')

print("Saved: best_ml_model.pkl, preprocessor.pkl, categorical_cols.pkl, numerical_cols.pkl")
print("Done!")
