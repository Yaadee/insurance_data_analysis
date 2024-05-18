import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_data(df):
    df = df.fillna(df.mean())
    df = df.dropna()
    return df

def encode_categorical_data(df, categorical_columns):
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)
    return df

def feature_engineering(df):
    df['ClaimRatio'] = df['TotalClaims'] / df['TotalPremium']
    return df

def prepare_data(df, target_column, test_size=0.3):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
