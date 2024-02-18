from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st


def remove_outliers(data, columns):
    for col in columns:
        try:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        except TypeError:
            st.warning(f"{col} sutunu numerik olmadigi icin aykiri degerler temizlenemedi.")            

    return data

def split_data(data, target):
    Y = data[target]
    X = data.drop(target, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, Y_train, Y_test

def preprocess_data(data, selected_columns, missing_values_option):
    data = data[selected_columns]
    # Handling missing values
    if missing_values_option == "Satiri sil":
        data = data.dropna()
    elif missing_values_option == "Ortalama yap":
        numeric_cols = data.select_dtypes(include='number').columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    elif missing_values_option == "Medyan yap":
        numeric_cols = data.select_dtypes(include='number').columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Ensure the data is in dense format
    if hasattr(data, 'to_dense'):  # For Pandas DataFrame with sparse columns
        data = data.to_dense()
    elif hasattr(data, 'toarray'):  # For Scipy sparse matrices
        data = data.toarray()
    return data

def scale_and_encode_features(data, target, should_use_sparse):
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    if target in numeric_features:
        numeric_features.remove(target)  # Remove target variable from numeric features if present
    # Scaling for numeric features
    scaler = StandardScaler()
    
    # Encoding for categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_features),
            ('cat', encoder, categorical_features)
        ], sparse_threshold=0.3 if should_use_sparse else 0)

    return preprocessor
