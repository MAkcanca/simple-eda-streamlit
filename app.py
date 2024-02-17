import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from contextlib import contextmanager as cxmanager

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics._scorer import accuracy_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Constants for model parameter grids
KNN_PARAM_GRID = {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
SVM_PARAM_GRID = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'sigmoid']}
NB_PARAM_GRID = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
# Global variable to track progress
current_progress = 0
total_progress = 0

# Thread-safe context manager for displaying a Matplotlib figure
@cxmanager
def st_plt_figure(figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    yield ax
    st.pyplot(fig)
    plt.close(fig)

def my_accuracy_scorer(estimator, X, y_true):
    # This is a hack to update the progress bar, as the `GridSearchCV` class does not provide a callback for each iteration
    global current_progress
    score = accuracy_scorer(estimator, X, y_true)
    # Update progress
    current_progress += 1
    st.session_state.progress_bar.progress(current_progress / total_progress)
    return score

@st.cache_data
def calculate_total_combinations(param_grid):
    # Calculate all combinations of parameter values
    all_names = sorted(param_grid)
    combinations = itertools.product(*(param_grid[name] for name in all_names))
    # Count the number of combinations
    total_combinations = sum(1 for _ in combinations)
    return total_combinations

@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def preprocess_data_old(data: pd.DataFrame) -> pd.DataFrame:
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data.drop('id', axis=1, inplace=True)
    data.drop('Unnamed: 32', axis=1, inplace=True)
    # Drop diagnosis none
    data = data.dropna(subset=['diagnosis'])
    return data


def preprocess_data(data):
    st.header("Data Preprocessing")
    
    # Column selection
    all_columns = data.columns.tolist()
    # Wisconsin default remove
    if 'Unnamed: 32' in all_columns:
        all_columns.remove('Unnamed: 32')
    if 'id' in all_columns:
        all_columns.remove('id')

    selected_columns = st.multiselect("Hangi kolonlar dahil edilsin", all_columns, default=all_columns)
    data = data[selected_columns]

    # Handling missing values
    missing_values_option = st.sidebar.selectbox("Eksik satir iceren verileri ne yapalim?", ["Satiri sil", "Ortalama yap", "Medyan yap", "Kalsin"])
    if missing_values_option == "Satiri sil":
        data = data.dropna()
    elif missing_values_option == "Ortalama yap":
        data = data.fillna(data.mean())
    elif missing_values_option == "Medyan yap":
        data = data.fillna(data.median())

    # Data type conversion (if needed)
    # ... [This part can be added based on specific requirements]

    return data

def scale_and_encode_features(data, target):
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
        ])

    return preprocessor

def select_target_variable(data):
    st.sidebar.header("Feature Scaling and Encoding")
    st.sidebar.header("Hedef Degisken Secimi")
    target = st.sidebar.selectbox("Hedef degisken/Y degerini secin", data.columns, index=0)
    return target


def custom_encode_target(data, target):
    st.sidebar.subheader(f"'{target}' Icin Custom Encoding")
    unique_values = data[target].unique()
    mappings = {}

    enable_custom_encoding = st.sidebar.checkbox("Custon Encoding?", len(unique_values) < 4)

    if enable_custom_encoding:
        for i, value in enumerate(unique_values):
            numerical_value = st.sidebar.number_input(f"'{value}' degeri icin numerik deger belirle", min_value=0, step=1, key=value, value=i)
            mappings[value] = numerical_value

        if len(unique_values) <= 2:
            default_mappings = {val: idx for idx, val in enumerate(unique_values)}
            data[target] = data[target].map(default_mappings)
            st.sidebar.success(f"Otomatik olarak encoding yapildi: {default_mappings}")

        if st.sidebar.button("Custom Encoding Uygula"):
            data[target] = data[target].map(mappings)
            st.sidebar.success("Custom Encoding Uygulandi!")

    return data

@st.cache_data
def calculate_correlation(data):
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr()

def plot_correlation_matrix(data: pd.DataFrame, threshold: float = 0.7, corr=None):
    with st_plt_figure(figsize=(15, 8)):
        if corr is None:
            corr = calculate_correlation(data)
        sns.heatmap(corr, mask=abs(corr) < threshold, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, square=True, linewidths=0.5)

def plot_scatter(data, x, y, target):
    with st_plt_figure(figsize=(7, 5)):
        sns.scatterplot(x=x, y=y, hue=target, data=data, palette=['green', 'red'], alpha=0.4, linewidth=0.7, edgecolor='face')

def plot_confusion_matrix(cm: np.ndarray):
    with st_plt_figure(figsize=(5, 5)):
        sns.heatmap(cm, annot=True, fmt='d', linewidths=0.3, linecolor='red')
        plt.xlabel('y_pred')
        plt.ylabel('y_true')

def plot_pie_chart(data, target):
    with st_plt_figure(figsize=(5, 5)):
        #data['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%')
        data[target].value_counts().plot.pie(autopct='%1.1f%%')
        #plt.legend(['Benign', 'Malignant'])
        plt.legend(data[target].unique())
        plt.title('Dağılım')

def plot_histograms_old(processed_data):
    selected_features = st.multiselect("Feature listesi", list(processed_data.columns), default=processed_data.columns[0])
    for feature in selected_features:
        if not feature:
            continue
        plt.figure(figsize=(8, 8))
        st.write(f"Histogram {feature}")
        sns.histplot(processed_data[feature], kde=True, color='blue')
        st.pyplot(plt)
        plt.close()


def plot_histograms(data):
    with st_plt_figure(figsize=(20, 15)) as ax:
        data.hist(bins=50, figsize=(20, 15), ax=ax)


def plot_roc_curve(y_test, y_probs):
    with st_plt_figure():
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = roc_auc_score(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')

def split_data(data, target):
    Y = data[target]
    X = data.drop(target, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, Y_train, Y_test

def perform_grid_search(estimator, param_grid, X_train, y_train):
    # Progress bar
    global total_progress, current_progress
    total_progress = calculate_total_combinations(param_grid) * 5
    current_progress = 0

    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring=my_accuracy_scorer)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # if Target is multiclass, use 'macro' average
    average = 'binary' if len(np.unique(y_test)) <= 2 else 'macro'
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=average),
        "Recall": recall_score(y_test, y_pred, average=average),
        "F1 Score": f1_score(y_test, y_pred, average=average)
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_probs

# Main app
def main():
    st.sidebar.header("About the App")
    st.sidebar.info("This app analyzes the Breast Cancer Wisconsin dataset. You can explore the data, train models, and evaluate their performance.")
    st.sidebar.header('Veri Seti Seçimi')
    dataset_options = {
        "Breast Cancer Wisconsin (Diagnostic) Data Set": "dataset/bcwds.csv",
    }
    dataset_name = st.sidebar.selectbox("Veri Seti", list(dataset_options.keys()), help="Var olan bir verisetini secin veya kendiniz yukleyin.")
    st.sidebar.write("Veya")
    uploaded_file = st.sidebar.file_uploader("Kendi veri setinizi yukleyin", type=["csv", "txt"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = load_data(dataset_options[dataset_name])

    st.title("Veri Analizi ve Preprocessing")
    st.write("Veri Seti:", dataset_name)
    st.write("Toplam Veri Sayısı:", data.shape[0])

    st.subheader("Veri Seti Önizleme")

    st.write("İlk 10 Satır", data.head(10))
    st.write("Sütunlar", data.columns)

    st.subheader("Veri seti hakkında bilgi")
    st.write(data.describe())
    st.write("Veri setinde, her bir hücrenin ölçümleri ve hücrelerin kanserli olup olmadığını gösteren 'diagnosis' sütunu bulunmaktadır. Bu sutunlari, Malign/M 1 ve Benign/B 0 olacak sekilde duzenleyelim. Ayrica ID ve Unnamed: 32 adında iki sütun bulunmaktadır. Bu sütunlar veri seti için gereksiz olduğu için çıkarılacaktır.")
    
    st.write("Veri setindeki histogramlari inceleyerek outlier analizi yapabiliriz.")
    with st.expander("Veri Seti Histogramlari"):
        plot_histograms(data)
    processed_data = preprocess_data(data)
    target = select_target_variable(processed_data)
    st.write("Secilen Y/hedef degeri:", target)
    processed_data = custom_encode_target(processed_data, target)

    if st.sidebar.checkbox("Standard Scaler ile Preprocessing yapilsin mi?", value=True):
        preprocessor = scale_and_encode_features(processed_data, target)
    else:
        preprocessor = None

    st.write("İşlenmiş Veri", processed_data.tail(10))

    st.sidebar.header("Model Egitimi")
    model_choice = st.sidebar.selectbox("Model Seçimi", ["KNN", "SVM", "Naive Bayes"])

    st.subheader("Korelasyon Matrisi")
    st.write("Korelasyon matrisi, veri setindeki sütunlar arasındaki ilişkiyi gösterir. Cok fazla sütun varsa, korelasyon matrisi görselleştirilerek sütunlar arasındaki ilişkiler kolayca anlaşılabilir.")
    threshold = st.slider('Korelasyon Eşik Değeri', min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    correlation = calculate_correlation(processed_data)
    with st.spinner("Korelasyon matrisi oluşturuluyor..."):
        plot_correlation_matrix(processed_data, threshold, correlation)
    # Now write the most related features
    st.write("En yüksek korelasyonlu sütunlar")
    st.write(correlation[target].abs().sort_values(ascending=False).head(10))

    st.subheader("Dağılım Grafiği")
    x_feature = st.selectbox("X Ekseni", processed_data.columns, index=1)
    y_feature = st.selectbox("Y Ekseni", processed_data.columns, index=2)
    plot_scatter(processed_data, x_feature, y_feature, target)#'radius_mean', 'texture_mean')
    plot_pie_chart(processed_data, target)

    st.divider()
    st.title("Model Eğitimi ve Değerlendirme")
    
    X_train, X_test, Y_train, Y_test = split_data(processed_data, target)

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    st.write("Eğitim seti boyutu:", X_train.shape)
    st.write("Test seti boyutu:", X_test.shape)
    
    st.write(f"Seçilen Model: {model_choice}")
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.progress(0)

    # Model seçimine göre GridSearch ve eğitim
    if model_choice == "KNN":
        model = perform_grid_search(KNeighborsClassifier(), KNN_PARAM_GRID, X_train, Y_train)
    elif model_choice == "SVM":
        model = perform_grid_search(SVC(probability=True), SVM_PARAM_GRID, X_train, Y_train)
    else:  # Naive Bayes
        model = perform_grid_search(GaussianNB(), NB_PARAM_GRID, X_train, Y_train)

    with st.expander("Grid Search CV=5 sonucu en iyi parametreler belirlendi. Modelin en iyi parametreleri"):
        st.json(model.get_params())

    metrics, cm, y_probs = evaluate_model(model, X_test, Y_test)
    columns = st.columns(4)
    for i, (metric, value) in enumerate(metrics.items()):
        beautiful_val = f"%{value * 100:.2f}" if metric == "Accuracy" else f"{value:.2f}"
        columns[i].metric(metric, beautiful_val, "-Muhtemel Overfit" if value == 1 else "")
    st.sidebar.metric("Accuracy", f"%{metrics['Accuracy'] * 100:.2f}")

    st.subheader("Karışıklık Matrisi")
    plot_confusion_matrix(cm)

    st.subheader("ROC Eğrisi")
    if y_probs is not None:
        plot_roc_curve(Y_test, y_probs)
        st.metric("AUC Score", roc_auc_score(Y_test, y_probs))
    else:
        st.write("ROC eğrisi çizilemiyor, çünkü modelin `predict_proba` metodu yok.")


if __name__ == "__main__":
    main()
