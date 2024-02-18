import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from contextlib import contextmanager

from sklearn.metrics import roc_auc_score, roc_curve
from data_preprocessing import preprocess_data, remove_outliers, scale_and_encode_features, split_data

from model_training import ModelTrainer

# Thread-safe context manager for displaying a Matplotlib figure
@contextmanager
def st_plt_figure(figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    yield ax
    st.pyplot(fig)
    plt.close(fig)

@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def custom_encode_target(data, target):
    unique_values = data[target].unique()
    mappings = {}

    # If wisconsin, Map Malignant to 1 and Benign to 0
    if target == "diagnosis":
        default_mappings = {'M': 1, 'B': 0}
        data[target] = data[target].map(default_mappings)
        st.sidebar.success(f"Otomatik olarak encoding yapildi, bu dataset icin hardcoded: {default_mappings}")
    else:
        for i, value in enumerate(unique_values):
            numerical_value = st.sidebar.number_input(f"'{value}' degeri icin numerik deger belirle", min_value=0, step=1, key=value, value=i)
            mappings[value] = numerical_value

        data[target] = data[target].map(mappings)
    return data

@st.cache_data
def calculate_correlation(data):
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr()


### PLOTTING START
def plot_correlation_matrix(data: pd.DataFrame, corr, threshold: float = 0.7):
    with st_plt_figure(figsize=(15, 8)):
        sns.heatmap(corr, mask=abs(corr) < threshold, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, square=True, linewidths=0.5)

def plot_scatter(data, x, y, target):
    # If the target is non-categorical, use normal scatter plot
    if data[target].nunique() > 2:
        hue = None
    else:
        hue = target
    with st_plt_figure(figsize=(7, 5)):
        sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=['green', 'red'], alpha=0.4, linewidth=0.7, edgecolor='face')

def plot_confusion_matrix(cm: np.ndarray):
    with st_plt_figure(figsize=(5, 5)):
        sns.heatmap(cm, annot=True, fmt='d', linewidths=0.3, linecolor='red')
        plt.xlabel('y_pred')
        plt.ylabel('y_true')

def plot_pie_chart(data, target):
    # If the target is non-categorical, skip pie chart
    if data[target].nunique() > 6:
        return
    with st_plt_figure(figsize=(5, 5)):
        data[target].value_counts().plot.pie(autopct='%1.1f%%')
        plt.legend(data[target].unique())
        plt.title('Dağılım')

@st.cache_resource
def plot_histograms(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    with st_plt_figure(figsize=(18, 15)) as ax:
        data[numeric_columns].hist(bins=25, figsize=(18, 15), ax=ax)
        
def plot_roc_curve(y_test, y_probs):
    with st_plt_figure():
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = roc_auc_score(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
### PLOTTING END
@st.cache_data
def get_preprocessed_data(data, selected_columns, missing_values_option):
    return preprocess_data(data, selected_columns, missing_values_option)
# Main app
def main():

    # We reduce the padding of sidebar
    st.markdown("""
    <style>
        div.st-emotion-cache-16txtl3.eczjsme4 {
        padding: 2rem 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("Veri Analizi ve Preprocessing")
    st.sidebar.info("Bu uygulama, Wisconsin Breast Cancer datasetine ozellestirilmis bir aplikasyon olsa da, diger sayisal agirlikli classification veri setleri icin de kullanilabilir.")
    st.sidebar.header('Veri Seti Seçimi', divider=True)
    dataset_options = {
        "Breast Cancer Wisconsin (Diagnostic) Data Set": "dataset/bcwds.csv",
    }
    dataset_name = st.sidebar.selectbox("Veri Seti", list(dataset_options.keys()), help="Var olan bir verisetini secin veya kendiniz yukleyin.")
    st.sidebar.write("Veya")
    uploaded_file = st.sidebar.file_uploader("Kendi veri setinizi yukleyin", type=["csv", "txt"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Veri Seti:", uploaded_file.name)
    else:
        data = load_data(dataset_options[dataset_name])
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

    st.header("Data Preprocessing")
    all_columns = data.columns.to_list()
    # Wisconsin default remove
    if 'Unnamed: 32' in all_columns:
        all_columns.remove('Unnamed: 32')
    if 'id' in all_columns:
        all_columns.remove('id')
    selected_columns = st.multiselect("Alakali kolonlar", all_columns, default=all_columns)
    missing_values_option = st.sidebar.selectbox("Eksik kolon iceren verileri ne yapalim?", ["Satiri sil", "Ortalama yap", "Medyan yap", "Kalsin"])
    processed_data = get_preprocessed_data(data, selected_columns, missing_values_option)

    
    st.sidebar.metric("Kullanilan Veri Satir Sayısı", processed_data.shape[0])
    
    st.sidebar.header("Encoding ve Scaling", divider=True)
    st.sidebar.subheader("Hedef Degisken Secimi")
    target = st.sidebar.selectbox("Hedef Değişken", processed_data.columns, index=0)
    st.write("Secilen Y/hedef degeri:", target)
    if target not in processed_data.columns:
        st.error(f"'{target}' column is not in the processed data.")

    enable_custom_encoding = st.sidebar.checkbox("Custom Encoding? ", value=len(processed_data[target].unique()) < 6, help="Eger hedef degiskeniniz binary degilse, custom encoding yapabilirsiniz.")
    if enable_custom_encoding:
        processed_data = custom_encode_target(processed_data, target)

    if st.sidebar.checkbox("Standard Scaler ile Preprocessing yapilsin mi?", value=True):
        should_use_sparse = st.sidebar.checkbox("Scale icin matrix sparsity kullanilsin mi?", value=True, help="Eger veri setiniz cok buyukse, bu secenegi secerek bellek kullanimini azaltabilirsiniz. Naive Bayes icin kullanilmamalidir.")
        preprocessor = scale_and_encode_features(processed_data, target, should_use_sparse)
    else:
        preprocessor = None
    clean_outliers = st.sidebar.checkbox("Otomatik outlier temizligi? (IQR 3)", value=False)

    if clean_outliers:
        veri_sayisi = processed_data.shape[0]
        processed_data = remove_outliers(processed_data, processed_data.columns)
        st.sidebar.write("Temizlenen veri sayisi:", veri_sayisi - processed_data.shape[0])
    st.write("İşlenmiş Veri", processed_data.tail(10))
    st.subheader("Korelasyon Matrisi")
    st.write("Korelasyon matrisi, veri setindeki sütunlar arasındaki ilişkiyi gösterir. Cok fazla sütun varsa, korelasyon matrisi görselleştirilerek sütunlar arasındaki ilişkiler kolayca anlaşılabilir.")
    threshold = st.slider('Korelasyon Eşik Değeri', min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    correlation = calculate_correlation(processed_data)
    with st.spinner("Korelasyon matrisi oluşturuluyor..."):
        plot_correlation_matrix(processed_data, correlation, threshold)
    # Now write the most related features
    st.write("En yüksek korelasyonlu sütunlar")
    st.write(correlation[target].abs().sort_values(ascending=False).head(10))

    st.subheader("Dağılım Grafiği")
    scatter_col_1, scatter_col_2 = st.columns(2)
    with scatter_col_1:
        x_feature = st.selectbox("X Ekseni", processed_data.columns, index=1)
    with scatter_col_2:
        y_feature = st.selectbox("Y Ekseni", processed_data.columns, index=2)
    
    plot_scatter(processed_data, x_feature, y_feature, target)
    plot_pie_chart(processed_data, target)

    st.divider()

    st.title("Model Eğitimi ve Değerlendirme")
    st.sidebar.header("Model Egitimi", divider=True)
    model_choice = st.sidebar.selectbox("Model Seçimi", ["KNN", "SVM", "Naive Bayes"])

    X_train, X_test, Y_train, Y_test = split_data(processed_data, target)

    # Apply preprocessing
    if preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

    st.write("Eğitim seti boyutu:", X_train.shape)
    st.write("Test seti boyutu:", X_test.shape)
    
    st.write(f"Seçilen Model: {model_choice}")
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.progress(0)

    # Model seçimine göre GridSearch ve eğitim
    trainer = ModelTrainer(model_choice, preprocessor)
    trainer.set_progress_bar(st.progress(0))

    multiprocess = st.checkbox("Multiprocessing ile egitim yapilsin mi?", help="Progress bar kullanilamaz. Butun cekirdekler kullanilir.")
    try:
        trainer.train_model(X_train, Y_train, multiprocess)
    except ValueError as e:
        if model_choice == "Naive Bayes":
            st.error("Naive Bayes modeli için preprocessing yaparken sparse matris kullanılamaz.")
        else:
            st.error(str(e))
        return
    metrics, cm, y_probs = trainer.evaluate_model(X_test, Y_test)

    with st.expander("Grid Search CV=5 sonucu en iyi parametreler belirlendi. Modelin en iyi parametreleri"):
        st.json(trainer.best_model.get_params())
    columns = st.columns(4)
    for i, (metric, value) in enumerate(metrics.items()):
        beautiful_val = f"%{value * 100:.2f}" if metric == "Accuracy" else f"{value:.2f}"
        columns[i].metric(metric, beautiful_val, "-Muhtemel Overfit" if value == 1 else "")
    st.sidebar.metric("Accuracy", f"%{metrics['Accuracy'] * 100:.2f}")

    st.subheader("Karışıklık Matrisi")
    plot_confusion_matrix(cm)
    
    st.subheader("ROC Eğrisi")
    roc_col_1, roc_col_2 = st.columns([0.7, 0.3])
    if y_probs is not None and len(np.unique(Y_test)) <= 2:
        with roc_col_1:
            plot_roc_curve(Y_test, y_probs)
        with roc_col_2:
            st.metric("AUC(Area Under Curve) Score", round(roc_auc_score(Y_test, y_probs), 3))
    else:
        st.write("ROC eğrisi çizilemiyor, çünkü modelin `predict_proba` metodu yok veya multiclass bir dataset.")
    
    st.sidebar.divider()
    st.sidebar.subheader("Mustafa Akcanca")


if __name__ == "__main__":
    main()
