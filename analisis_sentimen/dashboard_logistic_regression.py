import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import os

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score

script_dir = os.path.dirname(os.path.realpath(__file__))

ori_df = pd.read_excel(f'{script_dir}/dataset/fix_dataset.xlsx', index_col = 0)

data = pd.read_csv(f'{script_dir}/dataset/clean_fix_dataset.csv', index_col = 0)

data.dropna(axis = 0, inplace = True)

data = data[data['labeling'] != 'netral']

sw_df = pd.read_csv(f'{script_dir}/dataset/stopwords_indonesian.csv', index_col = 0)

X_raw = data["clean_text"]
y_raw = data["labeling"]

X_train, X_test, y_train, y_test = train_test_split(X_raw.values, y_raw.values, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit(X_train)
    
X_train_TFIDF = vectorizer.transform(X_train).toarray()
X_test_TFIDF  = vectorizer.transform(X_test).toarray()

kolom = vectorizer.get_feature_names_out()

train_tf_idf = pd.DataFrame(X_train_TFIDF, columns=kolom)
test_tf_idf = pd.DataFrame(X_test_TFIDF, columns=kolom)

# -------------------------------------------------------------------------------------------------
lr_all = LogisticRegression()
lr_all.fit(X_train_TFIDF, y_train)

y_pred_all = lr_all.predict(X_test_TFIDF)

kolom_all = ['negatif', 'positif']
confm_all = confusion_matrix(y_test, y_pred_all)
df_cm_all = pd.DataFrame(confm_all, index = kolom_all, columns = kolom_all)

# -------------------------------------------------------------------------------------------------
# chi square 500 fitur
chi2_500_features = SelectKBest(chi2, k=500)
X_500_best_features = chi2_500_features.fit_transform(train_tf_idf, y_train)

lr500 = LogisticRegression()
lr500.fit(X_500_best_features, y_train)

X_test_500_chi2 = chi2_500_features.transform(X_test_TFIDF)

y_pred_500 = lr500.predict(X_test_500_chi2)

kolom_500 = ['negatif', 'positif']
confm_500 = confusion_matrix(y_test, y_pred_500)
df_cm_500 = pd.DataFrame(confm_500, index = kolom_500, columns = kolom_500)

# -------------------------------------------------------------------------------------------------
# chi square 1543 fitur
chi2_1543_features = SelectKBest(chi2, k=1543)
X_1543_best_features = chi2_1543_features.fit_transform(train_tf_idf, y_train)

lr1543 = LogisticRegression()
lr1543.fit(X_1543_best_features, y_train)

X_test_1543_chi2 = chi2_1543_features.transform(X_test_TFIDF)

y_pred_1543 = lr1543.predict(X_test_1543_chi2)

kolom_1543 = ['negatif', 'positif']
confm_1543 = confusion_matrix(y_test, y_pred_1543)
df_cm_1543 = pd.DataFrame(confm_1543, index = kolom_1543, columns = kolom_1543)

# -------------------------------------------------------------------------------------------------

# Dashboard
menu = st.sidebar.selectbox('Pilih Menu:', ['Introduction', 'Model Performance', 'Insights'])

st.title('Sentimen Analisis Ketertarikan Pembelian Mobil Listrik SUZUKI Menggunakan Metode Klasifikasi Logistic Regression')

if menu == 'Introduction' :
    st.header('Introduction')
    
    with st.container() :
        st.video('https://www.youtube.com/watch?v=q1QA0VVSDwY')

        st.write('**Deskripsi**')
        st.markdown('''
                    <div style="text-align: justify;">
                    Beberapa waktu lalu, Suzuki memperkenalkan pertama kali seri terbaru mobil konsep listrik evX di India dan pertama kali di Indonesia 
                    pada acara Gaikindo Indonesia International Auto Show (GIIAS) 2024 yang diluncurkan perdana pada Rabu, 17 Juli 2024. Beberapa konten 
                    kreator telah mengulas konsep mobil listrik Suzuki ini pada kanal Youtube mereka dan menimbulkan pembahasan menarik pada kolom komentar, 
                    menunjukan ketertarikan dan kritik untuk mobil listrik tersebut. Penulis tertarik untuk melakukan analisis sentimen terhadap komentar 
                    pada video tersebut.
                    </div>''', unsafe_allow_html = True)
    
    with st.container() : 
        st.write('**Dataset**')  
        st.dataframe(ori_df)
    
        st.write('**Stop Words Bahasa Indonesia**')
        st.dataframe(sw_df, hide_index = True, use_container_width = True)
    

elif menu == 'Insights' :
    st.header('Insights')
    
    st.markdown("<h3 style='text-align: center;'>Jumlah Sentimen tiap Channel</h3>", unsafe_allow_html=True)
    with st.container() :
        sentimen_tiap_channel = pd.pivot_table(index = 'nama_channel', columns = 'labeling', aggfunc = 'size', data = ori_df, fill_value = 0)

        fig, ax = plt.subplots()
        sentimen_tiap_channel.plot(kind = 'bar', ax = ax)
        
        st.pyplot(fig)

        with st.expander('Insights') :
            st.write('Bla-bla')
        
    colors = sns.color_palette("viridis", n_colors = 3)
    
    st.markdown("<h3 style='text-align: center;'>WordCloud Sentimen Positif</h3>", unsafe_allow_html=True)
    

    data_positif = data[data['labeling'] == 'positif']

    with st.container() :
        text_positif = ' '.join(data_positif["clean_text"].values.tolist())

        wordcloud_positif = WordCloud(width = 800, height = 800, background_color = 'white', stopwords = None, min_font_size = 10).generate(text_positif)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud_positif, interpolation='bilinear')
        ax.axis('off')

        st.write(fig)
        
        with st.expander('Insights') :
            st.write('Bla-bla')
                

    st.markdown("<h3 style='text-align: center;'>WordCloud Sentimen Negatif</h3>", unsafe_allow_html=True)

    data_negatif = data[data['labeling'] == 'negatif']

    with st.container() :
        text_negatif = ' '.join(data_negatif["clean_text"].values.tolist())

        wordcloud_negatif = WordCloud(width = 800, height = 800, background_color = 'white', stopwords = None, min_font_size = 10).generate(text_negatif)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud_negatif, interpolation='bilinear')
        ax.axis('off')

        st.write(fig)
        
        with st.expander('Insights') :
            st.write('Bla-bla')
    
    with st.container() :
        
        st.markdown("<h4 style='text-align: center;'>Top 10 Kata Kunci dengan Nilai Rata-rata TF-IDF (Data Training)</h4>", unsafe_allow_html=True)

        # top10
        mean_tfidf = train_tf_idf.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()

        ax.bar(mean_tfidf.index, mean_tfidf.values, color = colors)
        ax.set_ylabel('Rata-rata Nilai TF-IDF')
        
        plt.xticks(rotation = 45)

        st.write(fig)
        
        with st.expander('Insights') :
            st.write('Bla-bla')
    
    with st.container() :
            
        st.markdown("<h4 style='text-align: center;'>Top 10 Kata Kunci dengan Nilai Rata-rata TF-IDF (Data Testing)</h4>", unsafe_allow_html=True)

        # top10
        mean_tfidf = test_tf_idf.mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
            
        ax.bar(mean_tfidf.index, mean_tfidf.values, color = colors)
        ax.set_ylabel('Rata-rata Nilai TF-IDF')
        
        plt.xticks(rotation = 45)

        st.write(fig)
        
        with st.expander('Insights') :
            st.write('Bla-bla')

elif menu == 'Model Performance' :
    
    st.header('Model Performance')
    
    st.markdown('''
                Model yang digunakan adalah Gaussian Naive Bayes, dan yang dipertimbangkan adalah jumlah fitur yang akan dipilih sebagai data yang siap untuk dilatih. Berikut adalah perbedaan antara dua pendekatan yang digunakan:
                - Semua Fitur
                - Pemilihan Fitur Menggunakan Chi-Square untuk 500 Fitur
                - Pemilihan Fitur Menggunakan Chi-Square untuk K-Fitur  
                
                Kita dapat melihat perbedaan performa melalui menu di bawah ini.''')
    
    
    options = ['---', "Semua Fitur", "Pemilihan fitur Menggunakan Chi-Square untuk 500 Fitur", "Pemilihan Fitur Menggunakan Chi-Square untuk K-Fitur"]
    selected_option = st.selectbox("Pilih salah satu opsi :", options, index = 0)
    
    if selected_option == options[1] :
    
        st.subheader("Semua Fitur")
        st.markdown("<h4 style='text-align: left;'>Number of Features</h4>", unsafe_allow_html=True)
        st.metric(value = "2241 Feature")

        with st.container() :
            fig, ax = plt.subplots()

            st.markdown("<h4 style='text-align: left;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            sns.heatmap(df_cm_all, cmap = 'Greens', annot=True, fmt=".0f", ax = ax)
            ax.set_xlabel('Sentimen Sebenarnya')
            ax.set_ylabel('Sentimen Prediksi')

            ax.invert_xaxis()

            st.pyplot(fig)

            st.markdown("<h4 style='text-align: left;'>Classification Report</h4>", unsafe_allow_html=True)
            st.text(classification_report(y_test, y_pred_all))

        with st.expander('Insights') :
            st.write('Bla-bla')
            
    elif selected_option == options[2] :
    
        st.subheader("Pemilihan fitur Menggunakan Chi-Square untuk 500 Fitur")
        st.markdown("<h4 style='text-align: left;'>Number of Features</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1 :
            st.metric(value = f'{train_tf_idf.shape[1]} Feature', label = '**Sebelum**')

        with col2 :
            st.metric(value = f'{X_500_best_features.shape[1]} Feature', label = '**Sesudah**')

        with st.container() :
            fig, ax = plt.subplots()

            st.markdown("<h4 style='text-align: left;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            sns.heatmap(df_cm_500, cmap = 'Greens', annot=True, fmt=".0f", ax = ax)
            ax.set_xlabel('Sentimen Sebenarnya')
            ax.set_ylabel('Sentimen Prediksi')

            ax.invert_xaxis()

            st.pyplot(fig)

            st.markdown("<h4 style='text-align: left;'>Classification Report</h4>", unsafe_allow_html=True)
            st.text(classification_report(y_test, y_pred_500))

        with st.expander('Insights') :
            st.write('Bla-bla')
        
    elif selected_option == options[3] :
    
        st.subheader("Pemilihan Fitur Menggunakan Chi-Square untuk K-Fitur")
        
        with st.container() :
            st.markdown("<h4 style='text-align: left;'>Eksperimen</h4>", unsafe_allow_html=True)

            st.image(f"{script_dir}/image/logistic_regression_plot.png", caption = "Plot Eksperimen k-feature", use_column_width=True)
        
        with st.expander('Insights') :
            st.write('Bla-bla')
        
        st.markdown("<h4 style='text-align: left;'>Best K-Features</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Base on Accuracy
        with col1 :
            st.metric(value = f'1543 Features', label = '**Base on Accuracy**')

        # Base on F1-Score
        with col2 :
            st.metric(value = '1543 Features', label = '**Base on F1-Score**')

        with st.container() :
            fig, ax = plt.subplots()
    
            st.markdown("<h4 style='text-align: left;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            sns.heatmap(df_cm_1543, cmap = 'Greens', annot=True, fmt=".0f", ax = ax)
            ax.set_xlabel('Sentimen Sebenarnya')
            ax.set_ylabel('Sentimen Prediksi')
    
            ax.invert_xaxis()
    
            st.pyplot(fig)
    
            st.markdown("<h4 style='text-align: left;'>Classification Report</h4>", unsafe_allow_html=True)
            st.text(classification_report(y_test, y_pred_1543))
        
        with st.expander('Insights') :
            st.write('Bla-bla')
