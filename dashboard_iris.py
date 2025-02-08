import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

df_iris = pd.read_csv('https://raw.githubusercontent.com/ghina2111/Data-Analysis-Iris/refs/heads/main/Iris.csv')

# Fungsi untuk pertanyaan 1
def question_1(df_iris):
    st.header("Bagaimana distribusi dari setiap fitur (sepal length, sepal width, petal length, petal width) untuk setiap spesies (setosa, versicolor, virginica)?")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Boxplot untuk masing-masing fitur
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    for i, feature in enumerate(features):
        row, col = i // 2, i % 2
        sns.boxplot(data=df_iris, x='Species', y=feature, hue='Species', palette="Set2", ax=axes[row, col], legend=False)
        axes[row, col].set_title(f"Distribusi {feature} untuk Tiap Spesies")

    plt.tight_layout()
    st.pyplot(fig)


# Fungsi untuk pertanyaan 2
def question_2(df_iris):
    st.header("Apakah ada korelasi yang kuat antara fitur-fitur seperti sepal length dan petal length?")
    fig1 = plt.figure(figsize=(8, 6))
    # Scatter plot untuk melihat hubungan langsung antara Sepal Length dan Petal Length
    sns.scatterplot(data=df_iris, x='SepalLengthCm', y='PetalLengthCm', hue='Species', palette='Set1')
    plt.title("Scatter Plot Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    st.pyplot(fig1)
     
     # Menghitung matriks korelasi
    corr_matrix = df_iris.iloc[:, 1:-1].corr()  # Mengambil semua fitur kecuali 'Id' dan 'Species'
    fig2 = plt.figure(figsize=(8,6))
# Membuat heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmap Korelasi Antar Fitur")
    st.pyplot(fig2)

def question_3(df_iris):
    st.header("Jika Anda membangun model klasifikasi sederhana (misalnya, decision tree atau logistic regression), fitur mana yang paling penting untuk membedakan spesies?")
# Mengambil fitur dan label
    X = df_iris.iloc[:, 1:-1]  # Semua fitur tanpa 'Id' dan 'Species'
    y = df_iris['Species']  # Label spesies

# Melatih model Decision Tree sederhana
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

# Mendapatkan pentingnya fitur
    importances = model.feature_importances_

# Visualisasi dalam bentuk bar chart
    fig3 = plt.figure(figsize=(8,6))
    plt.barh(X.columns, importances, color='skyblue')
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance dari Decision Tree")
    plt.gca().invert_yaxis()
    st.pyplot(fig3)

# Main function to run the Streamlit app
def main():
    st.title("Dashboard Analisis mengenai Iris")
    
    question_1(df_iris)
    question_2(df_iris)
    question_3(df_iris)

if __name__ == "__main__":
    main()