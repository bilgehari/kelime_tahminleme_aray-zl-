import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk import download
import tkinter as tk
from tkinter import scrolledtext

# NLTK verilerini indirin (bu bir defaya mahsus bir işlemdir)
download('punkt')
download('stopwords')

# Veri setini yükleyin
data = pd.read_csv('medium_data.csv')

# Stopwords (gereksiz kelimeler) listesini yükleyin
stop_words = set(stopwords.words('english'))

# Veri setindeki başlıkları birleştirin
all_titles = ' '.join(data['title'])

# Tokenize işlemi yapın
words = word_tokenize(all_titles)

# Stopwords'leri ve noktalama işaretlerini temizleyin
filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

# Word2Vec modelini eğitme
word2vec_model = Word2Vec([filtered_words], vector_size=100, window=5, min_count=1, workers=4)

def generate_titles():
    user_input = entry.get().lower()

    # Kullanıcının girdiği kelimenin vokabülerde olup olmadığını kontrol et
    if user_input in word2vec_model.wv:
        user_input_vector = word2vec_model.wv[user_input]
        similar_words = word2vec_model.wv.most_similar(user_input_vector, topn=30)
    else:
        # Kullanıcının girdiği kelime modelde yoksa, benzer kelimeleri bulun
        similar_words = word2vec_model.wv.most_similar(user_input, topn=30)

    # Yeni başlıkları oluşturun
    new_titles = [f"{user_input.capitalize()} {word.capitalize()} in Information Management" for word, _ in
                  similar_words]

    # Sonuçları göster
    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    for title in new_titles:
        result_text.insert(tk.END, title + "\n")
    result_text.config(state=tk.DISABLED)

def clear_results():
    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    result_text.config(state=tk.DISABLED)

def open_medium():
    import webbrowser
    webbrowser.open("https://medium.com/")

def list_dataset_words():
    dataset_words = set(word_tokenize(all_titles.lower()))
    word_list_text.config(state=tk.NORMAL)
    word_list_text.delete('1.0', tk.END)
    for word in dataset_words:
        word_list_text.insert(tk.END, word + "\n")
    word_list_text.config(state=tk.DISABLED)

# Tkinter arayüzü oluştur
root = tk.Tk()
root.title("Başlık Üretici")

# Giriş etiketi ve giriş kutusu
label = tk.Label(root, text="Lütfen bir kelime girin:")
label.pack(pady=10)
entry = tk.Entry(root)
entry.pack(pady=10)

# Başlık Üret ve Temizle düğmeleri
generate_button = tk.Button(root, text="Başlat ve Başlık Üret", command=generate_titles)
generate_button.pack(pady=10)

clear_button = tk.Button(root, text="Temizle", command=clear_results)
clear_button.pack(pady=10)

# Sonuçları görüntülemek için kaydırılabilir metin alanı
result_text = scrolledtext.ScrolledText(root, width=50, height=10, state=tk.DISABLED)
result_text.pack(pady=10)

# Veri setindeki kelimeleri listeleyen düğme
list_words_button = tk.Button(root, text="Veri Setindeki Kelimeleri Listele", command=list_dataset_words)
list_words_button.pack(pady=10)

# Veri setindeki kelimeleri gösteren kaydırılabilir metin alanı
word_list_text = scrolledtext.ScrolledText(root, width=50, height=10, state=tk.DISABLED)
word_list_text.pack(pady=10)

# Medium'a Git düğmesi
medium_button = tk.Button(root, text="Medium'a Git", command=open_medium)
medium_button.pack(pady=10)

# Çıkış düğmesi
exit_button = tk.Button(root, text="Çıkış", command=root.destroy)
exit_button.pack(pady=10)

# Arayüzü başlat
root.mainloop()



