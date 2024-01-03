import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Örnek veri seti
corpus = [
    'Bilge bir kız.',
    'Her kız güzeldir.',
    'Ve zeki bir kız güzeldir.',
    'Güzel kızların aklı da güzeldir.',
    'Zeki kızlar, güzel ve akıllı olabilir.',
    'Her kızın kendine özgü güzellik ve zekası vardır.',
    'Zeki kızlar, güzellikleriyle dikkat çeker.',
    'Güzel gülüşler, içsel zekayı yansıtabilir.',
    'Zeki kızların güzelliği kalpte iz bırakır.',
    'Güzel kızlar sadece dış değil, iç güzellikleriyle de önemlidir.',
    'Zeki kızların güzelliği, bilgelikle süslenmiş gibidir.',
    'Güzellik, içsel zeka ile tamamlanır.',
    'Her kızın güzelliği, zekası ve benzersizliğiyle belirlenir.'
]

# Tokenizer oluştur
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Giriş verilerini ve etiketleri oluştur
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Giriş ve etiketleri ayır
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Modeli oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=450, verbose=1)

def kelime_tahminle():
    seed_text = "Şöyle ki "  # Başlangıç metni
    next_words = 5

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    output.config(state=tk.NORMAL)
    output.delete('1.0', tk.END)
    output.insert(tk.END, seed_text)
    output.config(state=tk.DISABLED)

# Ana pencereyi oluştur
window = tk.Tk()
window.title("Kelime Tahminleme Projesi")

# Tahminleme butonu
predict_button = tk.Button(window, text="Başlat ve Tahmin Et", command=kelime_tahminle)
predict_button.pack(pady=10)

# Çıkış alanı
output = scrolledtext.ScrolledText(window, width=60, height=5, state=tk.DISABLED)
output.pack(pady=10)

# Çıkış butonu
exit_button = tk.Button(window, text="Çıkış", command=window.destroy)
exit_button.pack(pady=10)

# Pencereyi çalıştır
window.mainloop()

