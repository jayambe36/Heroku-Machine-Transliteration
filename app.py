from flask import *

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import jsonpickle

import pickle

app = Flask(__name__)

df = pd.read_csv('test.csv')
df = df.dropna()

english_sentence = list(df["english_sentence"])
hindi_sentence = list(df["hindi_sentence"])

def seq2vec(text, main):
	lol = dict()

	for t in text:
		lol[t] = main[t]

	return lol

def tokenize(sentence):
	text_tokenizer = Tokenizer()

	text_tokenizer.fit_on_texts(sentence)

	return text_tokenizer.texts_to_sequences(sentence), text_tokenizer

def logits_to_sentence2(logits, tokenizer):
	index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
	index_to_words[0] = '<empty>'
	#print([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


	print("~~~~> ", np.argmax(logits, 1))

	return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

@app.route('/', methods=["POST", "GET"])
def home():

	return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
	if(request.method == "POST"):
		#print("+-----------------------------------------+")
		#print([x for x in request.form.values()])
		#print("+-----------------------------------------+")

		text = request.form['eng'].lower().split(' ')

		eng_text_tokenized, eng_text_tokenizer = tokenize(english_sentence)
		hin_text_tokenized, hin_text_tokenizer = tokenize(hindi_sentence)

		max_english_len = len(max(eng_text_tokenized, key=len))
		max_hindi_len = len(max(hin_text_tokenized, key=len))

		english_pad_sentence = pad_sequences(eng_text_tokenized, max_english_len, padding="post")
		hindi_pad_sentence = pad_sequences(hin_text_tokenized, max_hindi_len, padding="post")

		english_pad_sentence = english_pad_sentence.reshape(*english_pad_sentence.shape, 1)
		hindi_pad_sentence = hindi_pad_sentence.reshape(*hindi_pad_sentence.shape, 1)

		res = seq2vec(text, eng_text_tokenizer.word_index)
#		print(res)

#		print(">>>>>> ", max_english_len)

		resValue = [v for k, v in res.items()]
#		print(resValue)
		resValue = pad_sequences([resValue], max_english_len, padding="post")
		resValue = resValue.reshape(*resValue.shape, 1)
#		print(resValue)

		print("\n")
		print(">....... Model ...........<\n")
		# Load model
		model = pickle.load(open('model.pkl', 'rb'))
		resModel = model.predict(resValue[0])

		finalRes = logits_to_sentence2(resModel[0], hin_text_tokenizer)
		print(finalRes)

		return render_template('index.html', predict=finalRes)

if __name__ == "__main__":
	app.run(debug=True)