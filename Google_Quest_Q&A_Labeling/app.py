from flask import Flask, jsonify, request
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, LSTM, GlobalAveragePooling1D
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig


import flask
app = Flask(__name__)

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    return phrase

def preprocess(text):
    """
    this function performs the preprocessing on the text by converting to lower case, removing html tags, preform decontractions,
    remove special characters from the input text
    """
    preprocessed_text = text
    preprocessed_text = preprocessed_text.lower()
    preprocessed_text = re.sub('<[^>]*>','',preprocessed_text)
    preprocessed_text = decontractions(preprocessed_text)
    preprocessed_text = re.sub('[^A-Za-z0-9 ]+', ' ', preprocessed_text)
    preprocessed_text = re.sub(' +', ' ', preprocessed_text)
    return preprocessed_text

def prepare_question_answer_string(question_title, question_body, answer, tokenizer ,max_question_length = 254,max_answer_length = 254):
  question = question_title + question_body
  question = tokenizer.tokenize(question)
  answer = tokenizer.tokenize(answer)
  if len(question) >= max_question_length and len(answer) >= max_answer_length:
    new_max_length_question = int(max_question_length / 2)
    question = question[:new_max_length_question] + question[-new_max_length_question:]
    new_max_length_answer = int(max_answer_length / 2)
    answer = answer[:new_max_length_answer] + answer[-new_max_length_answer:]
  elif len(question) < max_question_length and len(answer) < max_answer_length:
    pass
  elif len(question) < max_question_length and len(answer) >= max_answer_length:
    max_answer_length = max_answer_length + (max_question_length - len(question))
    new_max_length_answer = int(max_answer_length / 2)
    answer = answer[:new_max_length_answer] + answer[-new_max_length_answer:]
  elif len(question) >= max_question_length and len(answer) < max_answer_length:
    max_question_length = max_question_length + (max_answer_length - len(answer))
    new_max_length_question = int(max_question_length / 2)
    question = question[:new_max_length_question] + question[-new_max_length_question:]

  return question, answer

def transform_dataset_1(question_title, question_body, answer, tokenizer, max_sequence_length = 512):
  question_tokens, answer_tokens = prepare_question_answer_string(question_title, question_body, answer, tokenizer)
  input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]'] + ['[PAD]'] * (max_sequence_length - (len(question_tokens) + len(answer_tokens) + 3))
  input_sequence = tokenizer.convert_tokens_to_ids(input_tokens)
  mask_sequence = [1] * ((len(question_tokens) + len(answer_tokens) + 3)) + [0] * (max_sequence_length - (len(question_tokens) + len(answer_tokens) + 3))
  segmentation_sequence = [0] * (len(question_tokens) + 2) + [1] * (len(answer_tokens) + 1) + [0] * (max_sequence_length - (len(question_tokens) + len(answer_tokens) + 3))
  return np.array(input_sequence), np.array(mask_sequence), np.array(segmentation_sequence)

def prepare_question_string(question_title, question_body, tokenizer,max_question_length = 510):
  question = question_title + question_body
  question = tokenizer.tokenize(question)
  if len(question) > max_question_length:
    question = question[:max_question_length]
  return question

def transform_dataset_2(question_title, question_body , tokenizer, max_sequence_length = 512):
  question_tokens = prepare_question_string(question_title, question_body, tokenizer)
  input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + ['[PAD]'] * (max_sequence_length - (len(question_tokens)  + 2))
  input_sequence = tokenizer.convert_tokens_to_ids(input_tokens)
  mask_sequence = [1] * ((len(question_tokens) + 2)) + [0] * (max_sequence_length - (len(question_tokens) + 2))
  segmentation_sequence = [0] * (len(question_tokens) + 2) + [0] * (max_sequence_length - (len(question_tokens) + 2))
  return np.array(input_sequence), np.array(mask_sequence), np.array(segmentation_sequence)

def jsonify_result(result):
  target_labels = []
  with open(r'target_labels.txt', 'r') as fp:
      for line in fp:
          x = line[:-1]
          target_labels.append(x)
  Dict = {}
  for tv, value in zip(target_labels,result.flatten()):
    Dict[tv] = str(value)
  return jsonify(Dict)


@app.route('/')
def hello_world():
    return 'hello_world'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form.to_dict()
    question_title = preprocess(data['question_title'])
    question_body = preprocess(data['question_body'])
    answer = preprocess(data['answer'])
    input_sequence_d1, mask_sequences_d1, segmentation_sequence_d1 = transform_dataset_1(question_title, question_body , answer, tokenizer)
    input_sequence_d2, mask_sequences_d2, segmentation_sequence_d2 = transform_dataset_2(question_title, question_body , tokenizer)
    datapoint = {'input_word_ids_dataset_1': input_sequence_d1.reshape(1,-1),
                'input_mask_dataset_1': mask_sequences_d1.reshape(1,-1),
                'segment_ids_dataset_1': segmentation_sequence_d1.reshape(1,-1),
                'input_word_ids_dataset_2': input_sequence_d2.reshape(1,-1),
                'input_mask_dataset_2': mask_sequences_d2.reshape(1,-1),
                'segment_ids_dataset_2': segmentation_sequence_d2.reshape(1,-1)}
    results = gq_model.predict(datapoint)
    return jsonify_result(results)

def create_model():
  max_seq_length = 512
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  input_word_ids_dataset_1 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids_dataset_1")
  input_mask_dataset_1 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask_dataset_1")
  segment_ids_dataset_1 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids_dataset_1")

  input_word_ids_dataset_2 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids_dataset_2")
  input_mask_dataset_2 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask_dataset_2")
  segment_ids_dataset_2 = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids_dataset_2")

  bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
  bert_layer= TFBertModel.from_pretrained("bert-base-uncased", config = bert_config)

  result_dataset_1 = bert_layer([input_word_ids_dataset_1,input_mask_dataset_1,segment_ids_dataset_1])
  result_dataset_2 = bert_layer([input_word_ids_dataset_2,input_mask_dataset_2,segment_ids_dataset_2])

  # Last 4 hidden layers of bert
  h12_dataset_1 = tf.reshape(result_dataset_1['hidden_states'][-1][:,0],(-1,1,768))
  h11_dataset_1 = tf.reshape(result_dataset_1['hidden_states'][-2][:,0],(-1,1,768))
  h10_dataset_1 = tf.reshape(result_dataset_1['hidden_states'][-3][:,0],(-1,1,768))
  h09_dataset_1 = tf.reshape(result_dataset_1['hidden_states'][-4][:,0],(-1,1,768))
  concat_hidden_dataset_1 = tf.keras.layers.Concatenate(axis=2)([h12_dataset_1, h11_dataset_1, h10_dataset_1, h09_dataset_1])
  x_dataset_1 = GlobalAveragePooling1D()(concat_hidden_dataset_1)

  # Last 4 hidden layers of bert_q
  h12_dataset_2 = tf.reshape(result_dataset_2['hidden_states'][-1][:,0],(-1,1,768))
  h11_dataset_2 = tf.reshape(result_dataset_2['hidden_states'][-2][:,0],(-1,1,768))
  h10_dataset_2 = tf.reshape(result_dataset_2['hidden_states'][-3][:,0],(-1,1,768))
  h09_dataset_2 = tf.reshape(result_dataset_2['hidden_states'][-4][:,0],(-1,1,768))
  concat_hidden_dataset_2 = tf.keras.layers.Concatenate(axis=2)([h12_dataset_2, h11_dataset_2, h10_dataset_2, h09_dataset_2])
  x_dataset_2 = GlobalAveragePooling1D()(concat_hidden_dataset_2)


  x_question_answer = Dense(30, activation='sigmoid')(x_dataset_1)
  x_answer = Dense(9, activation='sigmoid')(x_dataset_1)
  x_question = Dense(21, activation='sigmoid')(x_dataset_2)
  x_question_answer_seperate =  tf.keras.layers.Concatenate(axis=-1)([x_question,x_answer])
  x_final = tf.keras.layers.Average()([x_question_answer, x_question_answer_seperate])

  model = Model(inputs = [input_word_ids_dataset_1,input_mask_dataset_1,segment_ids_dataset_1,input_word_ids_dataset_2,input_mask_dataset_2,segment_ids_dataset_2], outputs = x_final, name = 'BERT_Google_Quest')
  model.load_weights('gq_model_2_weights.h5')
  return   max_seq_length,tokenizer,model
max_seq_length,tokenizer,gq_model = create_model()

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)