#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import json
import re
import sys

#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

with open('CardVocab.json') as f:
    vocabDict = json.load(f)
    
char2idx = vocabDict["char2idx"]
idx2char = np.asarray(vocabDict["idx2char"])

model = tf.keras.models.load_model('cardGenerator')
model.compile(optimizer='adam')

def generate_text(model, start_list, temperature):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_list]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
    
        text_generated.append(idx2char[predicted_id])
        
        # Break at end of card character
        if predicted_id == char2idx["$"]:
            break

    return ' '.join(text_generated[:-1])

print("Welcome to the Magic card text generating neural network.")
while True:
    inStr = input("Please enter the start of a card text (or press enter if you want a completely random entry. \n")
    scrubbedList = re.sub('( )+', ' ', re.sub('([:|;|,|\.|\(|\)|\n|—|"])', r' \1 ', inStr)).split()
    inList = []
    for word in scrubbedList:
            inList += [word]
    try:
        temp = float(input("Enter a craziness factor between .1 and 10 (bigger produces weirder text). "))
    except:
        continue
    print(inStr + " " + generate_text(model, ["^"] + inList, temp) +"\n")

    cont = input("Would you like to create another card text? (y or n) ")
    if 'y' not in cont.lower():
        break