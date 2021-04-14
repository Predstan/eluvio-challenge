#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from embed import embed
import sys


dataset = pd.read_csv("data/new_set.csv")
loaded = tf.saved_model.load("Model")
model = loaded.signatures["serving_default"]
def main():
    sentence = ""
    arg = sys.argv[1:]
    for i in range(len(arg)):
        sentence += str(arg[i]) + " "
        if i == len(arg) - 1:
            sentence = sentence[:-1]
    embedding = embed([sentence])
    prediction = np.argmax(model(embedding)['predictions'])
    get_top(prediction)

def get_top(prediction):
    top = dataset[dataset["centroid"] == prediction]
    sort_date_created = top.sort_values('date_created', ascending=False)
    new = sort_date_created.sort_values('up_votes', ascending=False).head(50)

    print("RECOMMENDED NEWS".center(70))
    for i in range(len(new)):
        print(f"{i+1}.  {new['title'].to_numpy()[i]}, created on {new['date_created'].to_numpy()[i]}, by {new['author'].to_numpy()[i]}")


if __name__ == "__main__":
    main()