import tensorflow as tf
import numpy as np
import pandas as pd

loaded = tf.saved_model.load("Model")
model = loaded.signatures["serving_default"]
datas = pd.read_csv("data/centroid.csv")
datas = datas.to_numpy()
correct_predictions = 0
for i in range(len(datas)):
    data = tf.expand_dims(tf.convert_to_tensor(datas[i], dtype=np.float32), axis=0)
    correct_predictions += (np.argmax(model(data)['predictions'])==i)

print(' model accuracy is %f%% (Number of test samples=%d)' % (
        (correct_predictions * 100) / len(datas), len(datas)))