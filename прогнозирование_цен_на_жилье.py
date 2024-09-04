import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
  xs = np.array([1, 2, 3, 4, 5, 6]) # Количество комнат
  ys = np.array([1, 1.5, 2, 2.5, 3, 3.5]) # Цена в миллионах

  model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
  ])

  model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

  model.fit(xs, ys, epochs=500, verbose=0)

  y_new = np.array(y_new)

  return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

# Ответ: [3.9998987]
