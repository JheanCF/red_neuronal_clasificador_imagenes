from traitlets.traitlets import validate
#aumento de datos con imagedatgenetor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os

# crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    validation_split=0.2 # 20% para pruebas
)
#Generador para datasets entranamiento y pruebas

data_gen_entrenamiento = datagen.flow_from_directory(r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory(r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='validation')
#imprimir 10 imagenes del generador de entrenamiento

for imagen, etiqueta in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i])
  break
plt.show()

#%%
import tensorflow as tf
import tensorflow_hub as hub
#descargar el modelo de tensor flow
url ='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
mobilnetV2=hub.KerasLayer(url, input_shape=(224,224,3))

# Congelar el modelo descargado
mobilnetV2.trainable=False
#%%

modelo = tf.keras.Sequential([
    mobilnetV2,
    tf.keras.layers.Dense(3, activation='softmax')
    ])
#%%
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#%%
EPOCAS=50

historial = modelo.fit(
    data_gen_entrenamiento, epochs= EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas
)
# %%
#Graficas de precisión
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(50)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

# %%
