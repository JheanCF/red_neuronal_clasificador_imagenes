#%%
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np
from red_neuronal_imagenes import modelo
import cv2

def categorizar(url):
  respuesta = requests.get(url)
  img = Image.open(BytesIO(respuesta.content))
  img = np.array(img).astype(float)/255

  img = cv2.resize(img, (224,224))
  prediccion = modelo.modelo.predict(img.reshape(-1, 224, 224, 3))
  return np.argmax(prediccion[0], axis=-1)
# %%
#0 = cuchara, 1 = cuchillo, 2 = tenedor
url = 'https://image.shutterstock.com/image-photo/sharp-not-touch-chefs-kitchen-260nw-1281029980.jpg' #debe ser 2
prediccion = categorizar(url)
print(prediccion)

# %%
