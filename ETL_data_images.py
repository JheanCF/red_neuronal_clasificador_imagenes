# %%
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os
# %% 

plt.figure(figsize=(15,15))
carpeta = r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\cuchillos'
imagenes = os.listdir(carpeta)

for i, nombreimagen in enumerate(imagenes[:25]):
  plt.subplot(5,5,i+1)
  imagen= mping.imread(carpeta+'/'+nombreimagen)
  plt.imshow(imagen)
# %%
import shutil

## en esta celda se copia de la carpeta fuente de las imagenes al dataset
carpeta_fuente=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\cucharas'
carpeta_destino=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\dataset\cuchara'
imagenes = os.listdir(carpeta_fuente)

for i, nombreimagen in enumerate(imagenes):
  if i < 289:
    #copia de la carpeta fuente a la carpeta destino
    shutil.copy(carpeta_fuente + '/' + nombreimagen, carpeta_destino + '/'+ nombreimagen)

#%%
## en esta celda se copia de la carpeta fuente de las imagenes al dataset
carpeta_fuente=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\cuchillos'
carpeta_destino=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\dataset\cuchillo'
imagenes = os.listdir(carpeta_fuente)

for i, nombreimagen in enumerate(imagenes):
  if i < 289:
    #copia de la carpeta fuente a la carpeta destino
    shutil.copy(carpeta_fuente + '/' + nombreimagen, carpeta_destino + '/'+ nombreimagen)
#%%
carpeta_fuente=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\tenedores'
carpeta_destino=r'C:\Users\Usuario\Dropbox\Ingenieria_informatica\redes_neuronales\dataset\tenedor'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimagen in enumerate(imagenes):
  if i < 289:
    #copia de la carpeta fuente a la carpeta destino
    shutil.copy(carpeta_fuente + '/' + nombreimagen, carpeta_destino + '/'+ nombreimagen)

#%%