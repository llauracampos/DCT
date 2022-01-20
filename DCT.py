# -*- coding: utf-8 -*-
"""
Projeto 2 - Introdução ao Processamento Digital de Imagens
Autores: Laura Campos, Joison Oliveira e Wendson Carlos

"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.image as mpimg
from skimage import io
#%matplotlib inline
import PIL.Image
import cv2
import math
import soundfile as sf
from tqdm import tqdm
from numba import jit, njit


#############################################################################
#                 DEFINIÇÃO DAS FUNÇÕES PARA IMAGEM                         #
#############################################################################

@jit
def DCT(x):  

    N = len(x)
    
    dct = np.zeros(N)
    for k in range(N):
        som = 0
        
        if (k == 0):
            ck = sqrt(1.0/2.0)
        elif(k > 0):
            ck = 1.0
            
        for n in range(N):
            som += x[n] * cos((k*pi*n/N) + (k*pi/2.0*N))
            
        dct[k] = sqrt(2.0/N)*som*ck 

    return dct

@jit
def IDCT(x):  

    N = len(x)
    
    idct = np.zeros(N)
    for n in range(N):
        som = 0
            
        for k in range(N):
            
            if (k == 0):
                ck = sqrt(1.0/2.0)
            elif(k > 0):
                ck = 1.0
            
            som += x[k] * ck * cos((k*pi*n/N) + (k*pi/2.0*N))
            
        idct[n] = sqrt(2.0/N)*som
            
    return idct

def sortKey(e):
    return e[1]

#############################################################################
#               IMPORTAÇÃO DA IMAGEM E CONFIGURAÇÕES INICIAIS               #
#############################################################################

#Leitura da imagem
img = io.imread('lena256.png')

#Parametros iniciais
N = len(img)  
img_DCT = np.zeros(img.shape)
img_AC = np.zeros(img.shape)

for i, linha in tqdm(enumerate(img)):
    img_DCT[i] = DCT(linha)
    
img_DCT = img_DCT.transpose()

for i, linha in tqdm(enumerate(img_DCT)):
    img_DCT[i] = DCT(linha)


#############################################################################
#    Questão 1.1 - Módulo normalizado da DCT de I (sem DC)  e valor DC      #
#############################################################################

#Exibir imagem com e sem o DC

print("Matriz DCT:")
print(img_DCT)

plt.figure()
plt.imshow(img_DCT, cmap='gray')
plt.title("Matriz com DC")

img_DC = img_DCT[0][0]
print("Valor DC:")
print(img_DC)

img_AC = img_DCT.copy()
img_AC[0][0] = 0

print("Matriz sem DC")
print(img_AC)

plt.figure()
plt.imshow(img_AC, cmap='gray')
plt.title("Matriz sem DC")

#Processo de normalização

normalizado = np.zeros(img.shape)
normalizado_DC = np.zeros(img.shape)

#Modulo normalizado da DCT de I sem o nível DC
for i in range(len(img_AC)):
    for j in range(len(img_AC[i])):
        normalizado[i][j] = math.log(abs(img_AC[i][j]) + 1)


#Modulo normalizado da DCT de I com o nível DC
for i in range(len(img_DCT)):
    for j in range(len(img_DCT[i])):
        normalizado_DC[i][j] = math.log(abs(img_DCT[i][j]) + 1)
        
        
plt.figure()
plt.imshow(normalizado, cmap='gray')
plt.title("Modulo normalizado da DCT sem DC")

print("Modulo normalizado da DCT sem DC:")
print(normalizado)

img_DC_norm = normalizado_DC[0][0]
print("Valor DC normalizado:")
print(img_DC_norm)


#############################################################################
#    Questão 1.2 - Aproximação de I preservanndo o DC e AC importantes      #
#############################################################################

#Aplicação de uma filtragem de frequências

img_filtro = img_DCT.copy()

img_sorted =  list(np.ndenumerate(normalizado_DC))

img_sorted.sort(key=sortKey)

img_invertida = img_sorted[::-1]

img_freq_100 = [[] for _ in range(100)]

for i in range(100):
    img_freq_100[i] = img_invertida[i][0]

for i in range(len(normalizado_DC)):
    for j in range(len(normalizado_DC[i])):
        if (i, j) not in img_freq_100:
            if (i== 0 and j==0):
                img_filtro[i][j] = img_filtro[i][j]
            else:
                img_filtro[i][j] = 0
            

print("Matriz com cortes de frequência:")
print(img_filtro)

#Aplicação da IDCT na imagem com filtragem

img_IDCT = np.zeros(img.shape)

for i, linha in tqdm(enumerate(img_filtro)):
    img_IDCT[i] = IDCT(linha)
    
img_IDCT = img_IDCT.transpose()

for i, linha in tqdm(enumerate(img_IDCT)):
    img_IDCT[i] = IDCT(linha)
    
  
plt.figure()
plt.imshow(img_IDCT, cmap='gray')
plt.title("Imagem com cortes de frequência")
        
cv2.imwrite("Imagem_filtrada_100.png", img_IDCT)

#############################################################################
#               Questão 2 - Reforço dos graves de um sinal                  #
#############################################################################

jit
def filtro_grave(x):
    
    nivel = 6
    fc = 12520 
    g = 0.5

    N = len(x)
    #lista ck do tamanho N, no primeiro elemento raiz 0.5 e o resto é 1
    ck = [np.sqrt(1/2)]
    ck.extend([1]*(N-1))

    output = np.zeros(N)
    aux= []
    for k in range(N):
        aux.append(np.pi*((g/(np.sqrt(1 + (k/fc)**(2*nivel)))+1)))
    
    for k in range(N):
        output[k] = x[k] * aux[k]
    return output


#lendo o arquivo de audio do formato Wav
signal, samplerate = sf.read('e.wav')

plt.figure('signal', figsize=[30,6])
plt.plot(signal, linewidth=0.5,alpha=1,color="purple")
plt.title("Audio Original")
plt.ylabel("Amplitude")


y_dct = DCT(signal)  

plt.figure('y_dct', figsize=[30,6])
plt.plot(y_dct, linewidth=0.5,alpha=1,color="yellow")
plt.title("Audio com DCT")
plt.ylabel("Amplitude")

versaoFinal = np.zeros(y_dct.shape)
versaoFinal = filtro_grave(y_dct)

plt.figure('versão final dct', figsize=[30,6])
plt.plot(versaoFinal, linewidth=0.5,alpha=1,color="black")
plt.title("Filtro grave com DCT")
plt.ylabel("Amplitude")
plt.show()

versaoFinal = IDCT(versaoFinal)

plt.figure('versão final idct', figsize=[30,6])
plt.plot(versaoFinal, linewidth=0.5,alpha=1,color="green")
plt.title("Filtro grave depois de passar IDCT")
plt.ylabel("Amplitude")
plt.show()

sf.write('ult.wav', versaoFinal, samplerate)

