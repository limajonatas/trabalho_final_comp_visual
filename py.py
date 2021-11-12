import cv2 as cv
import numpy as np
import py2 #arquivo que contem funcoes 


img = cv.imread('multipla_escolha.png')

#definir tamanho de imagem e aplica
larguraImg = 510
alturaImg = 700
img = cv.resize(img, (larguraImg, alturaImg))

#pré-processamento
imgCinza = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #escala de cinza
imgBlur = cv.GaussianBlur(imgCinza, (5, 5), 1) #efeito glaussiano #imagem, tamanho do núcleo, sigma X
imgCanny = cv.Canny(imgBlur, 10, 50) #threshold 

#encontrando os contornos na imagem
imgCnt= img.copy() #copia imagem original
countours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(imgCnt, countours, -1, (0, 255,0), 10)

rectCon = py2.rectContour(countours)
biggestContour = rectCon[0]
print(len(biggestContour))

imgBlank = np.zeros_like(img)
#ADICIONANDO IMAGENS EM UM ARRAY
imgArray = ([img, imgCinza, imgBlur, imgCanny],  [imgCnt, imgBlank, imgBlank, imgBlank])
#CHAMANDO A FUNCAO PARA EMPILHAR AS IMAGENS
imgStacked = py2.empilharImagens(imgArray, 0.5) 

#mostra janela com a imagem original
cv.imshow('Imagens', imgStacked)


cv.waitKey(0)