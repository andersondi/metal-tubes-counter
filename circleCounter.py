import cv2
import numpy as np

def nothing():
    pass

imgOriginal = cv2.imread('resources/tubes3.jpg')

# elemento estrutural necessario como parametro para operações de mudança morfologica
kernel = np.ones((3, 3), np.uint8)

# Cria uma janela
cv2.namedWindow('Morphed')

cv2.createTrackbar('Treshold','Morphed',0,255,nothing)

cv2.createTrackbar('Iteractions','Morphed',0,10,nothing)

while(1):

    # Clone original image to not overlap drawings
    clone = imgOriginal.copy()
    
    # Converte a imagem para a escala de cinzas. Isso é feito para retirar detalhes desnecessarios
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    # imgGray = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_DEFAULT)

    # Pega a posicao atual das barras de monitoramento
    r = cv2.getTrackbarPos('Treshold','Morphed')
    iter_num = cv2.getTrackbarPos('Iteractions', 'Morphed')
    
    # Converte a imagem para preto e branco binarizado
    ret, imgThresholded = cv2.threshold( imgGray, r, 255, cv2.THRESH_BINARY_INV )
    
    # Operacao de erosao para reduzir ruidos na imagem. Essa operacao reduz tambem as regioes de interesse.
    morph = cv2.erode(imgThresholded, kernel, iterations=iter_num)

    # Operacao de dilatacao para transformar as regioes de interesse para um tamanho proximo do original.
    morph = cv2.dilate(morph, kernel, iterations=iter_num)
    
    # Faz a contagem de contornos com base em uma imagem binarizada.
    # Isso acaba gerando a contagem de regioes das circunferencias.
    contours, hierarchy = cv2.findContours(morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    print (f"Encontrados {str(len(contours))} regiões")
    
    # Mostra as janelas
    cv2.imshow("Morphed", morph)
    cv2.imshow("Original image", imgOriginal)

    # ESC para parar a aplicacao
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Fecha todas as janelas abertas
cv2.destroyAllWindows()

