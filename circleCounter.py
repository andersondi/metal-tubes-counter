import cv2
import numpy as np

def nothing():
    pass

font = cv2.FONT_HERSHEY_SIMPLEX

imgOriginal = cv2.imread('resources/tubes4.jpg')

# elemento estrutural necessario como parametro para operações de mudança morfologica
kernel = np.ones((3, 3), np.uint8)

# Cria uma janela
cv2.namedWindow('Morphed')

# Barra de selecao para o limiar
cv2.createTrackbar('Treshold','Morphed',0,255,nothing)

# Barra de selecao do numero de iteracoes de dilatacao e erosao
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
    
    dist = cv2.distanceTransform(morph,cv2.DIST_L2, cv2.DIST_MASK_3)

    cv2.normalize(dist,dist,0,1,cv2.NORM_MINMAX)

    # Blur na imagem binaria para criar um gradiente de cinzas nas bordas das regioes
    bilateral_filtered_image = cv2.bilateralFilter(morph, 5, 175, 175)
    
    # Faz a contagem de contornos com base em uma imagem binarizada.
    # Isso acaba gerando a contagem de regioes das circunferencias.
    contours, hierarchy = cv2.findContours(morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
    number_of_areas = 0
    accepted_error = 0.7
    
    for cnt in contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        area = cv2.contourArea(cnt)
        area_circle=3.14*radius*radius
        
        if area_circle:
            valid_range = (area / area_circle >= (1 - accepted_error)) and (area / area_circle <= (1 + accepted_error))
            if valid_range:
                number_of_areas += 1
            
    # Deteccao das circunferencias usando Hough Circles
    circles = cv2.HoughCircles(
                                bilateral_filtered_image,
                                cv2.HOUGH_GRADIENT,
                                1,
                                20,
                                param1=50,
                                param2=30,
                                minRadius=0,
                                maxRadius=0
                                )
    
    number_of_circles = 0
    
    if circles is not None:
        for i in circles[0,:]:
            # Desenha a circunferencia
            cv2.circle(clone,(i[0],i[1]),i[2],(0,255,0),2)
            # Desenha o centro da circunferencia
            cv2.circle(clone,(i[0],i[1]),2,(0,0,255),3)
            number_of_circles += 1

    resultText = f"Regioes: {str(len(contours))}"
    circlesResultText = f"Circulos: {number_of_circles}"
    areaResultText = f"Areas validas: {number_of_areas}"
    
    borderSize = 100

    clone = cv2.copyMakeBorder(clone, borderSize, 0, 0, 0, 
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    
    for i in range(100):
        cv2.putText(clone, resultText, (10, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(clone, circlesResultText, (200, 30), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(clone, areaResultText, (10, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostra as janelas
    cv2.imshow("Morphed", morph)
    cv2.imshow("Original image", clone)
    
    # ESC para parar a aplicacao
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Fecha todas as janelas abertas
cv2.destroyAllWindows()

