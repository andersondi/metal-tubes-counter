import cv2

im = cv2.imread('resources/tubes1.jpg')

# imgGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# imgGray = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_DEFAULT)
# im = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)

# converte imagem para escala HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# binarizar a imagem utilizando o arquivo HSV
th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# th, bw = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY)

cv2.imshow('binary', bw)
# elemento estrutural necessario como parametro para operações de mudança morfologica
# neste caso, foi usado uma chamada para um elemento eliptico pois buscamos formas aproximadas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# operação o fechamento de regioes desnecessarias seguidas de uma dilatacao 
# para gerar regioes com tamanho semelhante ao tamanho original
morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
cv2.imshow('dist', dist)

borderSize = 75

distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

cv2.imshow('distborder', distborder)

gap = 10                                
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
mn, mx, _, _ = cv2.minMaxLoc(nxcor)
th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
peaks8u = cv2.convertScaleAbs(peaks)
contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
    cv2.circle(im, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.drawContours(im, contours, i, (0, 0, 255), 2)

cv2.imshow('circles', im)
cv2.waitKey(0)
cv2.destroyAllWindows()