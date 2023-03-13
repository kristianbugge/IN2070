import numpy as np
import math
from imageio import imread, imwrite
import matplotlib.pyplot as plt

def histogram(bilde):
    hist = np.zeros(256)
    M,N = bilde.shape

    for i in range(M):
        for j in range(N):
            hist[int(bilde[i][j])] += 1

    return hist

def normalisertHistogram(bilde):
    M,N = bilde.shape

    hist = histogram(bilde)

    normalHist = hist / (N*M)

    return normalHist

def middelverdi(hist):
    mv = 0

    for i in range(len(hist)):
        mv += i * hist[i]

    return mv

#standardavvik
def sd(hist):
    n = len(hist)

    mv = middelverdi(hist)

    varians = sum((x - mv) ** 2 for x in hist) / n

    return math.sqrt(varians)

#Sette standardavvik og middelverdi til ønsket verdi
#Returne utbildet
def setSdMv(nySd, nyMv, bilde):
    M,N = bilde.shape
    utBilde = np.zeros((M,N))
    hist = normalisertHistogram(bilde)

    a = nySd / sd(hist)
    b = nyMv - (a*middelverdi(hist))

    for i in range(M):
        for j in range(N):
            utBilde[i][j] = a * bilde[i][j] + b

    return utBilde

def bilinearInterpolasjon(x, y, bilde):
    x0 = int(np.floor(x))
    x1 = int(np.ceil(x))
    y0 = int(np.floor(y))
    y1 = int(np.ceil(y))
    maxX = np.shape(bilde)[0]
    maxY = np.shape(bilde)[1]

    if x0 >= 0 and x1 < maxX and y0 >= 0 and y1 < maxY:
        l1 = bilde[x0][y0] + ((bilde[x1][y0] - bilde[x0][y0])) * (x - x0)
        l2 = bilde[x0][y1] + ((bilde[x1][y1] - bilde[x0][y0])) * (x - x0)
        return l1 + (l2 - l1) * (y - y0)
    else: return

def naboInterpolasjon(x, y, bilde):
    x1 = int(np.round(x))
    y1 = int(np.round(y))
    maxX, maxY = bilde.shape

    if x < maxX and y < maxY:
        return bilde[x1][y1]
    else: return

def forlengs(bilde, utBilde, transform):
    M1,N1 = bilde.shape
    M2,N2 = utBilde.shape
    utBilde = np.zeros((M2,N2))

    for x in range(M1):
        for y in range(N1):
            xy = np.dot(transform, [x, y, 1])
            if int(np.round(xy[0])) < M2 and int(np.round(xy[1])) < N2:
                utBilde[int(np.round(xy[0]))][int(np.round(xy[1]))] = bilde[x][y]
    return utBilde

def bilinearBaklengs(bilde, bildeShape, transform):
    M1,N1 = bilde.shape
    M2,N2 = bildeShape.shape
    utBilde = np.zeros((M2,N2))
    inv = np.linalg.inv(transform)

    for x in range(M2):
        for y in range(N2):
            xyUt = np.dot(inv, [x, y, 1])
            if xyUt[0] < M1 and xyUt[1] < N1:
                utBilde[x][y] = bilinearInterpolasjon(xyUt[0], xyUt[1], bilde)

    return utBilde


def naboBaklengs(bilde, bildeShape, transform):
    M1,N1 = bilde.shape
    M2,N2 = bildeShape.shape
    utBilde = np.zeros((M2,N2))
    inv = np.linalg.inv(transform)

    for x in range(M2):
        for y in range(N2):
            xyUt = np.dot(inv, [x, y, 1])
            if xyUt[0] < M1 and xyUt[1] < N1:
                utBilde[x][y] = naboInterpolasjon(xyUt[0], xyUt[1], bilde)

    return utBilde

def main():
    bildePortrett = imread("portrett.png", as_gray = True)
    bildeMaske = imread("geometrimaske.png", as_gray = True)

    #punktene fra portrett og maske
    x = np.array([[88,68,109],[84,120,129],[1,1,1]])
    y = np.array([[260,260,443],[170,341,257],[1,1,1]])

    tMatrise = np.dot(y,np.linalg.inv(x))

    normHist = normalisertHistogram(bildePortrett)

    nyttBilde = setSdMv(64, 127, bildePortrett)
    nyttHist = normalisertHistogram(nyttBilde)

    forlengsBilde = forlengs(nyttBilde, bildeMaske, tMatrise)
    naboBaklengsBilde = naboBaklengs(nyttBilde, bildeMaske, tMatrise)
    bilinearBaklengsBilde = bilinearBaklengs(nyttBilde, bildeMaske, tMatrise)


    nyttBilde = (np.round(nyttBilde)).astype(np.uint8)
    forlengsBilde = (np.round(forlengsBilde)).astype(np.uint8)
    naboBaklengsBilde = (np.round(naboBaklengsBilde)).astype(np.uint8)
    bilinearBaklengsBilde = (np.round(bilinearBaklengsBilde)).astype(np.uint8)

    imwrite("graaskala.png", nyttBilde)
    imwrite("forlengsBilde.png", forlengsBilde)
    imwrite("naboBaklengsBilde.png", naboBaklengsBilde)
    imwrite("bilinearBaklengsBilde.png", bilinearBaklengsBilde)

if __name__ == "__main__":
    main()
