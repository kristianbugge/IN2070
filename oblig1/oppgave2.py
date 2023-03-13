import numpy as np
import math
from imageio import imread, imwrite
import matplotlib.pyplot as plt

def konvulsjon(bilde, filter):
    M1,N1 = bilde.shape
    M2,N2 = filter.shape

    filter = np.rot90(filter, 2) #roterer 180 grader
    utBilde = np.zeros((M1, N1))
    a = int((M2 - 1) / 2)
    b = int((N2 - 1) / 2)

    utvidetBilde = np.pad(bilde, ((a, a), (b, b)), mode = 'edge')

    for i in range(M1):
        for j in range(N1):
            pVerdi = utvidetBilde[i:(i + 2 * a + 1),j:(j + 2 * b + 1)] * filter
            utBilde[i, j] = np.sum(pVerdi)
    return utBilde

def lagGaussfilter(sigma):
    n = np.round(1 + 8 * sigma)
    filter = np.zeros((n, n))
    sum = 0

    a = int((n - 1) / 2)
    for i in range(-a, a + 1):
        for j in range(-a, a + 1):
            k = np.exp(-((i**2 + j**2) / (2*sigma**2)))
            filter[i + a][j + a] = k
            sum += k
    return filter/sum

def gradientmagnitud(bilde):
    x = np.array([[0,1,0], [0,0,0], [0,-1,0]])
    y = np.array([[0,0,0], [1,0,-1], [0,0,0]])

    gx = konvulsjon(bilde, x)
    gy = konvulsjon(bilde, y)

    A = np.sqrt(gx**2 + gy**2)
    th = np.rad2deg(np.arctan2(gy,gx))

    max = np.max(A)
    min = np.min(A)

    utBilde = (255 - 0) / (max - min) * (A - min)

    return utBilde,th

def kanttynning(gradientM, gradientV):
    utM = np.copy(gradientM)
    M1, N1 = gradientV.shape
    padM = np.pad(utM, 1, 'constant')

    for i in range(M1):
        for j in range(N1):
            g = i + 1
            h = j + 1
            vinkel = gradientV[i, j]

            if (np.abs(vinkel) <= 22.5 or np.abs(vinkel) > 157.5):
                if((padM[g + 1, h] > padM[g, h]) or (padM[g - 1, h] > padM[g, h])):
                    utM[i, j] = 0

            elif ((22.5 < vinkel <= 67.5) or (-112.5 > vinkel >= -157.5)):
                if((padM[g + 1, h + 1] > padM[g, h]) or (padM[g - 1, h - 1] > padM[g, h])):
                    utM[i, j] = 0

            elif (67.5 < np.abs(vinkel) <= 112.5):
                if((padM[g, h + 1] > padM[g, h]) or (padM[g, h - 1] > padM[g, h])):
                    utM[i, j] = 0

            elif ((112.5 < vinkel <= 157.5) or (-22.5 > vinkel >= -67.5)):
                if((padM[g - 1, h + 1] > padM[g, h]) or (padM[g + 1, h - 1] > padM[g, h])):
                    utM[i, j] = 0
    return utM

def hysterese(tynnetBilde, lav, hoy):
    tynnetBildeH = np.where(tynnetBilde >= hoy, tynnetBilde, 0)
    tynnetBildeL = np.where((lav <= tynnetBilde) & (tynnetBilde < hoy), tynnetBilde, 0)

    merket = np.nonzero(tynnetBildeH)
    M,N = tynnetBilde.shape
    while(len(merket[1]) > 0):
        nyeMerket = [[],[]]
        for i,j in zip(merket[0], merket[1]):
            for g in range(-1, 2):
                for h in range(-1, 2):
                    x,y = i + g, j + h
                    if (0 <= x < M) & (0 <= y < N):
                        if tynnetBildeL[x, y] > 0:
                            tynnetBildeH[x, y] = tynnetBildeL[x, y]
                            tynnetBildeL[x, y] = 0
                            nyeMerket[0].append(x)
                            nyeMerket[1].append(y)
        merket = nyeMerket
    return tynnetBildeH

def main():
    celleBilde = imread('cellekjerner.png', as_gray = True)

    sigma = 5
    Tl = 50
    Th = 70
    print("prosseserer bildet...")
    gaussfilter = lagGaussfilter(sigma)
    filtrertBilde = konvulsjon(celleBilde, gaussfilter)
    magnitud, vinkelMagnitud = gradientmagnitud(filtrertBilde)
    tynnetM = kanttynning(magnitud, vinkelMagnitud)
    hystereseFerdig = hysterese(tynnetM, Tl, Th)

    print("Ferdig")
    filtrert = np.round(filtrertBilde).astype(np.uint8)
    kanttynnet = np.round(tynnetM).astype(np.uint8)
    magnitudBilde = np.round(magnitud).astype(np.uint8)
    hystereseFerdig = np.round(hystereseFerdig).astype(np.uint8)

    imwrite("magnitud.png", magnitudBilde)
    imwrite("filtrert.png", filtrert)
    imwrite("kanttynnet.png", kanttynnet)
    imwrite("hysterese.png", hystereseFerdig)

if __name__ == "__main__":
    main()
