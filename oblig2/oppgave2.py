from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

def oppgave2(bilde, q):
    origEntro = beregnEntropi(bilde)
    bilde -= 128
    Q = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])
    qQ = np.multiply(q, Q)

    #DCT
    DCT = DCtransform(bilde, qQ)

    entro = beregnEntropi(DCT)
    kompresjonsrate = origEntro / entro
    lagringsplass = 100 * (1 - entro / origEntro)

    print('Statistikk for kompresjon med q = ' + str(q) + ': ' + 'Kompresjonsrate = ' + str(kompresjonsrate)+ ', Lagringsplass spart= ' + str(lagringsplass) + '%')

    IDCT = IDCtransform(DCT, qQ)
    IDCT += 128

    return IDCT

def DCtransform(bilde, qQ):
    M, N = bilde.shape
    DCTut = np.zeros((M, N))
#8x8 blokker først
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            #Så bruke forenklet formel fra oppgavetekst
            for u in range(8):
                for v in range(8):
                    tot = 0
                    for x in range(8):
                        for y in range(8):
                            tot += bilde[x + i, y + j] * np.cos(((2 * x + 1) * u * np.pi) / 16) * np.cos(((2 * y + 1) * v * np.pi) / 16)
                    DCTut[u + i, v + j] = np.round((1 / 4 * c(u) * c(v) * tot) / qQ[u, v])
    return DCTut

def IDCtransform(bilde, qQ):
    M, N = bilde.shape
    DCTinn = np.zeros((M, N))
    IDCTut = np.zeros((M, N))
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            for x in range(8):
                for y in range(8):
                    tot = 0
                    for u in range(8):
                        for v in range(8):
                            DCTinn[i + u, j + v] = bilde[i + u, j + v] * qQ[u, v]
                            tot += c(u) * c(v) * DCTinn[u + i, v + j] * np.cos(((2 * x + 1) * u * np.pi) / 16) * np.cos(((2 * y + 1) * v * np.pi) / 16)
                    IDCTut[i + x, j + y] = np.round((1 / 4 * tot), 0)
    return IDCTut
#fra oppgaveteksten
def c(a):
    if a == 0:
        return 1/math.sqrt(2)
    else:
        return 1

def histogram(bilde):
    M,N = bilde.shape
    G = 500
    hist = [0] * G

    for i in range(M):
        for j in range(N):
            x = int(bilde[i, j] + 128)
            if(x in range(0, G - 1)):
                hist[x] = hist[x]+1
    return hist

def normalisertHistogram(bilde):
    hist = histogram(bilde)
    tot = sum(hist)
    normHist = hist.copy()
    for i in range(len(normHist)):
        normHist[i] = hist[i] / tot
    return normHist

def beregnEntropi(bilde):
    histo = normalisertHistogram(bilde)
    tot1 = 0
    for i in range(len(histo)):
        if(histo[i] != 0):
            tot1 += histo[i] * np.log2(1 / histo[i])
    return -tot1

##############################
print("Jobber...")
#q-verdiene vi skal komprimere med
qer = [0.1, 0.5, 2, 8, 32]

for q in qer:
    bilde = imread('bilder/uio.png', as_gray = True)
    resultat = oppgave2(bilde, q)
    qstring = str(q)
    plt.imsave('bilder/qverdi_' + qstring + '.png', resultat, cmap="gray")

print("Ferdig")
