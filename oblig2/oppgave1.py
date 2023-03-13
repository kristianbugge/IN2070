import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy import signal
import time
from timeit import default_timer as timer

def oppgave11(bilde):
    #middelverdi
    filter = lagfilter(15)
    bilde1 = signal.convolve2d(bilde, filter)
    plt.imsave("bilder/middelverdi15.png", bilde1, cmap="gray")

    #fourier
    M,N = bilde.shape

    fftFilter = np.fft.fft2(filter, (M,N))
    fftBilde = np.fft.fft2(bilde)
    utBilde = fftBilde * fftFilter

    bilde2 = np.real(np.fft.ifft2(utBilde))
    plt.imsave("bilder/fourier.png", bilde2, cmap="gray")

def oppgave13(bilde, filterStor):
    convP = convPlot(bilde, filterStor)
    fftP = fftPlot(bilde, filterStor)
    plot(convP, fftP)

def lagfilter(i):
    filter = np.array([[1 / (i * i)] * i] * i)
    return filter

def convPlot(bilde, filterStor):
    tid = [0]
    M,N = bilde.shape
    for i in range(1, filterStor * 2 + 1, 2):
        start = timer()
        filter = lagfilter(i)
        signal.convolve2d(bilde, filter)
        slutt = timer() - start
        tid.append(slutt)
    return tid

def fftPlot(bilde, filterStor):
    tid = [0]
    M,N = bilde.shape
    for i in range(1, filterStor * 2, 2):
        start = timer()
        filter = lagfilter(i)
        fftFilter = np.fft.fft2(filter, (M,N))
        fftBilde = np.fft.fft2(bilde)
        utBilde = fftBilde * fftFilter
        np.fft.ifft2(utBilde)
        slutt = timer() - start
        tid.append(slutt)
    return tid

def plot(convP, fftP):
    x = np.arange(len(convP))
    plt.figure()
    plt.plot(x, convP, 'r', label = 'Romlig')
    plt.plot(x, fftP, 'b', label = 'Frekvens')
    plt.xlabel('Filterstørrelse')
    plt.ylabel('Kjøretid i sekunder')
    plt.legend()
    plt.grid()
    plt.savefig('bilder/oppgave13.png')

###########################

print("Jobber...")

bilde = imread("bilder/cow.png", as_gray=True)

filterStor = 20;
oppgave11(bilde)
oppgave13(bilde, filterStor)

print("Ferdig")
