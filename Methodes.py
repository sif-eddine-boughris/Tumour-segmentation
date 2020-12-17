import numpy as np
from PIL import Image

from skimage.feature import greycomatrix, greycoprops



def imageSize(image):  # méthode qui calcule la taille de l'image
    size = image.shape[0] * image.shape[1]
    return size


def openImage(image):  # méthode pour ouvrir une image
    imageRGB = Image.open(image).convert('L')
    imageArray = np.array(imageRGB)
    print("image opend")
    return imageArray


def openGrayScaled(image):  # méthode pour ouvrir une image et la convertir en NDG
    imageNDG = Image.open(image).convert('L')
    return imageNDG


def convertToArray(image):  # méthode pour convertir une image en matrice
    im = np.array(image)
    return im


def showImage(name, image):  # méthode pour afficher une image
    IMG = Image.fromarray(image)
    # IMG.save("résultats.jpg")
    IMG.show()


def calcHistogram(image):  # méthode qui calcule l'histogramme d'une image
    histogram = []
    for i in range(0, 256):
        histogram.append(0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] = histogram[image[i, j]] + 1
    print("L'histogramme : ", histogram)
    return histogram


def calcWithinClassVariance(threshold, hist,
                            Size):  # méthode qui reçoit comme paramètres un seuil, l'histogramme d'une image et la taille et retourne le within Class Variance
    S1 = 0
    S2 = 0
    S3 = 0
    if (threshold == 0):
        weightBackground = 0
        varienceBackground = 0
    else:
        for i in range(0, threshold):
            S1 = S1 + hist[i]
            S2 = S2 + hist[i] * i
        weightBackground = S1 / Size
        if (S1 == 0):  # cas de division par 0
            varienceBackground = 0
        else:
            MeanBackground = S2 / S1
            for i in range(0, threshold):
                S3 = S3 + ((i - MeanBackground) * (i - MeanBackground) * hist[i])

            varienceBackground = S3 / S1

    S1 = 0
    S2 = 0
    S3 = 0
    for i in range(threshold, 256):
        S1 = S1 + hist[i]
        S2 = S2 + hist[i] * i

    weightForeground = S1 / Size
    if (S1 == 0):  # cas de division par 0
        varienceForeground = 0
    else:
        MeanForeground = S2 / S1

        for i in range(threshold, 256):
            S3 = S3 + ((i - MeanForeground) * (i - MeanForeground) * hist[i])
        varienceForeground = S3 / S1
    withinClassVariance = weightBackground * varienceBackground + weightForeground * varienceForeground

    return withinClassVariance


def variancesTable(hist,
                   size):  # méthode qui calcule les within Class Variances pour chaque seuil de 0 à 255, et retourne une liste de 256 variances
    variances = []
    for i in range(0, 256):
        variances.append(0)

    for i in range(0, 256):
        variances[i] = calcWithinClassVariance(i, hist, size)

    return variances


def varianceMin(
        varianceTable):  # méthode qui calcule la valeur minimale de la liste des variances, et retourne le seuil correspondent
    min = varianceTable[0]
    t = 0
    for i in range(1, 256):
        if (min > varianceTable[i]):
            min = varianceTable[i]
            t = i
    return t


def convertToBinary(t, img):  # méthode pour convertir une image en binaire avec un seuil t
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j] >= t):
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img


def saveImage(name, image):
    Image.fromarray(image).save(name)


def otsuMethod(imageName):
    image1 = imageName
    image = convertToArray(image1)
    size = imageSize(image)
    histogram = calcHistogram(image)
    variances = variancesTable(histogram, size)
    threshold = varianceMin(variances)
    print("le seuil = ", threshold)
    img = convertToBinary(threshold, image)
    return img


def otsu_threshold(h):
    p = h * 1. / h.sum()


    best_l = 0
    best_t = 0

    var_within = np.zeros(len(h))
    var_between = np.zeros(len(h))
    sep = np.zeros(len(h))
    for t in range(1, len(h) - 1):
        w0 = p[:t].sum()
        w1 = p[t:].sum()

        m0 = (np.arange(0, t) * p[:t]).sum() / w0
        m1 = (np.arange(t, len(h)) * p[t:]).sum() / w1

        s0 = (((np.arange(0, t) - m0) ** 2) * p[:t]).sum() / w0
        s1 = (((np.arange(t, len(h)) - m1) ** 2) * p[t:]).sum() / w1

        sw = w0 * s0 + w1 * s1

        sb = w0 * w1 * ((m1 - m0) ** 2)

        l = sb / sw
        if (l > best_l):
            best_l = l
            best_t = t
        var_within[t] = sw
        var_between[t] = sb
        sep[t] = l

    return best_t, var_within, var_between, sep


def texture_descriptor(N):
    displacement = 20
    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3]
    glcm = greycomatrix(N, [displacement], angles, 256)
    return greycoprops(glcm, 'dissimilarity').max()


def sliding_window_overlap(im, PATCH_SIZE, STRIDE):
    output = np.zeros((im.shape[0], im.shape[1]))
    for i in range(0, im.shape[0] - PATCH_SIZE[0] + 1, STRIDE):
        for j in range(0, im.shape[1] - PATCH_SIZE[1] + 1, STRIDE):
            patch = im[i:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]]
            c = (i + PATCH_SIZE[0] // 2, j + PATCH_SIZE[1] // 2)  # center of the patch
            output[c[0] - STRIDE:c[0] + STRIDE, c[1] - STRIDE:c[1] + STRIDE] = texture_descriptor(patch)
    return output


def crop_with_argwhere(image):
    mask = image > 0
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    cropped = image[x0:x1, y0:y1]
    return cropped

def count_pixel(image):
    white = 0
    other = 0



    for pixel in image.getdata():
        if pixel == 1:  # if your image is RGB (if RGBA, (0, 0,     0, 255) or so
            white += 1
        else:
            other += 1
    print('white=' + str(white) + ', Other=' + str(other))
    return white

def estimate_size(pix):
    cm = pix*0.115
    print('the tumer size = ',cm,'Cm')
    return cm

