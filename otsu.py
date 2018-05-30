import numpy as np

def total_pix(image):
    size = image.shape[0] * image.shape[1]
    return size

def histogramify(image):
    grayscale_array = []
    for w in range(0,image.size[0]):
        for h in range(0,image.size[1]):
            intensity = image.getpixel((w,h))
            grayscale_array.append(intensity)

    total_pixels = image.size[0] * image.size[1]
    bins = range(0,257)
    img_histogram = np.histogram(grayscale_array, bins)
    return total_pixels, img_histogram

# too slow
def otsu(image, threshold):
    total, hist = histogramify(image) # get total size and hist
    # total = total_pix(image) # get total size
    sumT, sum0, sum1 = 0, 0, 0
    w0, w1 = 0, 0
    varBetween, mean0, mean1 = 0, 0, 0
    for i in range(0,256):
        sumT += i * hist[0][i]
        if i < threshold:
            w0 += hist[0][i] # num of under threshold's pixel
            sum0 += i * hist[0][i]
    w1 = total - w0
    if w1 == 0:
        return 0

    sum1 = sumT - sum0
    mean0 = sum0/(w0*1.0)
    mean1 = sum1/(w1*1.0)
    varBetween = w0/(total*1.0) * w1/(total*1.0) * (mean0-mean1) * (mean0-mean1) # formulation form https://en.wikipedia.org/wiki/Otsu%27s_method
    # print "varBetween is:", varBetween
    return varBetween


def fast_ostu(image, threshold):
    image = np.transpose(np.asarray(image))
    total = total_pix(image)
    bin_image = image < threshold
    sumT = np.sum(image)
    w0 = np.sum(bin_image)
    sum0 = np.sum(bin_image * image)
    w1 = total - w0
    if w1 == 0:
        return 0
    sum1 = sumT - sum0
    mean0 = sum0 / (w0 * 1.0)
    mean1 = sum1 / (w1 * 1.0)
    varBetween = w0 / (total * 1.0) * w1 / (total * 1.0) * (mean0 - mean1) * (mean0 - mean1)  
    return varBetween

