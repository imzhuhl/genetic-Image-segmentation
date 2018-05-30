from PIL import Image
import numpy as np
from genetic import GA

filename = 'images/mikaso.jpeg'

def threshold(t, image):
    image_tmp = np.asarray(image)
    intensity_array = list(np.where(image_tmp<t, 0, 255).reshape(-1))

    image.putdata(intensity_array)
    image.show()
    image.save('images/output.png')

def main():
    im = Image.open(filename)
    im.load()
    im.show()
    im_gray = im.convert('L') # translate to  gray map

    ga = GA(im_gray)
    for x in xrange(50):
        ga.evolve()
    best_threshold = ga.result()
    print best_threshold

    threshold(best_threshold, im_gray)

if __name__ == "__main__":
    main()

