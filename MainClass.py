from Methodes import *
#findig the tumor
im = openImage("mri_brain.jpg")
output = sliding_window_overlap(im, (200, 90), 50)
h, bins = np.histogram(output[output > 0], bins=100)
t, vw, vb, s = otsu_threshold(h)
mask = output > bins[t]
im2 = im * mask
saveImage("Texture segmentation.jpg", im2)
im6 = otsuMethod(im2)
im3 = crop_with_argwhere(im2)
saveImage("corbed image.jpg",im3)
im4 = otsuMethod(im3)
saveImage("otsu on Texture segmentation.jpg",im6)
saveImage("segment.jpg", im4)

#calculating the size

im5 = Image.open("segment.jpg")
pix = count_pixel(im5)
estimate_size(pix)
