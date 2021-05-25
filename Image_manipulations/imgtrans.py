import cv2
from matplotlib import pyplot as plt
img = cv2.imread('dog1.png')
img =cv2.resize(img,(600,600))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale image', gray_img)
#cv2.waitKey()
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
"""
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey()
"""
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV image', hsv_img)
imh1 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
imh2= cv2.applyColorMap(img, cv2.COLORMAP_HSV)
imh3 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
imh4 = cv2.applyColorMap(img, 2)
imh5 = cv2.applyColorMap(img, 21)

#cv2.imshow('colormap', imh5)
"""
cv2.imshow('H channel', hsv_img[:, :, 0])
cv2.imshow('S channel', hsv_img[:, :, 1])
cv2.imshow('V channel', hsv_img[:, :, 2])
cv2.waitKey()
"""

dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05) # inbuilt function to generate pencil sketch in both color and grayscale
# sigma_s controls the size of the neighborhood. Range 1 - 200
# sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1
# shade_factor is a simple scaling of the output image intensity. The higher the value, the brighter is the result. Range 0 - 0.1
#cv2.imshow("Image", img)
"""
#dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
#cv2.imshow("Output2", dst_gray)
cv2.imshow("Output", dst_color)
cv2.imwrite("sketch.jpg",dst_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.axis('off')
dst_color = cv2.imread('sketch.jpg')
"""
#converting an image to grayscale
grayScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ReSized2 = cv2.resize(grayScaleImage, (960, 540))
#plt.imshow(ReSized2, cmap='gray')
#applying median blur to smoothen an image
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
#ReSized3 = cv2.resize(smoothGrayScale, (960, 540))
#plt.imshow(ReSized3, cmap='gray')
#retrieving the edges for cartoon effect
#by using thresholding technique
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 200, 
  cv2.ADAPTIVE_THRESH_MEAN_C, 
  cv2.THRESH_BINARY, 9, 9)
#ReSized4 = cv2.resize(getEdge, (960, 540))
#plt.imshow(ReSized4, cmap='gray')
#applying bilateral filter to remove noise 
#and keep edge sharp as required
bilat = cv2.bilateralFilter(img, 105, 100, 100)
#ReSized5 = cv2.resize(colorImage, (960, 540))
#plt.imshow(ReSized5, cmap='gray')
#masking edged image with our "BEAUTIFY" image
cartoonImage = cv2.bitwise_and(bilat, bilat, mask=getEdge)
#ReSized6 = cv2.resize(cartoonImage, (960, 540))

plt.subplot(5, 3, 1), plt.imshow(hsv_img, 'gray'),plt.axis('off')
plt.subplot(5, 3, 2), plt.imshow(imh1, 'cool'),plt.axis('off')
plt.subplot(5, 3, 3), plt.imshow(imh2, 'hsv'),plt.axis('off')
plt.subplot(5, 3, 4), plt.imshow(imh3, 'spring'),plt.axis('off')
plt.subplot(5, 3, 5), plt.imshow(imh4, 'plasma'),plt.axis('off')
plt.subplot(5, 3, 6), plt.imshow(yuv_img, 'gist_heat'),plt.axis('off')
plt.subplot(5, 3, 7), plt.imshow(dst_color, 'gist_heat'),plt.axis('off')
plt.subplot(5, 3, 8), plt.imshow(getEdge, 'gray'),plt.axis('off')
plt.subplot(5, 3, 9), plt.imshow(bilat, 'gray'),plt.axis('off')
plt.subplot(5, 3, 10), plt.imshow(cartoonImage, 'gray'),plt.axis('off')
plt.axis('off')
plt.show()




# loading library
import cv2
import numpy as np
img = cv2.imread('example6.png')
#img = cv2.imread('dog1.png')
# Specify the kernel size.
# The greater the size, the more the motion.
kernel_size = 30
# Create the vertical kernel.
kernel_v = np.zeros((kernel_size, kernel_size))
# Create a copy of the same for creating the horizontal kernel.
kernel_h = np.copy(kernel_v)
# Fill the middle row with ones.
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
# Normalize.
kernel_v /= kernel_size
kernel_h /= kernel_size
# Apply the vertical kernel.
vertical_mb = cv2.filter2D(img, -1, kernel_v)
# Apply the horizontal kernel.
horizonal_mb = cv2.filter2D(img, -1, kernel_h)
comb = vertical_mb+horizonal_mb
# Save the outputs.
cv2.imwrite('example6_horzout.jpg', horizonal_mb)
#cv2.imwrite('example6_verout.jpg', horizonal_mb)
#cv2.imwrite('example6_comb_out.jpg', comb)
imagem = cv2.bitwise_not(imh4)
#cv2.imwrite('invert.jpg', imagem)


from PIL import Image, ImageDraw
# read the image
image = Image.open('dog1.png')
red, green, blue = image.split()
new_image = Image.merge("RGB", (green, red, blue))
#new_image.save('new_image.jpg')

img = cv2.imread('dog1.png')
#img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.bilateralFilter(img,9,50,50)
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened = cv2.filter2D(img, -1, kernel_sharpening)
brightness = -70
contrast = -10
img = np.int16(img)
img = sharpened * (contrast/127+1) - contrast + brightness
img = np.clip(img, 0, 255)
img = np.uint8(img)
cv2.imwrite('Sharpened.png', img)




import cv2

image = cv2.imread('example5.png')
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

ROI_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image[y:y+h, x:x+w]
    cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)
    ROI_number += 1


#cv2.waitKey()


# import Pillow modules
from PIL import Image
from PIL import ImageFilter
import numpy as np
# Load the image
img = Image.open("example5.png");
# Display the original image
#img.show()
# Read pixels and apply negative transformation

for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
        # Get pixel value at (x,y) position of the image
        pixelColorVals = img.getpixel((i,j));
        #print(pixelColorVals[0])
        # Invert color
        redPixel    = 200 - pixelColorVals[0]; # Negate red pixel
        greenPixel  = 200 - pixelColorVals[1]; # Negate green pixel
        bluePixel   = 200 - pixelColorVals[2]; # Negate blue pixel
        # Modify the image with the inverted pixel values
        img.putpixel((i,j),(redPixel, greenPixel, bluePixel));


# Display the negative image
print(np.max(img))
img=img
img.show();
import cv2
import numpy as np
img= cv2.imread('example5.png')

# Load image
red   = [0,0,255]
green = [0,255,0]
blue  = [255,0,0]
white = [255,255,255]
black = [0,0,0]
gray =[47,79,79]

# Make all perfectly green pixels white
#img[np.all(img == list((255, 255, 255)), axis=-1)] = list((150,150,150))
img[np.all(img == white, axis=-1)] = gray

# Save result
cv2.imwrite('result1.png',img)



import cv2

def funcBrightContrast(bright=0):
    bright = cv2.getTrackbarPos('bright', 'Life2Coding')
    contrast = cv2.getTrackbarPos('contrast', 'Life2Coding')

    effect = apply_brightness_contrast(img,bright,contrast)
    cv2.imshow('Effect', effect)

def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

if __name__ == '__main__':

    original = cv2.imread("example5.png", 1)
    img = original.copy()

    cv2.namedWindow('Life2Coding',1)

    bright = 255
    contrast = 127

    #Brightness value range -255 to 255
    #Contrast value range -127 to 127

    cv2.createTrackbar('bright', 'Life2Coding', bright, 2*255, funcBrightContrast)
    cv2.createTrackbar('contrast', 'Life2Coding', contrast, 2*127, funcBrightContrast)
    funcBrightContrast(0)
    cv2.imshow('Life2Coding', original)


cv2.waitKey(0)
