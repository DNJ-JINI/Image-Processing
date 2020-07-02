from PIL import Image
import pytesseract
import cv2
import numpy as np
import os
pytesseract.pytesseract.tesseract_cmd = (r"./Tesseract-OCR/tesseract")

def remove_lines (filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, None, fx=5,fy=5, interpolation=cv2.INTER_CUBIC)
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    
    thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    

    thresh = thresh

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)   
##    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)
        # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)
    cv2.imwrite('result.png', result)
    cv2.waitKey()
    
fn="./pdf_train_single"
##fn ="test_oldmodel"
for filename in os.listdir(fn):
    
    filename =os.path.join(fn,filename)
    remove_lines (filename)
    kernel = np.ones((3, 3), np.uint8)
    img= cv2.imread("result.png")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##    img = cv2.bilateralFilter(img, 9, 55, 55, cv2.BORDER_DEFAULT)
    
##    cv2.imshow("img",img)
    print("................."+filename+ "..............")    
    print(pytesseract.image_to_string(img, lang='eng',config =' --psm 4 --oem 2'))



