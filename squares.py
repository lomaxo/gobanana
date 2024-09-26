import cv2
import numpy as np

def get_rect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]
    thresh2 = cv2.adaptiveThreshold(sharpen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    close2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=2)


    # Find contours and filter using threshold area
    cnts = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # print(image.shape)

    min_area = 5000
    max_area = 500000000
    image_number = 0
    x,y,w,h = 1,1,1,1
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,12,255), 2)
            image_number += 1
    # cv2.imshow('sharpen', sharpen)
    # cv2.imshow('close', close)
    # cv2.imshow('close2', close2)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('thresh2', thresh2)
    return (x,y,w,h)

# Load image, grayscale, median blur, sharpen image
image = cv2.imread('goban_test1.jpg')
image = cv2.resize(image, (600, 400), interpolation= cv2.INTER_LINEAR)

# image = get_rect(image)

# cv2.imshow('image', image)
cv2.waitKey()
cap = cv2.VideoCapture(1)
if cap.isOpened():
        ret, frame = cap.read()
        orig_shape = frame.shape
        orig_height, orig_width = orig_shape[:2]

        while ret:
            ret, frame = cap.read()
            resized = cv2.resize(frame, (600,400), interpolation = cv2.INTER_AREA)
            
            (x,y,w,h) = get_rect(resized)
            sw = w//8
            sh = h//8
            if sh > 0 and sw > 0:
                for sx in range(x, x+w-sw, sw):
                    for sy in range(y, y+h-sh, sh):
                        cv2.rectangle(resized, (sx, sy), (sx + sw, sy + sh), (36,255,12), 2)
            cv2.imshow("Video", resized)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
else:
    print("Error starting camera")


cv2.destroyAllWindows()
