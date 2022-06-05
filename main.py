import cv2
import numpy as np
def blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_b = cv2.inRange(hsv,lower_blue,upper_blue)
    return cv2.bitwise_and(img,img,mask= mask_b)

def red_mask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_red = np.array([160,100,20])
    upper_red = np.array([179,255,255])

    mask_r = cv2.inRange(hsv,lower_red,upper_red)
    return cv2.bitwise_and(img,img,mask = mask_r)

def crop_img(img,ROI):
    height, width, channel = img.shape

    mask = np.zeros_like(img)
    math_mask = (255,) * channel
    cv2.fillPoly(mask,
                 np.array([ROI], np.int32, ),
                 math_mask)

    return cv2.bitwise_and(img, mask)

def detect_sign(image):

    edges = cv2.Canny(image,75,150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >1000 :
            approximation = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
            cv2.drawContours(image, [approximation], 0, (255, 255, 0), 1 )
            x, y, w, h = cv2.boundingRect(approximation)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
            if len(approximation) == 3:
                cv2.putText(image, "Warning sign", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
            elif len(approximation) == 4:
                propotion_ratio = float(w) / h
                if propotion_ratio >= 0.95 and propotion_ratio <= 1.05:
                    cv2.putText(image, "square sign", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
                else:
                    cv2.putText(image, "information sign", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
            else:
                cv2.putText(image, "Limit Speed sign", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))


if __name__ == "__main__":
    img =cv2.imread("70 limit and road works .jpg")
    img1 = cv2.imread("parking.jpg")
    mask_color = red_mask(img)
    mask_color_img1 = blue_mask(img1)
    # uses median beucase least reduce unnessesary edges.Gauss here detect more too much edges in mask 3X3
    blur = cv2.medianBlur(img,5)
    blur_p = cv2.medianBlur(img1,5)
    edges_p = cv2.Canny(blur_p,75,150)
    edges = cv2.Canny(blur,75,150)
    detect_sign(blur_p)
    detect_sign(blur)
    kernel = np.ones((5, 5))
    # bold edges
    dil1 = cv2.dilate(edges,kernel,iterations= 1)
    dil2 = cv2.dilate(edges_p,kernel,iterations= 1)
    cv2.imshow("limit and warning ",blur)
    cv2.imshow("parking",blur_p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


