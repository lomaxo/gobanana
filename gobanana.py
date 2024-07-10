import cv2
import numpy as np

class BoardReader:
    def get_rect(self, image):
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

        min_area = 5000
        max_area = 500000000
        rects = []
        # image_number = 0
        x,y,w,h = 1,1,1,1
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                rects.append((x,y,w,h))
                # image_number += 1
        if rects:
            return rects[0]
        else:
            return None

    def start_capture(self, board_size):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error")
            return
        
        (x,y,board_w,board_h) = (1,1,1,1)
        ret = True
        board_dimensions = None

        while ret:
            ret, frame = cap.read()
            resized = cv2.resize(frame, (600,400), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 32:
                board_dimensions = self.get_rect(resized)
                if board_dimensions:
                    (x,y,board_w,board_h) = board_dimensions

            sw = board_w//(board_size-1)
            sh = board_h//(board_size-1)
            average_val = cv2.mean(gray)
            if board_dimensions: #sh > 0 and sw > 0:
                for i in range(0, board_size):
                    for j in range(0, board_size):
                        xpos = x + i*sw
                        ypos = y + j*sh
                        # cv2.rectangle(resized, (xpos, ypos), (xpos+sw, ypos+sh), (36,255,12), 2)
                        value = cv2.mean(gray[ypos-sh//2:ypos+sh//2, xpos-sw//2:xpos+sw//2])
                        if value[0] > average_val[0] + 30:
                            colour = (0,255, 0)
                        elif value[0] < average_val[0] - 30:
                            colour = (255, 0 ,0)
                        else:
                            colour = (0, 0 ,0)
                        # colour = (int(v) for v in value)
                        # print(colour)
                        cv2.rectangle(resized, (xpos-sw//2, ypos-sh//2), (xpos+sw//2, ypos+sh//2), colour, 2)
                        # print(value)
            cv2.imshow("Video", resized)

        cap.release()
        cv2.destroyAllWindows()

b = BoardReader()
b.start_capture(9)