import cv2
import numpy as np
from copy import deepcopy

class BoardState:
    def __init__(self, board_size) -> None:
        self.board_size = board_size
        self._stones = [0] * board_size * board_size

    def set_stone(self, index: int, val: int):
        self._stones[index] = val

    def __str__(self):
        ret_str = ""
        i = 0
        for y in range(self.board_size):
            for x in range(self.board_size):
                match self._stones[i]:
                    case -1:
                        ret_str += ' b '
                    case 1:
                        ret_str += ' w '
                    case 0:
                        ret_str += ' . '
                i += 1
            ret_str += "\n"
        return ret_str


class BoardReader:
    def __init__(self, board_size) -> None:
        self.board_size = board_size

        self.board_x, self.board_y = 0,0
        self.board_width, self.board_height = 0,0

        # self.stones = [0] * board_size * board_size
        self.state = BoardState(board_size)
        self.stored_states = []

    def get_rect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        # Threshold and morph close
        thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]
        thresh2 = cv2.adaptiveThreshold(sharpen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        close2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=2)

        cv2.imshow("close2", close2)


        # Find contours and filter using threshold area
        contours = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        min_area = 5000
        max_area = 500000000
        rects = []
        x,y,w,h = 1,1,1,1
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                rects.append((x,y,w,h))

        if rects:
            return rects[0]
        else:
            return None

    def start_capture(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error")
            return
        
        # (x,y,board_w,board_h) = (1,1,1,1)
        ret = True
        board_dimensions = None

        while ret:
            ret, frame = cap.read()
            resized = cv2.resize(frame, (600,400), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1)
            if key == 27:   # Escape
                break
            if key == 32:   # Space - recapture board position
                board_dimensions = self.get_rect(resized)
                print(board_dimensions)
                if board_dimensions:
                    (self.board_x,self.board_y, self.board_width, self.board_height) = board_dimensions
            if key & 0xFF == ord('s'):
                self.stored_states.append(deepcopy(self.state))
            if key & 0xFF == ord('r'):
                # Show all stored states
                # for s in self.stored_states:
                #     print("------------")
                #     print(s)
                # Just show the most recent
                if self.stored_states:
                    print(self.stored_states[-1])

            sw = self.board_width//(self.board_size-1)
            sh = self.board_height//(self.board_size-1)
            # average_val = cv2.mean(gray)
            blur = cv2.medianBlur(gray, 5)

            if board_dimensions: #sh > 0 and sw > 0:
                stone_index = 0
                values = []
                black_mask = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,251,50) # Black
                white_mask = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,451,-18)
                # thret, white_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
                # thret2, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
                
                cv2.imshow('Black', black_mask)
                cv2.imshow('White', white_mask)
                # cv2.imshow('Add', cv2.add(thresh, thresh2))

                for j in range(0, self.board_size):
                    for i in range(0, self.board_size):
                        xpos = self.board_x + i*sw
                        ypos = self.board_y + j*sh
                        black_value = cv2.mean(black_mask[ypos-sh//2:ypos+sh//2, xpos-sw//2:xpos+sw//2])
                        white_value = cv2.mean(white_mask[ypos-sh//2:ypos+sh//2, xpos-sw//2:xpos+sw//2])

                        # values.append(int(value[0]))
                        # if value[0] > average_val[0] + 35:
                        #     colour = (0,255, 0)
                        #     self.state.set_stone(stone_index, 1)
                        # elif value[0] < average_val[0] - 35:
                        #     colour = (255, 0 ,0)
                        #     self.state.set_stone(stone_index, -1)
                        # else:
                        #     colour = (0, 0 , 0)                            
                        #     self.state.set_stone(stone_index, 0)
                        if white_value[0] > 60:
                            colour = (0,255, 0)
                            self.state.set_stone(stone_index, 1)
                        elif black_value[0] > 100:
                            colour = (255, 0 ,0)
                            self.state.set_stone(stone_index, -1)
                        else:
                            colour = (0, 0 , 0)                            
                            self.state.set_stone(stone_index, 0)
                        stone_index += 1

                        cv2.rectangle(resized, (xpos-sw//2, ypos-sh//2), (xpos+sw//2, ypos+sh//2), colour, 2)
                # print(values)
                # print(np.digitize(values, bins=[100, 150]))
            cv2.imshow("Video", resized)

        cap.release()
        cv2.destroyAllWindows()

b = BoardReader(9)
b.start_capture()