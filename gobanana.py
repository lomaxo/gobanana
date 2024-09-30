import cv2
import numpy as np
from copy import deepcopy

type ImageType = np.ndarray

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

        self.state = BoardState(board_size)
        self.stored_states = []

    def process_board_image(self, image: ImageType) -> ImageType:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(grey, 3)
        sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        # Threshold and morph close
        thresh = cv2.adaptiveThreshold(sharpen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilation = cv2.dilate(thresh, kernel,iterations = 1)
        return dilation

    def get_board_rect_from_img(self, image: ImageType):
        processed_image = self.process_board_image(image)

        # Find contours and filter using threshold area
        contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # TODO: Add aspect ratio test to check for square boards not rectangles.
        rects = []
        for c in contours:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            if (0.7 < (max(w,h) / min(w,h)) < 1.3):          
                rects.append(((x,y,w,h), area))
                image = cv2.drawContours(image, [c], 0, (0,255,0), cv2.LINE_4, 8, hierarchy)
        cv2.imshow("Processed Board Image", processed_image)
        rects.sort(key = lambda x: x[1], reverse = True)
        
        if rects:
            return rects[0][0]
        else:
            return None

    def start_capture(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error starting camera")
            return
        
        ret = True
        board_dimensions = None

        while ret:
            ret, frame = cap.read()
            resized = cv2.resize(frame, (600,400), interpolation = cv2.INTER_AREA)
            grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1)
            if key == 27:   # Escape
                break
            if key == 32:   # Space - recapture board position
                board_dimensions = self.get_board_rect_from_img(resized)
                print(board_dimensions)
                if board_dimensions:
                    (self.board_x,self.board_y, self.board_width, self.board_height) = board_dimensions
            if key & 0xFF == ord('d'):  # Add the current board state to the list and display in the terminal
                self.stored_states.append(deepcopy(self.state))
                if self.stored_states:
                    print(self.stored_states[-1])

            sw = self.board_width//(self.board_size-1)
            sh = self.board_height//(self.board_size-1)
            blur = cv2.medianBlur(grey, 5)

            if board_dimensions:
                stone_index = 0
                black_mask = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,251,70)
                white_mask = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,451,-48)
                
                cv2.imshow('Black', black_mask)
                cv2.imshow('White', white_mask)

                # loop over board to check each space for black, white or no stone
                for j in range(0, self.board_size):
                    for i in range(0, self.board_size):
                        xpos, ypos = self.board_x + i*sw, self.board_y + j*sh
                        black_value = cv2.mean(black_mask[ypos-sh//2:ypos+sh//2, xpos-sw//2:xpos+sw//2])
                        white_value = cv2.mean(white_mask[ypos-sh//2:ypos+sh//2, xpos-sw//2:xpos+sw//2])

                        if white_value[0] > 60:
                            colour = (0,255, 0)
                            self.state.set_stone(stone_index, 1)
                        elif black_value[0] > 60:
                            colour = (255, 0 ,0)
                            self.state.set_stone(stone_index, -1)
                        else:
                            colour = (0, 0, 0)                            
                            self.state.set_stone(stone_index, 0)
                        stone_index += 1

                        # if colour != (0,0,0):
                        cv2.rectangle(resized, (xpos-sw//2, ypos-sh//2), (xpos+sw//2, ypos+sh//2), colour, 2)
                # Draw the board grid
                # for j in range(0, self.board_size-1):
                #     for i in range(0, self.board_size-1):
                #         xpos = self.board_x + i*sw
                #         ypos = self.board_y + j*sh
                #         cv2.rectangle(resized, (xpos, ypos), (xpos+sw, ypos+sh), (0,0,0), 2)

            cv2.imshow("Video", resized)

        cap.release()
        cv2.destroyAllWindows()

b = BoardReader(13)
b.start_capture()