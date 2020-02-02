import cv2
import numpy as np

class LineCheck_Edge:
    def __init__(self):
        self.image = cv2.imread("solidWhiteCurve.jpg")   # case 1
        # self.image = cv2.imread("slope_test.jpg") # case 2



        # self.image = cv2.resize(self.image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.height, self.width = self.image.shape[:2]
        self.vertices = np.array([[(50, self.height), (self.width / 2 - 45, self.height / 2 + 60),
                                   (self.width / 2 + 45, self.height / 2 + 60), (self.width - 50, self.height)]],
                                 dtype=np.int32)

        self.gray_img = self.grayscale(self.image)
        self.blur_img = self.gaussian_blur(self.gray_img, 3)
        self.canny_img = self.canny(self.blur_img, 70, 210)

        self.roi_img = self.region_of_interest(self.canny_img, self.vertices)
        self.hough_img = self.hough_lines(self.roi_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
        self.result = self.weighted_img(self.hough_img, self.image) # 원본 이미지에 검출된 이미지 overlap

        cv2.imshow("result", self.result)
        cv2.waitKey(0)

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    def canny(self, img, low_threshold, high_threshold):  # Canny 알고리즘
        return cv2.Canny(img, low_threshold, high_threshold)
    def gaussian_blur(self, img, kernel_size):  #가우시안 필터
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    def region_of_interest(self, img, vertices, color3 = (255,255,255), color1 = 255):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            color = color3
        else:
            color = color1
        cv2.fillPoly(mask, vertices, color)
        ROI_image = cv2.bitwise_and(mask, img)
        return ROI_image

    def daw_lines(self, img, lines, color=[0,255,0], thickness=2): # 선 그리기
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1,y1),(x2,y2),color,thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap): #허프 변환
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        self.daw_lines(line_img, lines)
        return line_img
    def weighted_img(self, img, initial_img, a=1, b=1., c=0.): # 두 이미지 operlap 하기
        return cv2.addWeighted(initial_img, a, img, b, c)

a = LineCheck_Edge()