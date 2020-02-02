import cv2
import numpy as np

class LineCheck_Edge:
    def __init__(self):
        self.image = cv2.imread("slope_test.jpg") # 이미지 입력     # case 1
        # self.image = cv2.imread("solidWhiteCurve.jpg") # 이미지 입력    # case 2


        self.height, self.width = self.image.shape[:2] # 이미지 사이즈 측정

        self.vertices = np.array([[(50, self.height), (self.width / 2 - 45, self.height / 2 + 60),
                                   (self.width / 2 + 45, self.height / 2 + 60), (self.width - 50, self.height)]],
                                 dtype=np.int32) # rest of interest를 위한 나머지 위치

        self.gray_img = self.grayscale(self.image)    # 흑백 변환
        self.blur_img = self.gaussian_blur(self.gray_img, 3) #가우시안 필터
        self.canny_img = self.canny(self.blur_img, 70, 210) # canny 필터

        self.roi_img = self.region_of_interest(self.canny_img, self.vertices) # roi 필터
        self.line_arr = self.hough_lines(self.roi_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
        self.slope_degrees = self.slope_degrees(self.line_arr) # 기울기들 추출
        self.line_arr, self.slope_degrees = self.slope_limit(self.line_arr, self.slope_degrees, 95, 160) # 95도 160도로 제한
        self.temp = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)

        self.L_lines, self.R_lines = self.line_arr[(self.slope_degrees > 0), :], self.line_arr[(self.slope_degrees < 0), :]
        self.left_fit_line = self.get_fitline(self.image, self.L_lines)
        self.right_fit_line = self.get_fitline(self.image, self.R_lines)


        self.draw_fit_line(self.temp, self.left_fit_line)
        self.draw_fit_line(self.temp, self.right_fit_line)

        self.result = self.weighted_img(self.temp, self.image) # 원본 이미지에 검출된 이미지 overlap

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

    def draw_lines(self, img, lines, color=[0,255,0], thickness=2): # 선 그리기
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1,y1),(x2,y2),color,thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap): #허프 변환

        # min_line_len = 선의 최소 길이, 너무 짧은 선분을 검출하기 싫다면 크기 업
        # max_line_gap = 선위 점들 사이의 최대 거리, 즉 이것 보다 큰 거리가 라면 동일 선분이 아니라고 간주하겠다는 말
        # output은 선분의 시작점과 끝점에 대한 좌표 값

        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap=max_line_gap)
        # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        # self.daw_lines(line_img, lines)
        return lines
    def weighted_img(self, img, initial_img, a=1, b=1., c=0.): # 두 이미지 operlap 하기
        return cv2.addWeighted(initial_img, a, img, b, c)
    def slope_degrees(self, line_arr):
        line_arr = np.squeeze(line_arr)
        slope_degrees = (np.arctan2((line_arr[:, 1] - line_arr[:, 3]),
                                        (line_arr[:, 0] - line_arr[:, 2])) * 180) / np.pi  #
        return slope_degrees
    def slope_limit(self, line_arr, slope_degrees, Lowest, Highest):
        line_arr = line_arr[np.abs(slope_degrees) < Highest]
        slope_degrees = slope_degrees[np.abs(slope_degrees) < Highest]

        line_arr = line_arr[np.abs(slope_degrees) > Lowest]
        slope_degrees = slope_degrees[np.abs(slope_degrees) > Lowest]

        return line_arr, slope_degrees
    def get_fitline(self, img, f_line): #대표 선 구하기
        lines = np.squeeze(f_line)
        lines = lines.reshape(lines.shape[0] * 2, 2)
        rows, cols = img.shape[:2]
        output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
        vx, vy, x, y = output[0], output[1], output[2], output[3]
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
        result = [x1,y1,x2,y2]
        return result
    def draw_fit_line(self, img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


a = LineCheck_Edge()