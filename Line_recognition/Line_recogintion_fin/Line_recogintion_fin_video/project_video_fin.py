import cv2
import numpy as np
import matplotlib.pyplot as plt

class LineCheck_Edge:
    def __init__(self):
        self.cap = cv2.VideoCapture("project_video.mp4")
        self.left_fit_line = None
        self.Right_fit_line = None
        while self.cap.isOpened():
            self.ret, self.image = self.cap.read()
            self.image = cv2.resize(self.image, dsize=(0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
            self.canny_img = self.canny(self.image, 160, 210) # canny 필터
            self.roi_img = self.region_of_interest(self.canny_img) # roi 필터

            self.line_arr = self.hough_lines(self.roi_img, 1, np.pi/180, 30, 20, 30) # 허프 변환


            self.slope_degrees = self.slope_degree(self.line_arr) # 기울기들 추출
            self.line_arr, self.slope_degrees = self.slope_limit(self.line_arr, self.slope_degrees, 135, 155) # 95도 160도로 제한

            self.L_lines, self.R_lines = self.line_arr[(self.slope_degrees > 0), :], self.line_arr[(self.slope_degrees < 0), :]
            self.temp = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)


            if self.L_lines.size > 0:
                self.left_fit_line = self.get_fitline(self.image, self.L_lines)
            if self.R_lines.size > 0:
                self.right_fit_line = self.get_fitline(self.image, self.R_lines)

            self.draw_fit_line(self.temp, self.left_fit_line)
            self.draw_fit_line(self.temp, self.right_fit_line)

            self.result = self.weighted_img(self.temp, self.image) # 원본 이미지에 검출된 이미지 overlap
            self.draw_roi(self.result)
            # plt.imshow(self.result)
            # plt.show()
            cv2.imshow("result", self.result)
            if cv2.waitKey(25) & 0xFF == ord("q"):  # 0xFF은 이진11111111 임
                break
        self.cap.release()
        cv2.destroyAllWindows()


    def canny(self, img, low_threshold, high_threshold):  # Canny 알고리즘
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, low_threshold, high_threshold)
        return canny

    def region_of_interest(self, img):
        polygons = np.array([
        [(130, 550),(950 ,550),(500, 270)]       # triangle의 세 꼭지점
        ])
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygons, (255,255,255))
        ROI_image = cv2.bitwise_and(mask, img)
        return ROI_image
    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap): #허프 변환

        # min_line_len = 선의 최소 길이, 너무 짧은 선분을 검출하기 싫다면 크기 업
        # max_line_gap = 선위 점들 사이의 최대 거리, 즉 이것 보다 큰 거리가 라면 동일 선분이 아니라고 간주하겠다는 말
        # output은 선분의 시작점과 끝점에 대한 좌표 값

        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap=max_line_gap)
        # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        # self.daw_lines(line_img, lines)
        return lines
    def daw_lines(self, img, lines, color=[0,255,0], thickness=5): # 선 그리기
        mask = np.copy(img)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask, (x1,y1),(x2,y2),color,thickness)
        return mask
    def draw_roi(self, img,color=[0,255,0], thickness=2): # 선 그리기
        cv2.line(img, (950, 550), (500, 270), color, thickness)
        cv2.line(img, (130, 550), (500, 270), color, thickness)
    def weighted_img(self, img, initial_img, a=0.8, b=1., c=0.): # 두 이미지 operlap 하기
        return cv2.addWeighted(initial_img, a, img, b, c)
    def slope_degree(self, line_arr):
        line_arrs = np.squeeze(line_arr)
        slope_degrees = (np.arctan2((line_arrs[:, 1] - line_arrs[:, 3]),(line_arrs[:, 0] - line_arrs[:, 2])) * 180) / np.pi  #
        return slope_degrees
    def slope_limit(self, line_arr, slope_degrees, Lowest, Highest):
        line_arr = line_arr[np.abs(slope_degrees) < Highest]
        slope_degrees = slope_degrees[np.abs(slope_degrees) < Highest]

        line_arr = line_arr[np.abs(slope_degrees) > Lowest]
        slope_degrees = slope_degrees[np.abs(slope_degrees) > Lowest]

        return line_arr, slope_degrees
    def get_fitline(self, img, f_line): #대표 선 구하기
        lines = np.squeeze(f_line)
        if lines.shape == (4,):
            lines = lines.reshape(2,2)
        else:
            lines = lines.reshape(lines.shape[0] * 2, 2)
        output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = output[0], output[1], output[2], output[3]
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
        result = [x1,y1,x2,y2]
        return result
    def draw_fit_line(self, img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
        if lines == None:
            pass
        else:
            cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


a = LineCheck_Edge()