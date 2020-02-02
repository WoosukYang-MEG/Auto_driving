import cv2  # opencv 사용
import numpy as np

class linechecker:
    def __init__(self):
        # self.cap = cv2.VideoCapture("solidWhiteRight.mp4") # 동영상 불러오기  # case 1
        self.cap = cv2.VideoCapture("challenge.mp4") # 동영상 불러오기   # case 2

        while self.cap.isOpened():
            self.ret, self.image = self.cap.read()
            self.height , self.width = self.image.shape[:2]
            self.vertices = np.array([[(50, self.height), (self.width / 2 - 45, self.height / 2 + 60),
                                       (self.width / 2 + 45, self.height / 2 + 60), (self.width - 50, self.height)]],
                                     dtype=np.int32)
            self.roi_img = self.region_of_interest(self.image, self.vertices,(0,255,0))
            self.mark = self.mark_img(self.roi_img)

            self.color_thresholds = (self.mark[:,:,0] == 0) & (self.mark[:, :, 1] > 200) & (self.mark[:, :, 2] == 0)

            self.image[self.color_thresholds] = [0,255,0]
            cv2.imshow("result",self.image)
            if cv2.waitKey(25) & 0xFF == ord("q"):        #0xFF은 이진11111111 임
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def region_of_interest(self, img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅
        mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지
        if len(img.shape) > 2: color = color3        # Color 이미지(3채널)라면
        else: color = color1                         # 흑백 이미지(1채널)라면

        # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
        cv2.fillPoly(mask, vertices, color)

        # 이미지와 color로 채워진 ROI를 합침
        ROI_image = cv2.bitwise_and(img, mask)
        return ROI_image

    def mark_img(self, img, blue_threshold=150, green_threshold=25, red_threshold=25):  # 흰색 차선 찾기
        mark = np.copy(img)  # roi_img 복사
        #  BGR 제한 값
        bgr_threshold = [blue_threshold, green_threshold, red_threshold]

        # BGR 제한 값보다 작으면 검은색으로
        thresholds = (self.image[:, :, 1] < bgr_threshold[1]) \
                     | (self.image[:, :, 2] < bgr_threshold[2])
        mark[thresholds] = [0, 0, 0]
        return mark

a = linechecker()
