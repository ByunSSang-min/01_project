# 미니 프로젝트 - 신호등 및 신호 색상 인식하기
# YOLO 모델을 훈련시켜 웹캠을 통해 신호등의 신호를 실시간으로 인식시킨다.

import cv2
from PIL import Image
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('../models/best_traffic_small_yolo.pt')

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 내에서 추론 시작
    results = model(frame)

    for r in results:
        frame = r.plot()

    # 결과 창 띄우기
    cv2.imshow('frame', frame)
    
    # 'q' 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()