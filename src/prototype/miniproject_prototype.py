# 미니 프로젝트 - 신호등 및 신호 색상 인식하기
# YOLO 모델을 훈련시켜 웹캠을 통해 신호등의 신호를 실시간으로 인식시킨다.
# + 추가 + ROS2의 turtle을 신호등 불빛에 맞춰 전진, 감속 그리고 정지 동작을 수행하게 한다.

import cv2
import socket
import time
import torch
from ultralytics import YOLO

# 신뢰 가능한 모델이라면 allowlist 등록해 안전 경고 회피
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

# ===== 설정 =====
UBUNTU_IP = "000.000.000.000"   # <-- 본인의 Ubuntu IP로 변경
UDP_PORT  = 9999                # Ubuntu에서 열어둔 포트
CAM_INDEX = 0                   # 연결된 웹캠 인덱스

MIN_CONF  = 0.50                # YOLO confidence threshold
DEBOUNCE_N = 3                  # 같은 상태가 N프레임 연속일 때만 송신 (튀는 값 방지)
SEND_INTERVAL = 0.20            # 최소 송신 간격(초) (스팸 방지)

# YOLO 모델 로드
model = YOLO('../models/best_traffic_small_yolo.pt')

# UDP 소켓
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 상태 디바운싱용
last_state = "NONE"
stable_counter = 0
last_sent_state = "NONE"
last_sent_time = 0.0

# 클래스명 표준화 매핑(학습한 라벨명이 다를 수 있어 유연하게 처리)
def norm_label(name: str) -> str:
    s = name.lower()
    # 자주 쓰는 변형들까지 흡수
    # "green"을 "go"처럼 라벨링했거나, yellow를 amber로 적은 경우도 커버
    if "red" in s or "빨강" in s:
        return "RED"
    if "yellow" in s or "amber" in s or "노랑" in s:
        return "YELLOW"
    if "green" in s or "초록" in s:
        return "GREEN"
    if "blue" in s or "파랑" in s:   # 만약 파란불(청색)로 라벨링되어 있다면
        return "GREEN"               # Ubuntu 쪽 매핑이 GREEN 기준이므로 GREEN으로 통일
    return "UNKNOWN"

# 한 프레임에서 최종 상태 결정: 가장 높은 conf의 신호 1개만 채택
def decide_state(results) -> str:
    best = ("NONE", 0.0)
    for r in results:
        # r.boxes: 각 감지 결과
        if r.boxes is None:
            continue
        # names: 라벨 id -> 이름 매핑 (ultralytics 모델에 포함)
        names = r.names if hasattr(r, "names") else getattr(r, "names", None)
        for b in r.boxes:
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            if conf < MIN_CONF:
                continue
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            label = norm_label(cls_name)
            if label in ("RED", "YELLOW", "GREEN") and conf > best[1]:
                best = (label, conf)
    return best[0]

# 카메라 열기
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError("Camera open failed")

print("Press 'q' to quit.")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # YOLO 추론
    results = model(frame)

    # 상태 결정
    state = decide_state(results)

    # 디바운스: 동일 상태가 DEBOUNCE_N 프레임 연속일 때만 확정
    if state == last_state:
        stable_counter += 1
    else:
        stable_counter = 1
        last_state = state

    now = time.time()
    should_send = (stable_counter >= DEBOUNCE_N) and (
        state != last_sent_state or (now - last_sent_time) >= SEND_INTERVAL
    )

    if should_send:
        # UDP 전송 (Windows -> Ubuntu)
        sock.sendto(state.encode("utf-8"), (UBUNTU_IP, UDP_PORT))
        last_sent_state = state
        last_sent_time = now
        # 디버그 출력
        print(f"[SEND] {state}")

    # 시각화: YOLO 결과 그리기
    vis = frame.copy()
    for r in results:
        vis = r.plot()  # 박스/라벨 오버레이
    cv2.putText(vis, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow('frame', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
