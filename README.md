# SkyScout

SkyScout는 드론의 온보드 컴퓨터에서 동작하는 실시간 영상 기반 물체 탐지 프로젝트입니다. 비행 중 카메라로 들어오는 영상에서 사람, 차량 및 기타 물체를 탐지하고, 탐지된 물체를 영상에 붉은 박스로 표시합니다. 처리된 영상은 외부로 스트리밍하거나 이미지 형태로 전송합니다.

## 동작 환경

- 실행 위치: 드론의 온보드 컴퓨터
- 영상 입력: 온보드 컴퓨터에 USB 또는 Ethernet으로 연결된 카메라
- 처리 대상: 드론 비행 중 카메라에서 들어오는 실시간 영상

## 주요 기능

1. 카메라에서 실시간 영상 이미지를 입력받습니다.
2. 영상에서 사람, 차량 및 기타 물체를 탐지합니다.
3. 탐지된 물체의 위치를 붉은 박스로 영상에 표시합니다.
4. 표시가 적용된 영상을 외부로 스트리밍하거나 이미지 형태로 전송합니다.

## 처리 흐름

`카메라 입력 → 물체 탐지 → 붉은 박스 표시 → 외부 전송`

## 권장 기술 스택

아래는 Ubuntu 기반 온보드 컴퓨터에서 개발하기 위한 제안입니다. 실제 패키지 버전과 가속 방식은 컴퓨터의 CPU/GPU 및 카메라 사양을 확인한 뒤 확정합니다.

| 영역 | 기술 | 용도 |
| --- | --- | --- |
| 운영체제 및 언어 | Ubuntu LTS, Python 3 | 온보드 애플리케이션 실행 |
| USB 카메라 입력 | [V4L2](https://docs.kernel.org/userspace-api/media/v4l/v4l2.html), [OpenCV VideoCapture](https://docs.opencv.org/4.13.0/d0/da7/videoio_overview.html) | USB 카메라의 실시간 프레임 수집 |
| Ethernet 카메라 입력 | [GStreamer](https://gstreamer.freedesktop.org/documentation/), OpenCV | 카메라가 제공하는 영상 프로토콜(예: RTSP)에 맞춰 프레임 수집 |
| 물체 탐지 | 경량 객체 탐지 모델(ONNX 형식), [ONNX Runtime](https://onnxruntime.ai/docs/api/python/api_summary.html) | 사람·차량 등 모델이 지원하는 종류의 물체 탐지. CPU로 시작하고, 지원 GPU가 있으면 해당 실행 제공자 사용 |
| 영상 표시 | [OpenCV](https://docs.opencv.org/4.13.0/d6/d6e/group__imgproc__draw.html) | 탐지 좌표에 붉은 경계 상자 그리기 및 이미지 인코딩 |
| 영상 스트리밍 | [GStreamer RTSP Server](https://gstreamer.freedesktop.org/documentation/additional/rtp.html), H.264 | 처리된 영상을 외부 수신자에게 RTSP로 전송 |
| 이미지 전송 | [FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/) | 처리된 최신 프레임을 JPEG 이미지로 제공 |
| 서비스 운영 | systemd | 부팅 시 자동 실행 및 프로세스 재시작 |

탐지 가능한 물체 종류는 선택한 모델의 학습 클래스에 따라 달라집니다. 카메라 연결 방식, 추론 속도, 영상 지연 및 전송 대역폭은 실제 온보드 장비에서 측정하여 조정합니다.

## 물체 탐지 방안 비교

| 방안 | 특징 | 적합한 경우 |
| --- | --- | --- |
| [경량 YOLO](https://docs.ultralytics.com/models/yolo11) + [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/) | 사전 학습 모델로 빠르게 시작할 수 있습니다. CPU에서 시험한 뒤 지원 GPU로 옮길 수 있고, 모델 크기와 입력 해상도로 속도와 정확도를 조절할 수 있습니다. | 첫 시제품 |
| YOLO + [TensorRT/DeepStream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html) | NVIDIA 장비에서 추론을 가속하고 카메라 입력, 추적, 영상 출력을 한 파이프라인으로 구성할 수 있습니다. 장비와 SDK에 맞는 설정이 필요합니다. | Jetson 등 NVIDIA 기반 온보드 컴퓨터 |
| SSD MobileNet 계열 + [MediaPipe Object Detector](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/ObjectDetector) | 처리 자원이 적은 장치에서 비교할 수 있는 경량 모델입니다. 드론 영상 속 작은 사람과 차량의 탐지 성능은 실제 영상으로 확인해야 합니다. | 저전력 CPU 또는 전용 가속기를 사용하는 장비 |
| [RT-DETR](https://arxiv.org/abs/2304.08069) | 실시간 탐지를 목표로 한 Transformer 계열 모델입니다. 정확도 중심의 비교 후보이며, 온보드 처리 속도와 메모리 사용량을 실측해야 합니다. | 정확도가 우선이고 연산 자원에 여유가 있는 장비 |

우선 경량 YOLO와 ONNX Runtime으로 `카메라 입력 → 탐지 → 붉은 박스 표시 → 외부 전송` 흐름을 구현합니다. 이후 실제 온보드 장비에서 초당 처리 프레임 수, 영상 지연, 전력 소모, 멀리 있는 사람·차량의 탐지율을 측정해 가속 방식이나 모델 변경 여부를 결정합니다. 드론의 높은 시점에서 촬영한 물체가 잘 탐지되지 않으면 해당 환경의 영상을 수집해 모델을 추가 학습합니다.
