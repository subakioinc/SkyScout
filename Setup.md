# SkyScout 설정 및 실행

이 문서는 현재 `src/skyscout` 구현을 Ubuntu 장비에서 실행하는 절차입니다. USB 카메라 또는 저장 영상을 입력받아 사람·차량을 탐지하고, 붉은 박스가 표시된 1080p 영상을 H.264/RTSP로 송출합니다. 아래 명령은 저장소 최상위 디렉터리에서 실행합니다.

## 1. 대상 장비 확인

- **Raspberry Pi:** Pi 4에서는 Ubuntu 22.04/24.04, Pi 5에서는 Ubuntu 24.04를 우선 대상으로 합니다. 실제 모델과 메모리, USB 카메라가 1080p를 지원하는지 확인합니다.
- **Jetson Nano:** Nano용 NVIDIA 공식 JetPack 환경은 Ubuntu 18.04 기반입니다. Ubuntu 22.04/24.04에서의 실행 경로는 확정되지 않았습니다. Nano에 설치하기 전에 [Plan.md](Plan.md)의 OS 결정 사항을 먼저 해결해야 합니다.
- **GCS:** 드론 장비의 IP 주소로 접속할 수 있고 RTSP/H.264를 수신할 수 있어야 합니다. 아래 예시는 RTSP TCP 전송으로 확인합니다.

현재 구현은 CPU용 ONNX Runtime과 GStreamer `x264enc`를 기본값으로 사용합니다. GPU 추론 및 장비별 하드웨어 인코더는 아직 실장비에서 검증되지 않았습니다.

## 2. Ubuntu 패키지와 Python 환경 설치

```bash
sudo apt update
sudo apt install python3-venv python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-tools v4l-utils
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
```

`--system-site-packages`는 Ubuntu의 `python3-gi`를 가상 환경에서 읽기 위해 필요합니다. 설치 후 다음 명령으로 RTSP 서버 바인딩과 기본 인코더를 확인합니다.

```bash
python -c 'import gi; gi.require_version("GstRtspServer", "1.0"); from gi.repository import GstRtspServer; print("GstRtspServer OK")'
gst-inspect-1.0 x264enc
```

## 3. USB 카메라 확인

카메라를 연결하고 장치 번호와 지원 포맷을 확인합니다.

```bash
ls /dev/video*
v4l2-ctl -d /dev/video0 --list-formats-ext
```

실행 시 `--source 0`은 카메라 인덱스 0을 뜻합니다. 필요하면 `--source /dev/video0`처럼 장치 경로를 지정합니다. 프로그램은 MJPEG, 1920×1080, 15fps를 요청하지만 카메라가 실제로 제공하는 값이 다를 수 있습니다. 실행 로그의 카메라 해상도·프레임률을 확인합니다. 카메라를 열 수 없다면 해당 사용자가 장치에 접근할 수 있는지 확인합니다.

## 4. 탐지 모델 준비

저장소에는 모델 가중치가 포함되지 않습니다. 첫 후보는 [VisDrone 영상으로 학습된 YOLO11n 가중치](https://huggingface.co/dronefreak/visdrone-yolo11n)입니다. 모델 카드의 라이선스와 데이터셋 사용 조건을 먼저 확인하고, 사용 가능한 경우 `best.pt`를 별도로 받습니다. 현재 탐지 코드는 이 모델의 **VisDrone 클래스 순서와 NMS가 포함되지 않은 YOLO11 ONNX 출력**을 전제로 합니다. 일반 COCO 모델이나 다른 구조의 ONNX 파일을 그대로 넣으면 실행되지 않습니다.

변환은 개발 컴퓨터에서 수행하는 편이 간편합니다. 개발 컴퓨터에도 이 저장소를 준비한 뒤 다음을 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . ultralytics
mkdir -p models
cp /path/to/best.pt models/best.pt
python -m skyscout.prepare_model --weights models/best.pt --image-size 640
```

생성된 `models/best.onnx`를 온보드 장비의 저장소 `models/`로 복사합니다. `models/`와 `*.onnx`는 Git 추적에서 제외됩니다. 이 변환 명령은 입력 크기가 고정된 모델을 만듭니다. 추론 입력 크기를 바꾸려면 다른 `--image-size`로 다시 변환해야 합니다.

## 5. 실행

온보드 장비에서 가상 환경을 활성화하고 실행합니다.

```bash
source .venv/bin/activate
skyscout --model models/best.onnx --source 0
```

기본값은 카메라 요청 및 송출 크기 1920×1080, 요청 프레임률 15fps, 추론 입력 640×640, RTSP 포트 8554, 경로 `/skyscout`입니다. `--fps`는 요청값이며 실제 처리·송출 속도를 보장하지 않습니다. 실행 중 종료는 `Ctrl+C`입니다.

저장 영상으로 반복 시험할 때는 다음처럼 실행합니다.

```bash
skyscout --model models/best.onnx --source /path/to/video.mp4 --loop
```

주요 옵션은 다음과 같습니다. 전체 목록은 `skyscout --help`로 확인할 수 있습니다.

| 옵션 | 기본값 | 용도 |
| --- | --- | --- |
| `--model` | 필수 | VisDrone YOLO11 ONNX 파일 경로 |
| `--source` | `0` | USB 카메라 인덱스, 장치 경로 또는 저장 영상 |
| `--width`, `--height` | `1920`, `1080` | 카메라 요청 및 RTSP 출력 크기 |
| `--fps` | `15` | 카메라 요청 및 RTSP 파이프라인의 기준 프레임률 |
| `--confidence`, `--iou` | `0.25`, `0.45` | 탐지 신뢰도·겹침 제거 임계값 |
| `--provider` | `CPUExecutionProvider` | 장비에 설치된 ONNX Runtime 실행 제공자 |
| `--port`, `--mount` | `8554`, `/skyscout` | RTSP 접속 주소 구성 |
| `--bitrate-kbps` | `2500` | 기본 `x264enc` 인코더의 비트레이트 |
| `--encoder-pipeline` | `x264enc` 설정 | 검증된 다른 GStreamer H.264 인코더 설정으로 교체 |

## 6. GCS에서 영상 수신

온보드 장비의 IP를 확인합니다.

```bash
hostname -I
```

GCS의 RTSP 수신 기능에 `rtsp://<드론-IP>:8554/skyscout`를 입력합니다. GCS가 FFmpeg의 `ffplay`를 제공한다면 다음 명령으로 먼저 확인할 수 있습니다.

```bash
ffplay -rtsp_transport tcp rtsp://<드론-IP>:8554/skyscout
```

송출 프로세스는 카메라를 계속 읽고 탐지하며, RTSP 클라이언트가 연결되면 영상을 전송합니다. 연결되지 않으면 드론과 GCS 사이의 IP 경로, 8554/TCP 접근, RTSP 주소와 GCS의 코덱 지원을 확인합니다.

## 7. 동작 확인과 문제 해결

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

| 현상 | 확인할 항목 |
| --- | --- |
| `model does not exist` | `models/best.onnx` 경로와 복사 여부 |
| `VisDrone classes` 또는 `unsupported YOLO output shape` | VisDrone YOLO11 가중치를 `nms=False`로 변환했는지 확인 |
| `GStreamer Python bindings are missing` | `python3-gi`, RTSP introspection 패키지 설치 및 `--system-site-packages` 가상 환경 사용 여부 |
| `no element x264enc` | `gstreamer1.0-plugins-ugly` 설치와 `gst-inspect-1.0 x264enc` 결과 |
| 카메라 연결 실패 | `/dev/video*`, 장치 접근 권한, 지원 해상도·포맷 확인 |
| RTSP 연결 실패 | GCS에서 IP·포트·경로 확인, RTSP TCP 접근 확인 |
| 탐지 속도가 느림 | 로그의 `processed ... fps` 확인. 모델 입력 크기 변경은 ONNX 재변환이 필요하며, 1080p 출력 인코딩 비용도 측정 |

현재 테스트는 탐지 후처리와 박스 표시를 확인합니다. Jetson Nano·Raspberry Pi의 실제 카메라, H.264 인코딩 성능, GCS 연결 및 비행 고도별 탐지 성능은 각 장비에서 추가 검증해야 합니다.
