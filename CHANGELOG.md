# Changelog

## 1.5.0

Initial public release of the current stable v1.5 firmware line.

### Media and system

- Dual-camera direct RTSP/RTP monitoring.
- RTP/JPEG reconstruction and JPEGDEC rendering.
- AAC audio decode with queue depth 8.
- 320×240 RGB565 framebuffer in PSRAM.
- 16-row internal/DMA staging buffer for LCD transfers.
- Media task stacks explicitly placed in PSRAM where supported.
- Dedicated no-stream image placeholder.

### Local analysis

- NoiseAnalyzer on decoded source PCM.
- Adaptive per-camera T1/T2 sound alarms.
- Local MotionDetector on decoded RGB565 frames.
- Local TensorFlow Lite Micro Cry ML on decoded PCM.
- Cry ML promoted to a production Level-2 alarm source.

### UI

- Left-side status strip: RTSP, battery, Wi-Fi, optional MUTE, alarm causes.
- Alarm cause bits for motion, T1, T2 and Cry ML.
- T2 visually supersedes T1 while independent causes remain visible.
- Alarm border rendered below UI elements.
- Camera names configurable through substitutions and shared by LCD and Home Assistant Camera select.

### Audio behavior

- Software audio intent defaults to ON after reboot.
- Physical hardware MUTE always overrides software SOUND ON.
