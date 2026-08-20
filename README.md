# ESP32-S3-BOX-3 Baby Monitor

A local-first dual-camera baby monitor for the Espressif ESP32-S3-BOX-3, built with ESPHome and a custom RTSP/RTP media component.

The monitor receives media directly from IP cameras: RTP/JPEG video is reconstructed and decoded to a 320×240 RGB565 framebuffer, while RTP/AAC audio is decoded and played through the ESP32-S3-BOX-3 audio path. Media work is separated from the ESPHome main loop to reduce blocking and improve responsiveness.

## Highlights

- Two RTSP cameras with fast switching.
- Direct MJPEG/RTP video rendering on the 320×240 LCD.
- AAC 16 kHz mono audio playback through ES8311/I2S.
- Separate RTSP, audio, video and control tasks.
- Monitor ON/OFF, software volume, hardware mute and Night Mode.
- Last-frame retention during short stream interruptions.
- Lightweight on-screen status and alarm overlays.
- Adaptive per-camera sound alarms using Home Assistant sound-level sensors.
- Optional motion correlation for stronger alarm classification.
- Optional external crying detection, disabled by default.
- Diagnostics for RTSP state, FPS, bitrate, dropped/replaced frames, memory, Wi-Fi, battery and alarm thresholds.

## Tested baseline

- ESPHome 2026.7.4
- ESP-IDF 5.5.5
- ESP32-S3 at 240 MHz
- Octal PSRAM at 80 MHz
- ESP32-S3-BOX-3 LCD: 320×240 RGB565
- MJPEG over RTSP/RTP
- AAC 16 kHz mono audio

## Camera stream recommendation

Use a **low-resolution camera substream**, not the high-resolution primary stream.

A typical deployment is:

```text
Camera
 ├─ Main stream  -> NVR / Frigate / recorder
 └─ Substream    -> ESP32-S3-BOX-3
                    MJPEG
                    modest frame rate
                    low practical resolution
                    AAC 16 kHz mono
```

The display is only 320×240 and the component is designed for lightweight monitoring streams. It is not intended to decode high-resolution primary camera streams. Exact RTSP URLs and substream selectors are camera-vendor specific; consult your camera documentation.

## Requirements

### Required for video/audio monitoring

- ESP32-S3-BOX-3
- ESPHome
- One or two compatible RTSP cameras
- MJPEG video stream
- AAC audio compatible with the tested 16 kHz mono path

### Required for adaptive sound alarms

- Home Assistant connected through the ESPHome API
- A sound-level sensor for each camera
- Optional motion binary sensors for motion/sound correlation

### Optional

- A Home Assistant binary sensor that reports crying events
- Frigate audio classification can provide such a sensor, but Frigate is not required

Cry detection is disabled by default. See [Optional cry detection](docs/configuration.md#optional-cry-detection).

## Repository layout

```text
.
├── baby_monitor.yaml
├── secrets.example.yaml
├── components/baby_monitor/
├── docs/
└── examples/
```

## Installation

1. Clone or copy this repository into your ESPHome configuration directory.
2. Copy `secrets.example.yaml` to `secrets.yaml` and enter your Wi-Fi, camera and network values.
3. Set the camera RTSP path in `baby_monitor.yaml` for your camera model.
4. Configure the four Home Assistant entity substitutions for motion and sound level.
5. Prefer a low-resolution MJPEG substream.
6. Validate and compile the configuration with ESPHome.
7. Flash the ESP32-S3-BOX-3.
8. Verify Monitor ON/OFF, audio, camera switching, Night Mode and stream recovery.

See [Configuration](docs/configuration.md) for details.

## Alarm behavior

The alarm engine maintains an independent adaptive background baseline for each camera. T1 and T2 are calculated from that baseline plus configurable offsets and safety floors. Quiet samples slowly update the baseline; a separate stable-background path can follow a persistent change in ambient noise without learning alarm-like levels.

See [Alarm engine](docs/alarm-engine.md) for the complete algorithm, timing, re-arming rules and tuning examples.

## Privacy and network configuration

Store credentials and installation-specific addresses in `secrets.yaml`. The provided `.gitignore` excludes that file.

Do not expose camera RTSP services or the ESPHome API directly to the public Internet.

## License

Original project source code and documentation in this repository are licensed under the Apache License 2.0.

Build-time dependencies retain their own licenses. In particular, `espressif/esp_audio_codec` uses the **Espressif Modified MIT License** and is restricted to use with Espressif Systems products. This project targets the ESP32-S3-BOX-3, an Espressif product.

See [Third-party notices](THIRD_PARTY_NOTICES.md) and [Licensing](docs/licensing.md).

## Trademarks

ESPHome, Home Assistant, Espressif, ESP32, ESP32-S3-BOX-3, Frigate and other product names or trademarks are the property of their respective owners. This independent project is not affiliated with or endorsed by those organizations.
