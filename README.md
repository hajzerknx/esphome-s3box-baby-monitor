# ESP32-S3-BOX-3 Baby Monitor

A local-first dual-camera baby monitor for the Espressif ESP32-S3-BOX-3, built with ESPHome and a custom RTSP/RTP media component.

Version **1.5.0** adds a mature local analysis pipeline on top of the stable media engine:

- direct RTP/JPEG video and AAC audio;
- local PCM sound analysis with adaptive T1/T2 alarms;
- local frame-based motion detection;
- local on-device Cry ML using TensorFlow Lite Micro;
- a left-side status strip showing RTSP, battery, Wi-Fi, MUTE and active alarm causes;
- explicit task and memory placement designed for ESP32-S3 internal RAM, DMA memory and PSRAM constraints.

## Key features

- Two RTSP cameras with user-defined display names.
- Camera names configured once in `substitutions` and reused on the LCD and in the Home Assistant `Camera` select entity.
- Low-latency MJPEG/RTP video to the 320×240 LCD.
- AAC 16 kHz mono audio through ES8311/I2S.
- Software volume and physical hardware MUTE.
- Night Mode with alarm-driven screen wake.
- Adaptive sound baseline per camera.
- Local motion detection from decoded RGB565 video.
- Local Cry ML from decoded source PCM.
- Alarm Levels 0/1/2 with independent cause icons.
- Dedicated 320×240 no-stream image placeholder.
- Extensive Home Assistant diagnostics and tuning controls.

## Camera names

Edit only these substitutions:

```yaml
substitutions:
  camera_1_name: "Nursery"
  camera_2_name: "Bedroom"
```

The values are used by:

1. the camera label rendered on the monitor;
2. the options exposed by the Home Assistant `Camera` select entity;
3. camera-specific alarm tuning entity names;
4. alarm-reason text and logs.

Use short names without embedded double quotes.

## Camera stream requirements

Use a **low-resolution MJPEG substream**.

The monitor display is 320×240 and the firmware is tuned for lightweight monitoring streams. A 1080p, 4 MP or 8 MP primary stream adds network, JPEG decode and memory pressure without improving the LCD output.

Recommended topology:

```text
Camera
 ├─ Main stream  -> NVR / recorder
 └─ Substream    -> ESP32-S3-BOX-3
                    MJPEG
                    modest frame rate
                    low practical resolution
                    AAC 16 kHz mono
```

RTSP paths are vendor-specific. Configure `camera_1_rtsp_path` and `camera_2_rtsp_path` for your camera models.

## Cry ML model setup

The public repository does **not** bundle the model binary.

Run:

```bash
python3 tools/fetch_cry_model.py
```

The script downloads the upstream `chayuto/yamnet-cry-distill-int8` model, verifies its pinned revision, expected 112,848-byte size and SHA-256, then generates the untracked `components/baby_monitor/cry_model_blob.inc` file.

Without this step, the firmware still builds and the media, sound and motion paths remain available, but Cry ML reports that no model is embedded.

See [ML Cry](docs/ml-cry.md) and [Third-party model](THIRD_PARTY_MODEL.md).

## Installation

1. Copy or clone the repository into your ESPHome configuration directory.
2. Copy `secrets.example.yaml` to `secrets.yaml`.
3. Enter Wi-Fi, camera and optional static-network values.
4. Set `camera_1_name` and `camera_2_name`.
5. Configure each camera's RTSP substream path.
6. Run `python3 tools/fetch_cry_model.py` if you want local Cry ML.
7. Validate and compile `baby_monitor.yaml`.
8. Flash the ESP32-S3-BOX-3.
9. Verify both cameras, audio, physical MUTE, Night Mode, motion, sound alarms and Cry ML diagnostics.

## Documentation

- [Architecture](docs/architecture.md)
- [Detailed dataflow](docs/dataflow.md)
- [Development history](docs/development-history.md)
- [Alarm engine and tuning](docs/alarm-engine.md)
- [Cry ML](docs/ml-cry.md)
- [Configuration](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Licensing and third-party software](docs/licensing.md)

## Tested baseline

The v1.5.0 release has been developed and tested on:

- **Hardware:** Espressif ESP32-S3-BOX-3.
- **ESPHome:** 2026.7.x, using the ESP-IDF framework.
- **Display:** the integrated 320×240 LCD of the ESP32-S3-BOX-3.
- **Audio output:** the integrated ES8311/I2S audio path.
- **Camera topology:** two IP cameras selected one at a time through a low-resolution RTSP substream.
- **Video:** RTP/JPEG (MJPEG), decoded locally to a 320×240 RGB565 framebuffer.
- **Audio:** RTP/AAC, 16 kHz mono in the tested configuration.
- **Local analysis:** NoiseAnalyzer, MotionDetector and optional CryDetector/TensorFlow Lite Micro.
- **Home Assistant integration:** native ESPHome API entities for control, diagnostics and alarm tuning.

The stable media baseline uses an AAC queue depth of 8, persistent JPEG decode resources, a 16-row LCD DMA staging buffer and PSRAM-backed media-task stacks.

Hardware compatibility beyond the **ESP32-S3-BOX-3 has not been tested**. In particular, the configuration assumes the BOX-3 display, touch/button inputs, ES8311 audio path, PSRAM layout and board-specific pin assignments.

## Requirements

Before using this project, you need:

- an **Espressif ESP32-S3-BOX-3**;
- a current ESPHome installation compatible with the configuration's declared minimum version;
- Home Assistant with the ESPHome integration if you want the exposed controls, diagnostics and tuning entities;
- one or two IP cameras providing an RTSP **MJPEG/RTP substream**;
- camera credentials and the correct vendor-specific RTSP paths;
- an AAC audio track compatible with the tested 16 kHz mono audio path if camera audio is required;
- a Wi-Fi network reachable by both the monitor and cameras;
- Python 3 with network access during setup if local Cry ML is enabled, so `tools/fetch_cry_model.py` can obtain and verify the optional upstream model.

A dedicated low-resolution camera substream is strongly recommended. The firmware is designed around the 320×240 display and is **not intended for high-resolution primary camera streams**.


## License

Original project source code and documentation in this repository are released under the Apache License 2.0.

Third-party dependencies and the optional downloaded Cry ML model retain their own licenses. In particular, `esp_audio_codec` uses the Espressif Modified MIT License and is restricted to use with Espressif products.

See `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_MODEL.md` and `docs/licensing.md`.

## Trademarks

Third-party names are used only to identify compatibility, dependencies or upstream artifacts. All trademarks remain the property of their respective owners. No affiliation or endorsement is implied.
