# Configuration

## Camera names

At the top of `baby_monitor.yaml`:

```yaml
camera_1_name: "Camera 1"
camera_2_name: "Camera 2"
```

Change these to your preferred names.

The same strings appear on the LCD and as the two Home Assistant `Camera` select options.

## Credentials and hosts

Copy:

```text
secrets.example.yaml -> secrets.yaml
```

Configure each camera independently.

## RTSP paths

Set:

```yaml
camera_1_rtsp_path: "/your/camera/substream/path"
camera_2_rtsp_path: "/your/camera/substream/path"
```

Use each camera's low-resolution MJPEG substream.

## DHCP

The provided configuration keeps a `manual_ip` block because the stable installation uses a fixed address.

If you prefer DHCP, remove the `manual_ip` block.

## Cry ML

Run:

```bash
python3 tools/fetch_cry_model.py
```

before compiling.

If you intentionally do not want Cry ML, leave the provided one-byte model stub in place. The media, NoiseAnalyzer and MotionDetector paths still operate; Cry ML reports that no usable model is embedded.

## Home Assistant

The firmware exposes controls, diagnostics and tuning entities through ESPHome API. The local analyzers themselves use the RTSP media stream and do not depend on Home Assistant for their input signal.
