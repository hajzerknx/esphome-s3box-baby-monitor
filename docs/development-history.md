# Development history

This project changed direction several times before reaching the current v1.5.0 architecture. The most useful history is not a list of every development build, but the sequence of engineering decisions that changed how the device works.

## Stage 1 — almost entirely ESPHome

The first working prototypes tried to build the complete monitor from standard ESPHome building blocks.

```text
IP camera
   |
   v
go2rtc / FFmpeg
   |
   v
HTTP JPEG snapshots
   |
   v
ESPHome online_image
   |
   v
LVGL
   |
   v
ESP32-S3-BOX-3 LCD
```

Audio followed a separate external path. A server-side bridge produced WAV/PCM audio that the ESP32 received over HTTP and sent to the ES8311/I2S output.

The approach worked, but JPEG downloads, image decode, LVGL, network traffic and audio competed for CPU time and memory. Refresh intervals, JPEG buffers, pre-scaled 320×240 snapshots, TCP buffering and display-update strategies were optimized, yet the high-level pipeline remained the main limitation.

The important conclusion was that further YAML-level optimization would not solve the core problem.

## Stage 2 — custom external component and direct RTSP

The project was rebuilt around a custom ESPHome external component named `baby_monitor`.

The first goal was lifecycle correctness:

```text
one active camera
      |
      v
STOP -> cleanup -> START
```

The component then gained a native RTSP client with OPTIONS, DESCRIBE, SDP parsing, Basic/Digest authentication, SETUP, PLAY, interleaved RTP/RTCP over TCP, reconnect and cleanup outside the ESPHome main loop.

This became the architectural turning point: ESPHome remained the device framework and Home Assistant integration layer, while time-critical media handling moved into dedicated C++ code.

## Stage 3 — direct RTP/JPEG video

The HTTP snapshot path was replaced by direct RTP/JPEG processing:

```text
RTSP/RTP
   |
   v
RTP/JPEG
   |
   v
RFC 2435 reconstruction
   |
   v
JPEGDEC
   |
   v
320x240 RGB565
   |
   v
LCD
```

LVGL disappeared from the camera-image path.

The media engine later adopted dedicated RTSP/video tasks, two JPEG frame slots, a latest-frame-wins queue, PSRAM-backed buffers and a full 320×240 RGB565 framebuffer.

That framebuffer later became a shared representation for both UI rendering and MotionDetector.

## Stage 4 — direct AAC audio

The external audio bridge was replaced with direct RTP/AAC:

```text
RTSP/RTP
   |
   v
MPEG4-GENERIC AAC
   |
   v
AAC decode
   |
   v
PCM16 / 16 kHz / mono
   |
   v
ES8311 / I2S
```

Audio received its own worker task and session lifecycle.

Later work separated source-signal analysis from playback volume. NoiseAnalyzer and CryDetector consume PCM before software gain, so speaker volume does not change alarm thresholds or ML input amplitude.

## Stage 5 — stable media engine and lightweight UI

The media engine converged on four clear ownership domains:

```text
baby_ctrl  -> monitor state and camera changes
baby_rtsp  -> RTSP/RTP network lifecycle
baby_audio -> AAC decode and PCM path
baby_video -> JPEG decode, analysis and display
```

The UI stayed intentionally lightweight and was drawn directly into the RGB565 framebuffer.

The device gained local controls, Wi-Fi/battery status, hardware MUTE, software volume, Night Mode, reconnect handling and a dedicated no-stream placeholder.

## Stage 6 — adaptive sound alarms

The first alarm generation introduced the adaptive two-threshold model:

```text
T1 = max(background baseline + T1 offset, T1 floor)
T2 = max(background baseline + T2 offset, T2 floor)
```

Its key feature was the adaptive background baseline rather than fixed thresholds. Confirmation, hold, hysteresis and re-arm rules were added so short transients or already-active sound would not create unstable alarms.

## Stage 7 — local analysis modules

A later refactor added zero-copy analysis points to the existing media pipeline:

```text
source PCM -----------------> NoiseAnalyzer
decoded RGB565 framebuffer -> MotionDetector
source PCM -----------------> CryDetector
```

### NoiseAnalyzer

NoiseAnalyzer moved sound-level measurement onto the BOX-3 itself.

It computes RMS, peak and dBFS from decoded source PCM and feeds the adaptive T1/T2 alarm logic.

### MotionDetector

MotionDetector reuses the existing framebuffer rather than allocating a second full image.

The image is reduced to a small luminance grid. Local differences are compared over time while global luminance change is removed. A scene-change guard suppresses exposure/IR day-night transitions.

## Stage 8 — CryDetector and the ML debugging phase

Cry detection was the most difficult local analysis module.

```text
PCM16 @ 16 kHz
      |
      v
ring buffer
      |
      v
15,600-sample snapshot
      |
      v
96 x 64 log-mel frontend
      |
      v
TensorFlow Lite Micro
      |
      v
521 logits
      |
      v
softmax
      |
      v
Cry score
```

Cry ML initially ran in diagnostic/shadow mode so its frontend, model loading, tensor arena, timing and score distribution could be measured before it affected production alarms.

The debugging effort included:

- validating model bytes and alignment;
- verifying tensor shapes and quantization;
- reproducing the TFLite Micro memory planner instead of guessing arena size;
- diagnosing tensor-arena allocation and graph preparation;
- measuring `Invoke()` time and dropped inference windows;
- preventing the ML worker from starving RTSP;
- validating the log-mel frontend;
- separating source PCM from playback gain;
- exposing class probabilities and inference diagnostics.

The final scheduler keeps ordinary frontend/post-processing work at low priority and temporarily boosts priority only around `MicroInterpreter::Invoke()`.

After this diagnostic period, local Cry ML was promoted into the production Level-2 alarm path.

## Stage 9 — internal RAM and DMA pressure

Adding local ML changed the optimization target.

The ESP32-S3 now had to keep these workloads alive together:

```text
RTSP
AAC
JPEGDEC
LCD
NoiseAnalyzer
MotionDetector
CryDetector / TFLM
```

Total free PSRAM was no longer enough to judge health. The critical metrics became:

```text
free internal RAM
largest internal block
free DMA-capable RAM
largest DMA-capable block
```

A system could still have substantial total free memory while failing one contiguous internal/DMA allocation.

Allocation ownership therefore became explicit:

- persistent JPEGDEC storage was reserved deliberately;
- large media-task stacks were moved to PSRAM where possible;
- Cry ML received its own tensor arena and PCM buffers;
- internal RAM was preserved for resources that actually require it.

## Stage 10 — SPI/DMA display failures

A difficult runtime symptom was intermittent SPI transmission failure (`err 101`).

The framebuffer lives in PSRAM:

```text
320 x 240 x 2 = 153,600 bytes
```

but the LCD transport requires DMA-capable internal memory.

The stable architecture introduced one persistent DMA staging buffer:

```text
320 x 16 x 2 = 10,240 bytes
```

The full frame is sent in bounded 16-row chunks.

Diagnostic snapshots around SPI failures recorded internal/DMA heap state, task context, display-flush state and concurrent Cry activity. This helped separate LCD transport failures from general PSRAM availability.

## Stage 11 — AAC allocation and coexistence

AAC became another important part of the RAM investigation.

The decoder and PCM buffers compete with the same constrained internal-memory environment used by display transport and parts of the ML/runtime stack.

The later architecture therefore:

- keeps the AAC access-unit queue bounded;
- uses queue depth 8 in the stable line;
- explicitly manages AAC PCM storage;
- records AAC open attempts, failures and heap state;
- keeps source PCM analysis independent from playback gain;
- preserves deterministic reset behavior across reconnects and camera changes.

This stage was primarily about making AAC coexist reliably with Cry ML, JPEG decode and LCD DMA rather than adding new audio features.

## Stage 12 — v1.5.0 stable architecture

By v1.5.0, the project had evolved from an ESPHome UI around externally prepared media into a mostly self-contained media and local-analysis system:

```text
IP camera substream
        |
        v
     RTSP/RTP
        |
   +----+--------------------------+
   |                               |
   v                               v
RTP/JPEG                         RTP/AAC
   |                               |
   v                               v
JPEGDEC                        AAC decoder
   |                               |
   v                               v
RGB565 framebuffer              source PCM
   |                               |
   +-> MotionDetector              +-> NoiseAnalyzer
   |                               |
   |                               +-> CryDetector / TFLM
   |                               |
   +---------------+---------------+
                   |
                   v
             alarm evaluator
                   |
         +---------+----------+
         |                    |
         v                    v
      local UI          Home Assistant
```

The v1.5 UI exposes independent causes for motion, T1/T2 sound and Cry ML. T2 visually supersedes T1, while unrelated causes remain visible. Physical MUTE retains priority over software audio intent.

The central architectural lesson is that ESPHome remained highly useful as the device framework and integration layer, while reliable media and real-time analysis required explicit lifecycle, scheduling and memory ownership inside a purpose-built external component.
