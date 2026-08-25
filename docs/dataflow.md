# Detailed dataflow

This document follows the actual v1.5.0 stable data path from the network to display/audio and then into local analysis and alarm presentation.

## 1. Session control

```text
Home Assistant / touchscreen / Home button
                 |
                 v
             baby_ctrl
                 |
        +--------+--------+
        |                 |
     Monitor           camera switch
      ON/OFF                |
        |                   |
        +------ serialized -+
                 |
                 v
        start/stop media tasks
```

The control task serializes lifecycle changes. A camera switch clears old per-camera alarm holds and sound-gate state before selecting the new camera.

## 2. RTSP and RTP ingress

```text
IP camera substream
       |
       | RTSP setup / play
       v
    baby_rtsp
       |
       +--------------------+
       |                    |
       v                    v
   RTP/JPEG              RTP/AAC
       |                    |
       v                    v
 video reassembly       AAC AU queue
                        depth = 8
```

The RTSP task owns the socket, handshake, RTP receive and reconnect. It does not perform full video or audio decode.

## 3. Video path

```text
RTP/JPEG packets
      |
      v
RFC 2435 JPEG reconstruction
      |
      v
latest-frame video queue
      |
      v
baby_video (core 1, P3)
      |
      v
JPEGDEC
      |
      v
320x240 RGB565 framebuffer in PSRAM
      |
      +------------------------+
      |                        |
      v                        v
MotionDetector            UI overlay
      |                        |
      |                        v
      |                 alarm border + status
      |                        |
      +------------------------+
               |
               v
       16-row DMA staging
          10,240 bytes
               |
               v
              LCD
```

The framebuffer is the shared decoded representation for display and motion analysis. Motion detection therefore does not create a second full-resolution image buffer.

### MotionDetector

The 320×240 framebuffer is reduced to a 40×30 luma grid. Each grid cell samples four points. A global luma delta is removed before per-cell change scoring, helping distinguish local movement from whole-frame exposure changes.

A scene-change guard suppresses events such as IR/day-night switching or large exposure jumps. After a scene change, motion remains suppressed for the configured settle time.

## 4. Audio path

```text
RTP/AAC
   |
   v
AAC AU queue (8)
   |
   v
baby_audio (core 1, P5)
   |
   v
Espressif AAC decoder
   |
   v
source PCM16, 16 kHz mono
   |
   +--------------------+---------------------+
   |                    |                     |
   v                    v                     v
NoiseAnalyzer        CryDetector          playback gain
   |                    |                     |
   v                    v                     v
dBFS metrics        PCM ring buffer       ES8311 / I2S
   |                    |
   |                    v
   |             15,600-sample patch
   |                    |
   |                    v
   |             96 x 64 log-mel
   |                    |
   |                    v
   |          TensorFlow Lite Micro
   |                    |
   |                    v
   |            521 output logits
   |                    |
   |                    v
   |        softmax class probabilities
   |                    |
   +----------+---------+
              |
              v
       local alarm inputs
```

Both NoiseAnalyzer and CryDetector receive **source PCM before playback gain**. Changing speaker volume therefore does not change the sound threshold domain or the ML input amplitude.

## 5. NoiseAnalyzer flow

For every decoded PCM block:

1. calculate block RMS and peak;
2. convert them to dBFS;
3. apply attack/release RMS smoothing;
4. expose smoothed dBFS and diagnostics;
5. feed the YAML adaptive T1/T2 alarm logic.

The smoothing constants are approximately 120 ms attack and 700 ms release.

## 6. Cry ML flow

```text
PCM16 @ 16 kHz
   |
   v
ring buffer
   |
   | newest 15,600 samples
   v
snapshot
   |
   | 25 ms analysis window
   | 10 ms frame hop
   v
96 frames
   |
   v
FFT512 -> 64-bin mel projection -> log
   |
   v
96 x 64 INT8 input tensor
   |
   v
TFLite Micro student model
   |
   v
521 INT8 logits
   |
   v
softmax
   |
   +--> class 19: Crying, sobbing
   +--> class 20: Baby cry, infant cry
   |
   v
Cry score = P(class 19) + P(class 20)
   |
   v
Cry score >= Cry ML Threshold
   |
   v
Cry ML Candidate
```

A new inference window becomes eligible every 7,680 new samples, approximately 480 ms at 16 kHz. If the ML worker is still busy, the pending-window logic avoids building an unbounded backlog and exposes dropped-window diagnostics.

## 7. Alarm aggregation

The current camera contributes four independent cause states:

```text
local motion --------------------+
confirmed T1 sound --------------+
confirmed T2 sound --------------+--> 250 ms alarm evaluator
local Cry ML candidate ----------+
                                      |
                                      +--> Level 0 / 1 / 2
                                      +--> cause bit mask
                                      +--> Last Alarm Reason
                                      +--> Night Mode wake deadline
                                      +--> LCD alarm border/icons
```

Cause-mask bits are:

- bit 0: motion;
- bit 1: T1;
- bit 2: T2;
- bit 3: Cry ML.

When T2 is active, the UI shows T2 instead of T1, while unrelated causes such as motion or Cry ML remain visible.

## 8. Home Assistant role

Home Assistant is used for control, tuning and diagnostics. The core motion, sound analysis and Cry ML computation are local to the BOX-3. The RTSP camera remains the source of the media data used by those analyzers.
