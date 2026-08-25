# Architecture

## Design goal

The ESP32-S3-BOX-3 performs media transport, decode, local analysis, alarm evaluation and UI rendering on one device. The design therefore separates work by ownership and uses PSRAM for large long-lived allocations while reserving internal/DMA-capable memory for hardware paths that require it.

## Main execution contexts

| Context | Core | Priority | Main responsibility |
|---|---:|---:|---|
| `baby_ctrl` | 0 | 2 | Serialize Monitor ON/OFF and camera changes |
| `baby_rtsp` | 0 | 4 | RTSP session, RTP receive, reconnect |
| `baby_audio` | 1 | 5 | AAC decode, source PCM analysis, speaker output |
| `baby_video` | 1 | 3 | JPEG decode, motion analysis, UI and LCD transfer |
| Cry ML worker | 0 | 1 normally | Log-mel frontend and TFLite Micro inference |
| Cry ML `Invoke()` window | 0 | 5 temporarily | Deterministic TFLite graph execution |

The RTSP, audio and video task stacks are requested in PSRAM. Task control blocks and system resources that require internal RAM remain internal.

## Memory strategy

- QVGA framebuffer: `320 × 240 × 2 = 153,600` bytes in PSRAM.
- LCD staging: `320 × 16 × 2 = 10,240` bytes in internal DMA-capable RAM.
- Audio queue depth: 8 compressed AAC access units.
- Cry ML tensor arena: approximately 1 MiB.
- Cry PCM ring and snapshot buffers: 15,600 samples each, allocated in PSRAM.

See [Detailed dataflow](dataflow.md).
