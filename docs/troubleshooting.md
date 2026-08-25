# Troubleshooting

## No video

Verify host, port, credentials and RTSP path. Confirm that the selected substream is MJPEG.

The firmware is not a general H.264/H.265 decoder and is not intended for high-resolution primary streams.

## Audio missing

Confirm that the RTSP session advertises compatible AAC audio and that the hardware MUTE latch is not active.

After reboot, software audio intent starts ON, but physical MUTE has priority.

## Cry ML Model Ready is OFF

Run:

```bash
python3 tools/fetch_cry_model.py
```

and rebuild.

Check the script's size and SHA-256 verification output.

## Cry ML false positives

Increase **Cry ML Threshold** gradually, for example by 0.005.

Observe the individual speech, yell/scream, babbling and cry diagnostics while reproducing the false-positive sound.

## Real cries are missed

Lower **Cry ML Threshold** in small steps and verify that source audio is healthy.

Do not compensate for missing/very low RTSP audio by lowering the threshold aggressively.

## Sound alarm triggers too easily

Determine whether the active threshold is controlled by the offset or floor.

Then increase the relevant offset or make the floor less negative.

## Sound alarm does not trigger

Check:

- Local Sound Level;
- Alarm Baseline;
- Alarm T1;
- Alarm T2;
- Alarm T1/T2 confirmation;
- sound-gate behavior after camera switch.

Lower the relevant offset or make the floor more negative only after confirming the input signal is valid.

## Motion false alarms during IR switching

Tune Motion Scene Area Threshold, Motion Scene Luma Threshold and Motion Scene Settle Time before reducing normal motion sensitivity.

## Frequent no-stream placeholder

Check Wi-Fi quality, camera RTSP server stability and substream workload.

## SPI err 101 / display instability

The stable architecture intentionally uses a 16-row, 10,240-byte internal DMA stage rather than pushing the full PSRAM framebuffer directly through the SPI transfer path. Avoid increasing DMA staging without measuring `largest_internal` and `largest_dma`.
