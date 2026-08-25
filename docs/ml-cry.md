# Local Cry ML

## Purpose

v1.5.0 performs cry classification locally on the ESP32-S3 and uses the local classifier as a production Level-2 alarm source.

## Model

The firmware expects the upstream model:

```text
chayuto/yamnet-cry-distill-int8
```

The upstream model is a compact INT8 audio classifier with a 521-logit output. The firmware uses the class indexes documented by the upstream model card.

The public repository does not bundle its binary. Use:

```bash
python3 tools/fetch_cry_model.py
```

The fetcher verifies:

```text
size    112,848 bytes
SHA-256 cf7f879e2ae065f06ddc209830395bc1b448a5f819db44c7d40195095758f5ba
```

## Frontend

The input is source PCM16 at 16 kHz mono.

Important constants from the implementation:

```text
ring / patch samples       15,600
new-sample hop              7,680
mel frames                     96
mel bins                       64
FFT size                      512
analysis window samples       400
frame hop samples             160
```

At 16 kHz:

- 400 samples = 25 ms;
- 160 samples = 10 ms;
- 15,600 samples = 0.975 s of source PCM;
- 7,680 samples = 0.480 s between eligible inference windows.

The firmware creates a 96×64 log-mel INT8 tensor.

## Inference scheduler

The Cry ML task normally runs on core 0 at priority 1.

Only `MicroInterpreter::Invoke()` is temporarily boosted to priority 5, then immediately returned to the previous priority. This reduces RTSP-driven jitter during the graph execution window without running the entire frontend/postprocessing path at the high priority.

## Output interpretation

The model produces 521 logits.

The firmware applies softmax and publishes selected classes:

```text
0   Speech
4   Babbling
9   Yell
11  Screaming
19  Crying, sobbing
20  Baby cry, infant cry
```

The production Cry score is:

```text
P(19) + P(20)
```

and `Cry ML Candidate` is true when that score is at or above **Cry ML Threshold**.

## Alarm integration

A new candidate for the active camera creates a 60-second Cry hold.

```text
Cry ML Candidate
       |
       v
cry_until = now + 60 s
       |
       v
Alarm Level 2
```

The current UI also sets the Cry bit in the alarm-cause mask.

## Limitations

This is an audio classifier, not a safety-certified medical or life-safety detector.

The upstream model card reports deployment-specific threshold sensitivity and limited generalization testing. Treat Cry ML as an additional monitoring signal, not as a guaranteed detector.

Room acoustics, microphone gain, camera audio processing, speech and background noise can all change the score distribution. Always calibrate in the target environment.

## Diagnostics

Home Assistant exposes:

- Cry ML Status
- Cry ML Model Ready
- Cry ML Candidate
- Cry ML Score
- Cry ML Baby Cry
- Cry ML Crying
- Cry ML Speech
- Cry ML Yell Scream
- Cry ML Babbling
- Cry ML Inference
- Cry ML Inference Count
- Cry ML Dropped Windows
- Cry ML Model Size
