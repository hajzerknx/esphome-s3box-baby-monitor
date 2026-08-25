# Alarm engine and tuning

The v1.5.0 alarm system combines three local signal sources:

1. adaptive sound thresholds T1/T2;
2. frame-based local motion;
3. local Cry ML.

The final decision is evaluated every 250 ms for the currently selected camera.

# Alarm levels

## Level 0

No current alarm cause.

## Level 1

Level 1 is produced by either:

```text
motion
OR
confirmed T1 sound
```

## Level 2

Level 2 is produced by any of:

```text
Cry ML candidate
OR
confirmed T2 sound
OR
motion + confirmed T1 sound
```

Each cause has its own hold time. The cause mask is independent from the final numeric level, so the UI can show multiple simultaneous reasons.

# Adaptive sound thresholds

Each camera has an independent baseline `B`.

```text
T1 = max(B + T1 offset, T1 floor)
T2 = max(B + T2 offset, T2 floor)
```

The offset makes the thresholds move with the room background. The floor prevents an extremely quiet learned background from making the thresholds overly sensitive.

## Default values

Camera 1:

```text
initial baseline = -76 dBFS
T1 offset        = +15 dB
T1 floor         = -60 dBFS
T2 offset        = +32 dB
T2 floor         = -40 dBFS
```

Camera 2:

```text
initial baseline = -65 dBFS
T1 offset        = +20 dB
T1 floor         = -45 dBFS
T2 offset        = +30 dB
T2 floor         = -32 dBFS
```

The two default sets are deliberately independent. Treat them as starting points, not universal calibration.

# Baseline learning

## Normal quiet-background path

The current source-PCM RMS dBFS is compared with the current T1.

If a valid sample is at least 6 dB below T1:

```text
sample < T1 - 6 dB
```

it is considered clearly safe background.

The baseline is updated with a slow EMA:

```text
B_new = 0.98 * B_old + 0.02 * sample
```

and clamped to the supported baseline range:

```text
-82 dBFS <= B <= -60 dBFS
```

This path follows long-term quiet changes without allowing a short loud event to move the baseline quickly.

## Temporary baseline recovery block

When sound approaches an alarm-like region or creates a real candidate, baseline learning is paused for a short recovery interval. This prevents the loud event itself from immediately raising the learned background.

## Stable elevated-background rescue

A room may become persistently louder because of a fan, ventilation, rain or another steady source. Such a level may no longer satisfy `T1 - 6 dB` but still be clearly below an alarm.

The rescue path can run only when:

```text
sample < T1 - 3 dB
```

and no blocking alarm-like condition is present.

It maintains a candidate background using approximately:

```text
candidate = 0.90 * candidate + 0.10 * sample
```

while tracking minimum and maximum values.

If the candidate spread exceeds 4 dB, the stability window restarts.

After at least 60 seconds of stable background, the real baseline may move toward the candidate. Movement is limited to roughly 1 dB per 30 seconds. This makes the adaptation intentionally conservative.

# Sound-gate arming

After Monitor ON, reconnect or camera switch, sound thresholds are not immediately armed.

The current camera must first provide a real quiet sample:

```text
sample < T1 - 3 dB
```

Only then are T1 and T2 armed.

This avoids treating a sound that was already present before a camera became active as a brand-new alarm event.

# Confirmation

A threshold crossing must persist.

Defaults:

```text
T1 confirmation = 1.0 s
T2 confirmation = 1.0 s
```

If the signal drops below the release boundary before confirmation completes, the candidate timer resets.

# Hysteresis and re-arm

After a confirmed threshold event, that threshold becomes disarmed.

It can re-arm only after the level falls below:

```text
threshold - hysteresis
```

Default hysteresis:

```text
2.0 dB
```

This prevents repeated triggers while the sound hovers around one threshold.

# Hold times

Defaults:

```text
T1 hold    = 20 s
T2 hold    = 60 s
motion hold = 20 s
Cry ML hold = 60 s
```

A hold keeps a cause active after its initial confirmation and avoids unstable UI/alarm flicker.

# Motion detector

The default motion detector parameters are:

```text
cell luma threshold      = 12
changed-area threshold   = 8%
motion confirmation      = 400 ms
release ratio            = 60% of trigger
release duration         = 600 ms
scene-change area        = 30%
scene global luma delta  = 5
scene settle             = 1.8 s
```

The detector compares a 40×30 grid derived from the decoded frame. It subtracts global luminance change before calculating changed area, then suppresses large exposure/IR/day-night scene transitions.

## Motion tuning

If ordinary image noise or compression produces false motion:

- increase **Motion Luma Threshold**;
- increase **Motion Area Threshold**;
- increase **Motion Confirmation**.

If real small movements are missed:

- decrease **Motion Luma Threshold** gradually;
- decrease **Motion Area Threshold** gradually.

If day/night IR switching creates alarms, tune the scene-change thresholds before making the normal motion detector less sensitive.

# Cry ML

Cry ML calculates:

```text
Cry score = P(Crying, sobbing) + P(Baby cry, infant cry)
```

Default candidate threshold:

```text
0.050 = 5%
```

A candidate immediately starts the 60-second Cry ML hold for the active camera and therefore produces Alarm Level 2.

The model also exposes diagnostic probabilities for speech, babbling, yell/scream and the two cry classes. These are useful when calibrating the threshold.

## Cry ML tuning

Start with `0.050`.

Observe:

- **Cry ML score**
- **Cry ML baby cry**
- **Cry ML crying**
- **Cry ML speech**
- **Cry ML yell scream**
- **Cry ML babbling**

during several real environments:

1. normal quiet room;
2. caregiver speech;
3. TV/music;
4. ordinary infant vocalization;
5. actual crying;
6. loud non-cry transients.

If false Cry ML alarms occur, raise the threshold in small steps such as `0.005`.

If real cries are consistently below the threshold, lower it in the same small steps.

Do not calibrate from one event. The upstream model card explicitly notes deployment-dependent calibration and reports useful operating points in approximately the 0.03–0.05 region for its own environment.

# Recommended sound tuning workflow

## Step 1 — observe the baseline

Leave the room in ordinary background conditions and watch:

- Alarm Baseline;
- Local Sound Level;
- Sound RMS dBFS.

Allow the baseline to settle.

## Step 2 — inspect T1 and T2

Watch:

- Alarm T1;
- Alarm T2.

Remember that the floor may dominate. For example, if:

```text
B = -76
offset = +15
floor = -60
```

then:

```text
B + offset = -61
T1 = max(-61, -60) = -60 dBFS
```

The floor, not the offset, is controlling T1.

## Step 3 — tune T1

T1 should represent meaningful sound activity, not only loud alarms.

For more sensitivity:

- lower the offset;
- or make the floor more negative.

For less sensitivity:

- raise the offset;
- or make the floor less negative.

## Step 4 — tune T2

T2 should represent a clearly stronger acoustic event than T1.

Keep enough separation between T1 and T2 that ordinary T1 activity does not repeatedly become Level 2.

## Step 5 — tune timing last

Only after the acoustic levels are sensible should you change:

- confirmation times;
- hold times;
- hysteresis.

Longer confirmation rejects short transients.
Longer hold changes alarm persistence, not detector sensitivity.
More hysteresis requires a larger release before the same threshold can trigger again.

# Freshness watchdog

Each camera records the timestamp of its most recent valid local sound sample.

If the active camera has no valid local sound sample for more than 30 seconds, the current alarm is cleared rather than trusting stale sound state.

# Night Mode

During Night Mode the media pipeline continues running.

Any alarm wakes the backlight and refreshes the wake deadline. After alarm activity ends, the display remains awake for the configured **Alarm Wake Time**, then turns off again if Night Mode is still active.
