# Licensing and IP review

This document records the technical licensing audit for the public v1.5.0 source repository. It is not legal advice.

## Repository-owned material

Original project source code, YAML configuration, helper scripts and documentation stored in this repository are licensed under Apache License 2.0 unless a file states otherwise.

## Material present in this repository

The repository contains project-owned ESPHome YAML, external-component C++/Python source, NoiseAnalyzer, MotionDetector and CryDetector integration code, documentation, helper scripts and a minimal Cry ML fallback stub.

The repository does not vendor ESPHome, ESP-IDF, JPEGDEC, `esp_audio_codec`, `esp-tflite-micro`, the Cry ML model artifact, vendor schematic files, third-party logos or branding artwork.

## ESPHome

ESPHome is a direct build/runtime dependency.

Upstream ESPHome states that its C++/runtime codebase is GPLv3 and its Python/other code is MIT-licensed.

ESPHome is not copied into this repository and retains its upstream licensing.

## JPEGDEC 1.8.4

The external component requests JPEGDEC 1.8.4.

Upstream JPEGDEC source identifies BitBank Software, Inc. and uses Apache License 2.0.

JPEGDEC is downloaded as a dependency and is not copied into this repository.

## Espressif esp_audio_codec 2.6.2

The firmware requests `espressif/esp_audio_codec` 2.6.2.

Upstream source uses:

```text
SPDX-License-Identifier: LicenseRef-Espressif-Modified-MIT
```

Its terms restrict use to Espressif Systems products and prohibit redistribution for use with non-Espressif products.

The target ESP32-S3-BOX-3 is an Espressif product.

This dependency is not relicensed by this project.

## Espressif esp-tflite-micro 1.3.1

The firmware requests `espressif/esp-tflite-micro` 1.3.1.

Espressif publishes that component and its examples under Apache License 2.0 and notes that TensorFlow library code and `third_party` code keep their respective upstream licenses.

The component is not vendored here.

## Cry ML model

The firmware can optionally use `chayuto/yamnet-cry-distill-int8`.

The upstream publisher declares the model under the MIT License.

This repository does not redistribute the model artifact. The fetch helper pins a model revision, verifies the expected file size and SHA-256, and generates an ignored local include file.

The project's Apache-2.0 license does not relicense the model.

The upstream model card remains the authoritative source for model architecture, training methodology, data provenance, evaluation and limitations.

## ESP-IDF

ESP-IDF is an external framework/toolchain dependency.

ESP-IDF original source is generally Apache License 2.0 and also contains third-party material under additional licenses. Upstream source-file headers and license inventories take precedence.

ESP-IDF is not vendored here.

## Standards and protocol data

The custom media component implements RTSP/RTP and RTP/JPEG reconstruction and uses standard JPEG reconstruction tables.

No third-party JPEG decoder source is copied into the project component; JPEG decoding is provided by the separately downloaded JPEGDEC dependency.

## Trademarks

Third-party product and project names are used only to identify compatibility, dependencies or upstream artifacts.

No third-party logos or branding artwork are included.

## Audit conclusion

For publication of this source repository, no paid commercial library or proprietary third-party source file was identified in the files being published.

The material points that should remain visible to users are:

- repository-owned source: Apache-2.0;
- ESPHome: upstream GPLv3/MIT split;
- JPEGDEC: Apache-2.0;
- `esp_audio_codec`: Espressif Modified MIT with the Espressif-product restriction;
- `esp-tflite-micro`: Apache-2.0 component with upstream subcomponent licenses;
- Cry ML model: separate MIT-licensed upstream artifact, not stored in this repository.
