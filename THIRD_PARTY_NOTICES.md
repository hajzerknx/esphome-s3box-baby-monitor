# Third-party notices

This file inventories the third-party software and artifacts directly referenced by this source package. It does not attempt to reproduce the full dependency tree of ESPHome or ESP-IDF.

## ESPHome

**Upstream:** `https://github.com/esphome/esphome`

ESPHome is the build framework and runtime used by this project.

The upstream ESPHome license states that:

- the C++/runtime codebase is GPLv3;
- the Python codebase and other non-C++ parts are MIT-licensed.

ESPHome is not vendored in this repository.

## JPEGDEC 1.8.4

**Upstream:** `https://github.com/bitbank2/JPEGDEC`

**License:** Apache License 2.0

**Copyright identified upstream:** BitBank Software, Inc.

**Vendored here:** no

The external component requests JPEGDEC as a build dependency.

## Espressif esp_audio_codec 2.6.2

**Upstream:** `https://github.com/espressif/esp-adf-libs`

**License:** Espressif Modified MIT License  
**SPDX identifier used upstream:** `LicenseRef-Espressif-Modified-MIT`

**Vendored here:** no

The upstream license permits use exclusively with Espressif Systems products and prohibits redistribution for use with non-Espressif products.

The target board, ESP32-S3-BOX-3, is an Espressif product.

## Espressif esp-tflite-micro 1.3.1

**Upstream:** `https://github.com/espressif/esp-tflite-micro`

**Component license:** Apache License 2.0

**Vendored here:** no

Espressif states that TensorFlow library code and third-party code contained in that component retain the licenses specified in their respective upstream sources.

## ESP-IDF

**Upstream:** `https://github.com/espressif/esp-idf`

ESP-IDF original source is generally Apache License 2.0 and also includes third-party material under additional licenses. Upstream file headers and the ESP-IDF copyright/license inventory take precedence.

ESP-IDF is not vendored in this repository.

## Cry ML model

**Artifact:** `chayuto/yamnet-cry-distill-int8`

**License declared upstream:** MIT

**Bundled in this repository:** no

The optional fetch tool downloads a pinned model revision and verifies its SHA-256. See `THIRD_PARTY_MODEL.md`.
