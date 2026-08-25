#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Fetches the optional third-party Cry ML model from its upstream publisher.
# The downloaded model is not relicensed by this project.

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

MODEL_PAGE = "https://huggingface.co/chayuto/yamnet-cry-distill-int8"
MODEL_SOURCE_REPO = "https://github.com/chayuto/yamnet-cry-distill-int8"

# Pin the model artifact by revision and verify it independently by SHA-256.
MODEL_REVISION = "5cd4cd25ad3c178c0792efbcffc09eff73cd0064"
MODEL_URL = (
    "https://huggingface.co/chayuto/yamnet-cry-distill-int8/"
    f"resolve/{MODEL_REVISION}/model.tflite?download=true"
)
MODEL_SHA256 = "cf7f879e2ae065f06ddc209830395bc1b448a5f819db44c7d40195095758f5ba"
MODEL_SIZE = 112_848

root = Path(__file__).resolve().parents[1]
component_dir = root / "components" / "baby_monitor"
cache_dir = root / ".model-cache"
raw_model = cache_dir / "yamnet-cry-distill-int8.tflite"
blob_out = component_dir / "cry_model_blob.inc"

print("Downloading the upstream Cry ML model...")
print(f"Source: {MODEL_PAGE}")
print("Upstream declares the model under the MIT License.")

with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
    data = response.read()

digest = hashlib.sha256(data).hexdigest()
if digest != MODEL_SHA256:
    raise SystemExit(f"SHA-256 mismatch: {digest} != {MODEL_SHA256}")
if len(data) != MODEL_SIZE:
    raise SystemExit(f"Unexpected model size: {len(data)} != {MODEL_SIZE}")

cache_dir.mkdir(exist_ok=True)
raw_model.write_bytes(data)

rows = []
for offset in range(0, len(data), 16):
    chunk = data[offset:offset + 16]
    rows.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

# This generated include contains the upstream model bytes. It is not tracked by
# this repository and remains subject to the upstream model's MIT License.
content = (
    "// Generated third-party model data. DO NOT COMMIT.\n"
    "// Upstream: chayuto/yamnet-cry-distill-int8\n"
    "// License declared upstream: MIT\n"
    f"// Model revision: {MODEL_REVISION}\n"
    f"// SHA-256: {MODEL_SHA256}\n\n"
    "alignas(16) const uint8_t g_cry_model_data[] = {\n"
    + "\n".join(rows)
    + "\n};\n"
      "const size_t g_cry_model_data_len = sizeof(g_cry_model_data);\n"
      f'const char g_cry_model_sha256[] = "{MODEL_SHA256}";\n'
)
blob_out.write_text(content, encoding="utf-8")

print(f"Verified model size: {len(data)} bytes")
print(f"Verified SHA-256: {digest}")
print(f"Generated local blob: {blob_out}")
print()
print("The generated blob is ignored by Git.")
print("If you redistribute a firmware image containing the model, review and")
print("comply with the upstream MIT license and attribution requirements:")
print(MODEL_SOURCE_REPO)
