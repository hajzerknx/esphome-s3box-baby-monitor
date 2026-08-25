# Third-party Cry ML model

## Direct artifact used by the firmware

**Model:** `chayuto/yamnet-cry-distill-int8`

**Upstream model page:** `https://huggingface.co/chayuto/yamnet-cry-distill-int8`

**Upstream source repository:** `https://github.com/chayuto/yamnet-cry-distill-int8`

**License declared upstream:** MIT

**Pinned model revision:**

```text
5cd4cd25ad3c178c0792efbcffc09eff73cd0064
```

**Expected model size:** 112,848 bytes

**Expected SHA-256:**

```text
cf7f879e2ae065f06ddc209830395bc1b448a5f819db44c7d40195095758f5ba
```

## How this repository handles the model

The model binary is not included in the public repository.

Run:

```bash
python3 tools/fetch_cry_model.py
```

The script downloads the pinned upstream artifact, verifies its size and SHA-256, and generates:

```text
components/baby_monitor/cry_model_blob.inc
```

That generated file is ignored by Git. The tracked `cry_model_data.cpp` file is only a project-owned wrapper and fallback stub.

## License boundary

The downloaded model remains a third-party MIT-licensed artifact. This project's Apache-2.0 license does not relicense the model.

If you redistribute the model itself, the generated blob, or a firmware image that embeds the model, review and comply with the upstream MIT license and attribution requirements.

For architecture, training methodology, dataset provenance, evaluation and limitations, refer to the upstream model card. Those upstream materials are not dependencies or content redistributed by this repository.
