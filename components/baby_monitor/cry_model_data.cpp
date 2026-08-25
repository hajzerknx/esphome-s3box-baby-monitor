// SPDX-License-Identifier: Apache-2.0
//
// Project-owned wrapper for the optional third-party Cry ML model.
//
// The public repository does not contain the model bytes. Run
// tools/fetch_cry_model.py to generate cry_model_blob.inc locally.
// The generated file is intentionally ignored by Git.

#include "cry_model_data.h"

namespace esphome::baby_monitor {

#if __has_include("cry_model_blob.inc")
#include "cry_model_blob.inc"
#else
alignas(16) const uint8_t g_cry_model_data[] = {0x00};
const size_t g_cry_model_data_len = sizeof(g_cry_model_data);
const char g_cry_model_sha256[] = "";
#endif

}  // namespace esphome::baby_monitor
