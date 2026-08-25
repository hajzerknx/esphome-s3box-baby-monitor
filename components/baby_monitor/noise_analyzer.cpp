// SPDX-License-Identifier: Apache-2.0
#include "noise_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace esphome::baby_monitor {

namespace {
inline float to_dbfs(float normalized) {
  if (!(normalized > 0.0f)) return -96.0f;
  const float value = 20.0f * log10f(normalized);
  return value < -96.0f ? -96.0f : (value > 0.0f ? 0.0f : value);
}
}  // namespace

void NoiseAnalyzer::reset() {
  rms_.store(0.0f);
  rms_dbfs_.store(DBFS_FLOOR);
  peak_dbfs_.store(DBFS_FLOOR);
  smoothed_dbfs_.store(DBFS_FLOOR);
  active_.store(false);
  sample_rate_hz_.store(0);
  smoothed_rms_ = 0.0f;
  above_ms_ = 0;
  below_ms_ = 0;
}

void NoiseAnalyzer::process(const int16_t *samples, size_t sample_count, uint32_t sample_rate_hz) {
  if (samples == nullptr || sample_count == 0 || sample_rate_hz == 0) return;

  uint64_t sum_sq = 0;
  uint32_t peak = 0;
  for (size_t i = 0; i < sample_count; i++) {
    const int32_t s = samples[i];
    const uint32_t mag = static_cast<uint32_t>(s < 0 ? -s : s);  // safe for -32768 in int32
    peak = std::max(peak, mag);
    sum_sq += static_cast<uint64_t>(s * s);
  }

  constexpr float full_scale = 32768.0f;
  const float mean_sq = static_cast<float>(sum_sq) / static_cast<float>(sample_count);
  const float block_rms = sqrtf(mean_sq) / full_scale;
  const float block_peak = static_cast<float>(peak) / full_scale;
  const float block_dbfs = to_dbfs(block_rms);

  // Use real block duration, so behavior does not depend on AAC output block size.
  const float block_ms_f = (1000.0f * static_cast<float>(sample_count)) / static_cast<float>(sample_rate_hz);
  uint32_t block_ms = static_cast<uint32_t>(block_ms_f + 0.5f);
  if (block_ms == 0) block_ms = 1;

  const float tau_ms = block_rms > smoothed_rms_ ? ATTACK_TAU_MS : RELEASE_TAU_MS;
  const float alpha = 1.0f - expf(-block_ms_f / tau_ms);
  if (smoothed_rms_ <= 0.0f)
    smoothed_rms_ = block_rms;
  else
    smoothed_rms_ += alpha * (block_rms - smoothed_rms_);

  const float smooth_dbfs = to_dbfs(smoothed_rms_);

  rms_.store(block_rms);
  rms_dbfs_.store(block_dbfs);
  peak_dbfs_.store(to_dbfs(block_peak));
  smoothed_dbfs_.store(smooth_dbfs);
  sample_rate_hz_.store(sample_rate_hz);

  // Hysteresis + time confirmation. Saturating counters avoid millis() rollover
  // concerns and keep the analyzer independent from ESPHome/Arduino timing APIs.
  if (!active_.load()) {
    below_ms_ = 0;
    if (smooth_dbfs >= active_threshold_dbfs_) {
      above_ms_ = std::min<uint32_t>(active_confirm_ms_, above_ms_ + block_ms);
      if (above_ms_ >= active_confirm_ms_) {
        active_.store(true);
        above_ms_ = 0;
      }
    } else {
      above_ms_ = 0;
    }
  } else {
    above_ms_ = 0;
    if (smooth_dbfs <= inactive_threshold_dbfs_) {
      below_ms_ = std::min<uint32_t>(inactive_confirm_ms_, below_ms_ + block_ms);
      if (below_ms_ >= inactive_confirm_ms_) {
        active_.store(false);
        below_ms_ = 0;
      }
    } else {
      below_ms_ = 0;
    }
  }
}

}  // namespace esphome::baby_monitor
