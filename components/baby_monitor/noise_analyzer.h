// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace esphome::baby_monitor {

// v1.2.0: lightweight zero-allocation PCM16 analyzer.
// One writer (baby_audio task), lock-free readers (ESPHome main loop).
class NoiseAnalyzer {
 public:
  void process(const int16_t *samples, size_t sample_count, uint32_t sample_rate_hz);
  void reset();

  // Instantaneous block metrics. RMS is normalized to full-scale [0..1].
  float rms() const { return rms_.load(); }
  float rms_dbfs() const { return rms_dbfs_.load(); }
  float peak_dbfs() const { return peak_dbfs_.load(); }

  // Attack/release smoothed RMS level in dBFS, intended for thresholds/UI.
  float dbfs() const { return smoothed_dbfs_.load(); }
  bool active() const { return active_.load(); }
  uint32_t sample_rate_hz() const { return sample_rate_hz_.load(); }

  // Generic sound-activity detector. The alarm engine may use its own adaptive
  // T1/T2 thresholds; this state is an independent local diagnostic primitive.
  void set_active_threshold_dbfs(float value) { active_threshold_dbfs_ = value; }
  void set_inactive_threshold_dbfs(float value) { inactive_threshold_dbfs_ = value; }
  void set_active_confirm_ms(uint32_t value) { active_confirm_ms_ = value; }
  void set_inactive_confirm_ms(uint32_t value) { inactive_confirm_ms_ = value; }

 private:
  static constexpr float DBFS_FLOOR = -96.0f;
  static constexpr float ATTACK_TAU_MS = 120.0f;
  static constexpr float RELEASE_TAU_MS = 700.0f;

  std::atomic<float> rms_{0.0f};
  std::atomic<float> rms_dbfs_{DBFS_FLOOR};
  std::atomic<float> peak_dbfs_{DBFS_FLOOR};
  std::atomic<float> smoothed_dbfs_{DBFS_FLOOR};
  std::atomic<bool> active_{false};
  std::atomic<uint32_t> sample_rate_hz_{0};

  // Written only from baby_audio task.
  float smoothed_rms_{0.0f};
  float active_threshold_dbfs_{-50.0f};
  float inactive_threshold_dbfs_{-55.0f};
  uint32_t active_confirm_ms_{300};
  uint32_t inactive_confirm_ms_{1200};
  uint32_t above_ms_{0};
  uint32_t below_ms_{0};
};

}  // namespace esphome::baby_monitor
