// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace esphome::baby_monitor {

class MotionDetector {
 public:
  static constexpr uint16_t GRID_WIDTH = 40;
  static constexpr uint16_t GRID_HEIGHT = 30;
  static constexpr size_t GRID_SIZE = static_cast<size_t>(GRID_WIDTH) * GRID_HEIGHT;

  void process_rgb565_be(const uint8_t *framebuffer, uint16_t width, uint16_t height, size_t stride_bytes);
  void reset();

  void set_luma_threshold(uint8_t value);
  void set_score_threshold(float value_percent);
  void set_confirm_ms(uint32_t value_ms);
  void set_scene_score_threshold(float value_percent);
  void set_scene_luma_delta_threshold(uint8_t value);
  void set_scene_settle_ms(uint32_t value_ms);

  float score() const { return score_percent_.load(); }
  bool detected() const { return detected_.load(); }
  int16_t global_luma_delta() const { return global_luma_delta_.load(); }
  bool initialized() const { return initialized_.load(); }
  bool scene_change_active() const { return scene_change_active_.load(); }
  uint32_t scene_change_count() const { return scene_change_count_.load(); }

 private:
  static uint8_t rgb565_be_luma_(const uint8_t *p);
  void reset_state_();

  std::array<uint8_t, GRID_SIZE> previous_{};
  std::array<uint8_t, GRID_SIZE> current_{};

  std::atomic<uint8_t> luma_threshold_{12};
  std::atomic<float> score_threshold_percent_{8.0f};
  std::atomic<uint32_t> confirm_ms_{400};
  std::atomic<float> scene_score_threshold_percent_{30.0f};
  std::atomic<uint8_t> scene_luma_delta_threshold_{5};
  std::atomic<uint32_t> scene_settle_ms_{1800};

  std::atomic<float> score_percent_{0.0f};
  std::atomic<bool> detected_{false};
  std::atomic<int16_t> global_luma_delta_{0};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> reset_requested_{false};
  std::atomic<bool> scene_change_active_{false};
  std::atomic<uint32_t> scene_change_count_{0};

  uint32_t candidate_since_ms_{0};
  uint32_t quiet_since_ms_{0};
  uint32_t scene_settle_until_ms_{0};
};

}  // namespace esphome::baby_monitor
