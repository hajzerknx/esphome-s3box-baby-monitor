// SPDX-License-Identifier: Apache-2.0
#include "motion_detector.h"

#include <algorithm>
#include <cmath>

#include "esphome/core/hal.h"

namespace esphome::baby_monitor {

namespace {
constexpr uint32_t RELEASE_MS = 600;
constexpr float RELEASE_RATIO = 0.60f;
}

uint8_t MotionDetector::rgb565_be_luma_(const uint8_t *p) {
  const uint16_t px = (static_cast<uint16_t>(p[0]) << 8) | p[1];
  const uint8_t r5 = static_cast<uint8_t>((px >> 11) & 0x1F);
  const uint8_t g6 = static_cast<uint8_t>((px >> 5) & 0x3F);
  const uint8_t b5 = static_cast<uint8_t>(px & 0x1F);

  const uint16_t r = static_cast<uint16_t>((r5 << 3) | (r5 >> 2));
  const uint16_t g = static_cast<uint16_t>((g6 << 2) | (g6 >> 4));
  const uint16_t b = static_cast<uint16_t>((b5 << 3) | (b5 >> 2));
  return static_cast<uint8_t>((77u * r + 150u * g + 29u * b) >> 8);
}

void MotionDetector::set_luma_threshold(uint8_t value) {
  luma_threshold_.store(std::max<uint8_t>(2, std::min<uint8_t>(80, value)));
}

void MotionDetector::set_score_threshold(float value_percent) {
  if (!std::isfinite(value_percent)) return;
  score_threshold_percent_.store(std::max(0.5f, std::min(80.0f, value_percent)));
}

void MotionDetector::set_confirm_ms(uint32_t value_ms) {
  confirm_ms_.store(std::max<uint32_t>(100, std::min<uint32_t>(5000, value_ms)));
}

void MotionDetector::set_scene_score_threshold(float value_percent) {
  if (!std::isfinite(value_percent)) return;
  scene_score_threshold_percent_.store(std::max(10.0f, std::min(95.0f, value_percent)));
}

void MotionDetector::set_scene_luma_delta_threshold(uint8_t value) {
  scene_luma_delta_threshold_.store(std::max<uint8_t>(1, std::min<uint8_t>(80, value)));
}

void MotionDetector::set_scene_settle_ms(uint32_t value_ms) {
  scene_settle_ms_.store(std::max<uint32_t>(200, std::min<uint32_t>(5000, value_ms)));
}

void MotionDetector::reset() {
  reset_requested_.store(true);
}

void MotionDetector::reset_state_() {
  previous_.fill(0);
  current_.fill(0);
  score_percent_.store(0.0f);
  detected_.store(false);
  global_luma_delta_.store(0);
  initialized_.store(false);
  scene_change_active_.store(false);
  candidate_since_ms_ = 0;
  quiet_since_ms_ = 0;
  scene_settle_until_ms_ = 0;
}

void MotionDetector::process_rgb565_be(const uint8_t *framebuffer, uint16_t width, uint16_t height,
                                       size_t stride_bytes) {
  if (reset_requested_.exchange(false)) reset_state_();

  if (framebuffer == nullptr || width < GRID_WIDTH || height < GRID_HEIGHT ||
      stride_bytes < static_cast<size_t>(width) * 2u) {
    return;
  }

  const uint16_t cell_w = width / GRID_WIDTH;
  const uint16_t cell_h = height / GRID_HEIGHT;
  if (cell_w == 0 || cell_h == 0) return;

  size_t n = 0;
  for (uint16_t gy = 0; gy < GRID_HEIGHT; gy++) {
    const uint16_t y0 = static_cast<uint16_t>(gy * cell_h);
    const uint16_t ya = std::min<uint16_t>(height - 1, static_cast<uint16_t>(y0 + cell_h / 3u));
    const uint16_t yb = std::min<uint16_t>(height - 1, static_cast<uint16_t>(y0 + (cell_h * 2u) / 3u));

    for (uint16_t gx = 0; gx < GRID_WIDTH; gx++, n++) {
      const uint16_t x0 = static_cast<uint16_t>(gx * cell_w);
      const uint16_t xa = std::min<uint16_t>(width - 1, static_cast<uint16_t>(x0 + cell_w / 3u));
      const uint16_t xb = std::min<uint16_t>(width - 1, static_cast<uint16_t>(x0 + (cell_w * 2u) / 3u));

      const uint8_t *p00 = framebuffer + static_cast<size_t>(ya) * stride_bytes + static_cast<size_t>(xa) * 2u;
      const uint8_t *p01 = framebuffer + static_cast<size_t>(ya) * stride_bytes + static_cast<size_t>(xb) * 2u;
      const uint8_t *p10 = framebuffer + static_cast<size_t>(yb) * stride_bytes + static_cast<size_t>(xa) * 2u;
      const uint8_t *p11 = framebuffer + static_cast<size_t>(yb) * stride_bytes + static_cast<size_t>(xb) * 2u;

      const uint16_t sum = static_cast<uint16_t>(rgb565_be_luma_(p00)) +
                           static_cast<uint16_t>(rgb565_be_luma_(p01)) +
                           static_cast<uint16_t>(rgb565_be_luma_(p10)) +
                           static_cast<uint16_t>(rgb565_be_luma_(p11));
      current_[n] = static_cast<uint8_t>((sum + 2u) / 4u);
    }
  }

  if (!initialized_.load()) {
    previous_ = current_;
    initialized_.store(true);
    score_percent_.store(0.0f);
    detected_.store(false);
    candidate_since_ms_ = 0;
    quiet_since_ms_ = 0;
    return;
  }

  int32_t signed_sum = 0;
  for (size_t i = 0; i < GRID_SIZE; i++) {
    signed_sum += static_cast<int16_t>(current_[i]) - static_cast<int16_t>(previous_[i]);
  }

  const int16_t global_delta = static_cast<int16_t>(
      signed_sum >= 0
          ? (signed_sum + static_cast<int32_t>(GRID_SIZE / 2)) / static_cast<int32_t>(GRID_SIZE)
          : (signed_sum - static_cast<int32_t>(GRID_SIZE / 2)) / static_cast<int32_t>(GRID_SIZE));
  global_luma_delta_.store(global_delta);

  const int16_t threshold = static_cast<int16_t>(luma_threshold_.load());
  uint16_t changed = 0;
  for (size_t i = 0; i < GRID_SIZE; i++) {
    const int16_t delta = (static_cast<int16_t>(current_[i]) - static_cast<int16_t>(previous_[i])) - global_delta;
    if (std::abs(delta) >= threshold) changed++;
  }

  previous_ = current_;

  const float score = 100.0f * static_cast<float>(changed) / static_cast<float>(GRID_SIZE);
  score_percent_.store(score);

  const uint32_t now = millis();

  // v1.3.0: suppress global scene changes such as IR/day-night/light/exposure.
  const bool scene_change =
      score >= scene_score_threshold_percent_.load() &&
      std::abs(global_delta) >= static_cast<int16_t>(scene_luma_delta_threshold_.load());

  if (scene_change) {
    scene_change_active_.store(true);
    scene_change_count_.fetch_add(1);
    scene_settle_until_ms_ = now + scene_settle_ms_.load();
    detected_.store(false);
    candidate_since_ms_ = 0;
    quiet_since_ms_ = 0;
    return;
  }

  if (scene_settle_until_ms_ != 0 &&
      static_cast<int32_t>(scene_settle_until_ms_ - now) > 0) {
    scene_change_active_.store(true);
    detected_.store(false);
    candidate_since_ms_ = 0;
    quiet_since_ms_ = 0;
    return;
  }

  scene_settle_until_ms_ = 0;
  scene_change_active_.store(false);

  const float trigger = score_threshold_percent_.load();
  const float release = trigger * RELEASE_RATIO;

  if (!detected_.load()) {
    quiet_since_ms_ = 0;
    if (score >= trigger) {
      if (candidate_since_ms_ == 0) candidate_since_ms_ = now;
      if (static_cast<uint32_t>(now - candidate_since_ms_) >= confirm_ms_.load()) {
        detected_.store(true);
        candidate_since_ms_ = 0;
      }
    } else {
      candidate_since_ms_ = 0;
    }
  } else {
    candidate_since_ms_ = 0;
    if (score < release) {
      if (quiet_since_ms_ == 0) quiet_since_ms_ = now;
      if (static_cast<uint32_t>(now - quiet_since_ms_) >= RELEASE_MS) {
        detected_.store(false);
        quiet_since_ms_ = 0;
      }
    } else {
      quiet_since_ms_ = 0;
    }
  }
}

}  // namespace esphome::baby_monitor
