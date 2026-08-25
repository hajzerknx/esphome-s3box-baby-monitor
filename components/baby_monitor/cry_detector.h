// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

namespace esphome::baby_monitor {

class CryDetector {
 public:
  static constexpr uint32_t SAMPLE_RATE = 16000;
  static constexpr size_t PATCH_SAMPLES = 15600;
  static constexpr size_t HOP_SAMPLES = 7680;
  static constexpr size_t MEL_FRAMES = 96;
  static constexpr size_t MEL_BINS = 64;
  static constexpr size_t FFT_SIZE = 512;
  static constexpr size_t WINDOW_SAMPLES = 400;
  static constexpr size_t FRAME_HOP = 160;

  void feed(const int16_t *samples, size_t sample_count, uint32_t sample_rate_hz);
  void reset();
  void set_candidate_threshold(float value);

  // v1.4.0-dev42: run the one-shot TFLM initialization synchronously before
  // media worker tasks start. Failure remains non-fatal for RTSP/audio/video.
  bool prepare_runtime() { return ensure_runtime_(); }

  bool model_ready() const { return model_ready_.load(); }
  bool candidate() const { return candidate_.load(); }
  float cry_score() const { return cry_score_.load(); }
  float baby_cry_score() const { return baby_cry_score_.load(); }
  float crying_score() const { return crying_score_.load(); }
  float speech_score() const { return speech_score_.load(); }
  float yell_scream_score() const { return yell_scream_score_.load(); }
  float babbling_score() const { return babbling_score_.load(); }
  float candidate_threshold() const { return candidate_threshold_.load(); }
  uint32_t inference_ms() const { return inference_ms_.load(); }
  uint32_t inference_count() const { return inference_count_.load(); }
  // v1.4.1-dev4: read-only scheduler correlation marker for LCD diagnostics.
  bool inference_active() const { return inference_active_.load(); }
  uint32_t dropped_windows() const { return dropped_windows_.load(); }
  uint32_t model_bytes() const { return model_bytes_.load(); }
  const char *status() const;

 private:
  enum class Status : uint8_t {
    UNINITIALIZED = 0,
    MODEL_NOT_EMBEDDED,
    ALLOC_FAILED,
    MODEL_INVALID,
    TENSOR_ALLOC_FAILED,
    TENSOR_SHAPE_MISMATCH,
    READY,
    INVOKE_FAILED,
    BAD_SAMPLE_RATE,
  };

  struct MlState;

  bool ensure_runtime_();
  bool ensure_buffers_();
  void append_samples_(const int16_t *samples, size_t count);
  bool snapshot_latest_patch_();
  bool build_logmel_and_quantize_();
  void run_inference_();
  static void task_entry_(void *arg);
  void task_loop_();

  void init_frontend_();
  void fft512_(float *re, float *im) const;
  float mel_weight_(size_t fft_bin, size_t mel_bin) const;

  std::atomic<Status> status_{Status::UNINITIALIZED};
  std::atomic<bool> model_ready_{false};
  std::atomic<bool> candidate_{false};
  std::atomic<float> candidate_threshold_{0.05f};
  std::atomic<float> cry_score_{0.0f};
  std::atomic<float> baby_cry_score_{0.0f};
  std::atomic<float> crying_score_{0.0f};
  std::atomic<float> speech_score_{0.0f};
  std::atomic<float> yell_scream_score_{0.0f};
  std::atomic<float> babbling_score_{0.0f};
  std::atomic<uint32_t> inference_ms_{0};
  std::atomic<uint32_t> inference_count_{0};
  std::atomic<bool> inference_active_{false};
  std::atomic<uint32_t> dropped_windows_{0};
  std::atomic<uint32_t> model_bytes_{0};

  int16_t *ring_{nullptr};
  int16_t *snapshot_{nullptr};
  size_t ring_write_{0};
  size_t ring_filled_{0};
  size_t samples_since_window_{0};

  float *fft_re_{nullptr};
  float *fft_im_{nullptr};
  float *hann_{nullptr};
  float mel_edges_hz_[MEL_BINS + 2]{};

  // dev42: sparse precomputed mel filterbank. For a triangular mel bank each
  // FFT bin can contribute to at most two adjacent filters. The map is built
  // once from mel_weight_(), preserving the exact dev39 filter mathematics.
  static constexpr size_t FFT_MAG_BINS = FFT_SIZE / 2 + 1;
  uint8_t mel_filter_a_[FFT_MAG_BINS]{};
  uint8_t mel_filter_b_[FFT_MAG_BINS]{};
  float mel_weight_a_[FFT_MAG_BINS]{};
  float mel_weight_b_[FFT_MAG_BINS]{};
  uint8_t mel_active_count_[FFT_MAG_BINS]{};
  bool mel_sparse_valid_{false};
  uint16_t mel_sparse_active_weights_{0};

  MlState *ml_{nullptr};
  SemaphoreHandle_t mutex_{nullptr};
  TaskHandle_t task_{nullptr};
  std::atomic<bool> window_pending_{false};
  std::atomic<bool> reset_requested_{false};
  // Runtime initialization is deliberately one-shot. A failed TFLM init
  // must never allocate a new tensor arena for every incoming audio window.
  std::atomic<bool> runtime_init_attempted_{false};
};

}  // namespace esphome::baby_monitor
