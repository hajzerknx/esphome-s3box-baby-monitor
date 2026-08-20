// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "esphome/components/display/display.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/core/component.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <JPEGDEC.h>
#include "decoder/impl/esp_aac_dec.h"

namespace esphome::baby_monitor {

struct CameraConfig {
  std::string name, host, username, password, path;
  uint16_t port{554};
};

struct RtspAuth {
  enum class Type : uint8_t { NONE, BASIC, DIGEST };
  Type type{Type::NONE};
  std::string realm, nonce, opaque, qop, algorithm;
  uint32_t nonce_count{0};
};

struct RtspResponse {
  int status{0};
  std::string status_line, headers, body;
  std::string header(const char *name) const;
};

struct SdpTrack {
  bool valid{false};
  std::string media, control, rtpmap, fmtp;
  int payload_type{-1};
};

struct VideoFrame {
  uint32_t epoch{0};
  uint32_t length{0};
  uint8_t slot{0};
};

struct AudioAu {
  static constexpr size_t MAX_BYTES = 2048;
  uint32_t epoch{0};
  uint16_t length{0};
  std::array<uint8_t, MAX_BYTES> data{};
};

struct JpegFrameState {
  bool active{false};
  bool valid{false};
  uint32_t timestamp{0};
  uint16_t expected_seq{0};
  uint8_t type{0};
  uint8_t q{0};
  uint8_t width_blocks{0};
  uint8_t height_blocks{0};
  uint16_t dri{0};
  std::array<uint8_t, 64> lqt{};
  std::array<uint8_t, 64> cqt{};
};

class BabyMonitor : public Component {
 public:
  void setup() override;
  void loop() override {}
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  void set_display(display::Display *value) { display_ = value; }
  void set_speaker(speaker::Speaker *value) { speaker_ = value; }
  void set_audio_volume(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    audio_volume_.store(value);
  }
  void set_audio_enabled(bool enabled);
  bool is_audio_enabled() const { return audio_enabled_.load(); }
  float get_audio_volume() const { return audio_volume_.load(); }
  void set_ui_battery_percent(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 100.0f) value = 100.0f;
    ui_battery_percent_.store(static_cast<int16_t>(value + 0.5f));
  }
  void set_camera_name(uint8_t i, const std::string &v);
  void set_camera_host(uint8_t i, const std::string &v);
  void set_camera_port(uint8_t i, uint16_t v);
  void set_camera_username(uint8_t i, const std::string &v);
  void set_camera_password(uint8_t i, const std::string &v);
  void set_camera_path(uint8_t i, const std::string &v);

  void set_enabled(bool enabled);
  void set_camera(uint8_t index);
  void set_alarm_level(uint8_t level) { ui_alarm_level_.store(level > 2 ? 2 : level); }
  uint8_t get_alarm_level() const { return ui_alarm_level_.load(); }
  float get_free_heap_kb() const;
  float get_video_fps() const { return diagnostics_video_fps_x100_.load() / 100.0f; }
  float get_video_bitrate_kbps() const { return diagnostics_video_bitrate_x10_.load() / 10.0f; }
  float get_audio_bitrate_kbps() const { return diagnostics_audio_bitrate_x10_.load() / 10.0f; }
  uint32_t get_video_drops() const { return diagnostics_video_drops_.load(); }
  uint32_t get_video_replaced() const { return diagnostics_video_replaced_.load(); }
  uint32_t get_audio_drops() const { return diagnostics_audio_drops_.load(); }
  bool is_rtsp_connected() const { return rtsp_connected_.load(); }

  static std::string trim_(const std::string &v);
  static std::string lowercase_(std::string v);

 protected:
  static void control_task_trampoline_(void *arg);
  static void rtsp_task_trampoline_(void *arg);
  static void audio_task_trampoline_(void *arg);
  static void video_task_trampoline_(void *arg);
  void control_task_();
  void rtsp_task_();
  void audio_task_();
  void video_task_();
  void start_session_();
  void stop_session_();
  void close_socket_();
  void request_audio_reset_();
  void request_video_reset_();
  void release_media_resources_();
  void reset_diagnostics_();

  bool connect_rtsp_(const CameraConfig &cam);
  bool run_rtsp_handshake_(const CameraConfig &cam);
  void receive_interleaved_loop_();
  bool send_rtsp_(const CameraConfig &cam, const std::string &method, const std::string &uri,
                  const std::string &extra_headers, RtspResponse &response, bool allow_auth_retry = true);
  bool send_all_(const char *data, size_t length);
  bool read_rtsp_response_(RtspResponse &response);
  bool read_exact_(uint8_t *dst, size_t length);

  bool parse_auth_challenge_(const std::string &header);
  std::string build_authorization_(const CameraConfig &cam, const std::string &method, const std::string &uri);
  std::string md5_hex_(const std::string &input) const;
  std::string basic_authorization_(const CameraConfig &cam) const;
  bool parse_sdp_(const std::string &sdp, SdpTrack &video, SdpTrack &audio, std::string &session_control);
  std::string resolve_control_url_(const std::string &base, const std::string &control) const;
  std::string request_url_(const CameraConfig &cam) const;
  static std::string find_param_(const std::string &source, const char *name);

  void handle_video_rtp_(const uint8_t *packet, size_t length);
  void handle_audio_rtp_(const uint8_t *packet, size_t length);
  bool enqueue_aac_au_(const uint8_t *data, size_t length);
  bool decode_aac_au_(const uint8_t *data, size_t length);
  bool ensure_aac_decoder_();
  void close_aac_decoder_();
  void configure_aac_from_sdp_(const SdpTrack &audio);
  static int parse_fmtp_int_(const std::string &fmtp, const char *name, int fallback);
  bool parse_rtp_header_(const uint8_t *packet, size_t length, size_t &payload_offset, size_t &payload_length,
                         uint16_t &seq, uint32_t &timestamp, bool &marker) const;
  bool begin_jpeg_frame_(const uint8_t *payload, size_t length, uint32_t timestamp, uint16_t seq,
                         size_t &scan_offset, uint32_t &fragment_offset);
  void finish_jpeg_frame_();
  bool ensure_jpeg_buffers_();
  void free_jpeg_buffers_();
  bool build_interchange_jpeg_(uint8_t *out, size_t capacity, size_t &jpeg_length);
  bool decode_and_display_(const uint8_t *jpeg, size_t jpeg_length);

  // Lightweight framebuffer UI (no LVGL).
  void render_ui_overlay_();
  void ui_draw_mute_icon_();
  void update_ui_wifi_();
  void ui_set_pixel_(int x, int y, uint16_t rgb565);
  uint16_t ui_get_pixel_(int x, int y) const;
  static uint16_t ui_blend565_(uint16_t dst, uint16_t src, uint8_t alpha);
  void ui_blend_rect_(int x, int y, int w, int h, uint16_t rgb565, uint8_t alpha);
  void ui_blend_round_rect_(int x, int y, int w, int h, int radius, uint16_t rgb565, uint8_t alpha);
  void ui_fill_rect_(int x, int y, int w, int h, uint16_t rgb565);
  void ui_draw_rect_(int x, int y, int w, int h, uint16_t rgb565);
  void ui_draw_text_(int x, int y, const char *text, uint16_t fg, uint16_t bg, uint8_t scale = 1);
  void ui_draw_text_fg_(int x, int y, const char *text, uint16_t fg, uint8_t scale = 1);
  void render_no_stream_placeholder_();
  void ui_draw_camera_name_();
  void ui_draw_wifi_();
  void ui_draw_battery_placeholder_();
  void ui_draw_media_status_();
  void ui_draw_alarm_border_(uint8_t level);
  static uint16_t ui_glyph3x5_(char c);
  static int jpeg_draw_callback_(JPEGDRAW *draw);
  int draw_jpeg_block_(JPEGDRAW *draw);

  static void make_quant_tables_(int q, uint8_t *lqt, uint8_t *cqt);
  static bool append_byte_(uint8_t *out, size_t capacity, size_t &pos, uint8_t value);
  static bool append_bytes_(uint8_t *out, size_t capacity, size_t &pos, const uint8_t *data, size_t length);
  static bool append_quant_header_(uint8_t *out, size_t capacity, size_t &pos, const uint8_t *qt, uint8_t table_no);
  static bool append_huffman_header_(uint8_t *out, size_t capacity, size_t &pos, const uint8_t *lens, size_t lens_len,
                                     const uint8_t *symbols, size_t symbols_len, uint8_t table_no, uint8_t table_class);
  static bool append_jpeg_headers_(uint8_t *out, size_t capacity, size_t &pos, uint8_t type, uint8_t w_blocks,
                                   uint8_t h_blocks, const uint8_t *lqt, const uint8_t *cqt, uint16_t dri);

  CameraConfig cameras_[2];
  display::Display *display_{nullptr};
  speaker::Speaker *speaker_{nullptr};
  std::atomic<float> audio_volume_{0.70f};
  std::atomic<bool> audio_enabled_{true};
  std::atomic<bool> enabled_{false}, stop_requested_{false}, task_running_{false}, audio_task_running_{false}, video_task_running_{false};
  std::atomic<bool> audio_reset_requested_{false}, video_reset_requested_{false};
  std::atomic<bool> placeholder_requested_{false};
  std::atomic<uint32_t> placeholder_due_ms_{0};
  std::atomic<uint32_t> media_epoch_{0};
  std::atomic<uint8_t> requested_camera_{0};
  // Last completed five-second diagnostic window; read-only from Home Assistant.
  std::atomic<bool> rtsp_connected_{false};
  std::atomic<uint32_t> diagnostics_video_fps_x100_{0};
  std::atomic<uint32_t> diagnostics_video_bitrate_x10_{0};
  std::atomic<uint32_t> diagnostics_audio_bitrate_x10_{0};
  std::atomic<uint32_t> diagnostics_video_drops_{0};
  std::atomic<uint32_t> diagnostics_video_replaced_{0};
  std::atomic<uint32_t> diagnostics_audio_drops_{0};
  uint8_t active_camera_{0};
  TaskHandle_t control_task_handle_{nullptr};
  TaskHandle_t rtsp_task_handle_{nullptr};
  TaskHandle_t audio_task_handle_{nullptr};
  TaskHandle_t video_task_handle_{nullptr};
  QueueHandle_t audio_queue_{nullptr};
  QueueHandle_t video_queue_{nullptr};
  SemaphoreHandle_t video_queue_mutex_{nullptr};
  std::atomic<int> video_busy_slot_{-1};
  int socket_{-1};
  uint32_t cseq_{1};
  std::string session_id_, content_base_;
  RtspAuth auth_;
  int video_rtp_channel_{-1}, audio_rtp_channel_{-1};
  uint64_t video_bytes_{0}, audio_bytes_{0};
  uint32_t video_packets_{0}, audio_packets_{0}, last_stats_ms_{0};
  std::atomic<uint32_t> decoded_frames_{0}, dropped_frames_{0}, video_replaced_frames_{0};
  uint32_t audio_aus_{0};
  std::atomic<uint32_t> audio_decoded_aus_{0}, audio_dropped_aus_{0};
  std::atomic<uint64_t> video_decode_us_total_{0};
  // v0.6.2 diagnostics (unchanged from stable v0.6.1): total frame time, pure JPEGDEC time and the single
  // full-screen LCD flush time.
  std::atomic<uint64_t> video_lcd_us_total_{0}, video_other_us_total_{0};
  std::atomic<uint32_t> video_lcd_us_max_{0}, video_other_us_max_{0};
  std::atomic<uint32_t> video_lcd_blocks_total_{0};
  // Accessed only by baby_video / JPEGDEC callback for the current frame.
  // In v0.6.2 blocks count JPEG decoder callbacks; no LCD I/O happens there.
  uint64_t current_frame_lcd_us_{0};
  uint32_t current_frame_lcd_blocks_{0};
  std::atomic<uint64_t> audio_decode_us_total_{0};
  std::atomic<uint32_t> video_decode_count_{0}, video_decode_us_max_{0};
  std::atomic<uint32_t> audio_decode_count_{0}, audio_decode_us_max_{0};
  uint32_t session_media_packets_{0};
  int aac_size_length_{13}, aac_index_length_{3}, aac_index_delta_length_{3};
  bool aac_rtp_supported_{false};
  void *aac_decoder_{nullptr};
  std::vector<uint8_t> aac_pcm_;
  JpegFrameState jpeg_state_;
  uint8_t *jpeg_scan_buffer_{nullptr};
  std::array<uint8_t *, 2> jpeg_frame_buffers_{{nullptr, nullptr}};
  size_t jpeg_scan_length_{0};
  size_t jpeg_scan_peak_{0};
  size_t jpeg_interchange_peak_{0};
  // Allocate JPEGDEC lazily on the heap on the first real video frame.
  // This keeps its large working state both off the RTSP task stack and out of
  // the static BabyMonitor object constructed during early boot.
  JPEGDEC *jpeg_decoder_{nullptr};
  uint8_t *video_framebuffer_{nullptr};

  // UI state.
  int8_t ui_wifi_rssi_{-127};
  uint32_t ui_last_wifi_ms_{0};
  std::atomic<uint8_t> ui_alarm_level_{0};
  std::atomic<int16_t> ui_battery_percent_{-1};
};

}  // namespace esphome::baby_monitor
