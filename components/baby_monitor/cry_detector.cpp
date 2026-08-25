// SPDX-License-Identifier: Apache-2.0
#include "cry_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include "cry_model_data.h"
#include "esp_heap_caps.h"
#include "esp_attr.h"
#include "esp_timer.h"
#include "esphome/core/log.h"

// v1.4.0-dev46: scheduler stabilization build. Keep the proven dev42-dev44a
// runtime, allocator, sparse MEL, fixed FFT512 plan and custom INT8 MEAN/FC
// unchanged. The Cry ML task remains at base priority P1 on core 0; only
// MicroInterpreter::Invoke() is temporarily boosted to P5, then restored to P1
// immediately after Invoke returns. dev44a demonstrated that P3/P4 still allow
// substantial RTSP-driven jitter while P5 makes Invoke nearly deterministic.
// The dev43/dev44 per-node Invoke profiler is removed from the hot path; only
// aggregate frontend/Invoke/post/total timing is retained every five successful
// inferences. RTC journal, guarded 1 MiB arena, exact resolver and failed-runtime
// quarantine remain unchanged.
// TFLite Micro is mandatory. Keep the custom INT8 MEAN and
// per-channel INT8 FULLY_CONNECTED kernels. dev36 fixed Prepare-time temp
// TfLiteTensor lifetime and made AllocateTensors succeed; dev43 removes the same
// class of allocation from Invoke by using TfLiteEvalTensor only and caching
// quantization metadata during Prepare. Runtime ResetTemp logging is throttled.
// RTC journal, guarded 1 MiB arena, exact resolver and failed-runtime quarantine
// remain unchanged.
// Do not silently compile a "no TFLM" fallback. If one of these headers is
// unavailable in the managed Espressif component, the build must fail loudly.
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"



namespace esphome::baby_monitor {

static const char *const TAG = "cry_detector";
static constexpr float PI_F = 3.14159265358979323846f;
static constexpr size_t TENSOR_ARENA_BYTES = 1024 * 1024;
static constexpr size_t DEV43_ARENA_GUARD_BYTES = 512;
static constexpr uint8_t DEV43_ARENA_GUARD_PREFIX = 0xA5;
static constexpr uint8_t DEV43_ARENA_GUARD_SUFFIX = 0x5A;

// dev46 production scheduler policy. The task itself stays at P1; only the
// TFLM Invoke window runs at P5 so it cannot be preempted by baby_rtsp (P4) on
// core 0. The original priority is restored on every return path after Invoke.
static constexpr UBaseType_t DEV46_ML_PRIORITY_BASE = 1;
static constexpr UBaseType_t DEV46_ML_PRIORITY_INVOKE = 5;

static constexpr size_t IDX_SPEECH = 0;
static constexpr size_t IDX_BABBLING = 4;
static constexpr size_t IDX_YELL = 9;
static constexpr size_t IDX_SCREAMING = 11;
static constexpr size_t IDX_CRYING = 19;
static constexpr size_t IDX_BABY_CRY = 20;

// v1.4.0-dev43: immutable FFT512 execution plan. Keeping the tables in
// flash avoids ~3 KiB of permanent internal-RAM pressure while removing
// per-frame bit-reversal generation and per-butterfly twiddle recurrence.
// The transform remains the same iterative radix-2 DIT FFT used by dev40.
static constexpr uint16_t DEV43_BITREV_512[512] = {
  0, 256, 128, 384, 64, 320, 192, 448, 32, 288, 160, 416, 96, 352, 224, 480,
  16, 272, 144, 400, 80, 336, 208, 464, 48, 304, 176, 432, 112, 368, 240, 496,
  8, 264, 136, 392, 72, 328, 200, 456, 40, 296, 168, 424, 104, 360, 232, 488,
  24, 280, 152, 408, 88, 344, 216, 472, 56, 312, 184, 440, 120, 376, 248, 504,
  4, 260, 132, 388, 68, 324, 196, 452, 36, 292, 164, 420, 100, 356, 228, 484,
  20, 276, 148, 404, 84, 340, 212, 468, 52, 308, 180, 436, 116, 372, 244, 500,
  12, 268, 140, 396, 76, 332, 204, 460, 44, 300, 172, 428, 108, 364, 236, 492,
  28, 284, 156, 412, 92, 348, 220, 476, 60, 316, 188, 444, 124, 380, 252, 508,
  2, 258, 130, 386, 66, 322, 194, 450, 34, 290, 162, 418, 98, 354, 226, 482,
  18, 274, 146, 402, 82, 338, 210, 466, 50, 306, 178, 434, 114, 370, 242, 498,
  10, 266, 138, 394, 74, 330, 202, 458, 42, 298, 170, 426, 106, 362, 234, 490,
  26, 282, 154, 410, 90, 346, 218, 474, 58, 314, 186, 442, 122, 378, 250, 506,
  6, 262, 134, 390, 70, 326, 198, 454, 38, 294, 166, 422, 102, 358, 230, 486,
  22, 278, 150, 406, 86, 342, 214, 470, 54, 310, 182, 438, 118, 374, 246, 502,
  14, 270, 142, 398, 78, 334, 206, 462, 46, 302, 174, 430, 110, 366, 238, 494,
  30, 286, 158, 414, 94, 350, 222, 478, 62, 318, 190, 446, 126, 382, 254, 510,
  1, 257, 129, 385, 65, 321, 193, 449, 33, 289, 161, 417, 97, 353, 225, 481,
  17, 273, 145, 401, 81, 337, 209, 465, 49, 305, 177, 433, 113, 369, 241, 497,
  9, 265, 137, 393, 73, 329, 201, 457, 41, 297, 169, 425, 105, 361, 233, 489,
  25, 281, 153, 409, 89, 345, 217, 473, 57, 313, 185, 441, 121, 377, 249, 505,
  5, 261, 133, 389, 69, 325, 197, 453, 37, 293, 165, 421, 101, 357, 229, 485,
  21, 277, 149, 405, 85, 341, 213, 469, 53, 309, 181, 437, 117, 373, 245, 501,
  13, 269, 141, 397, 77, 333, 205, 461, 45, 301, 173, 429, 109, 365, 237, 493,
  29, 285, 157, 413, 93, 349, 221, 477, 61, 317, 189, 445, 125, 381, 253, 509,
  3, 259, 131, 387, 67, 323, 195, 451, 35, 291, 163, 419, 99, 355, 227, 483,
  19, 275, 147, 403, 83, 339, 211, 467, 51, 307, 179, 435, 115, 371, 243, 499,
  11, 267, 139, 395, 75, 331, 203, 459, 43, 299, 171, 427, 107, 363, 235, 491,
  27, 283, 155, 411, 91, 347, 219, 475, 59, 315, 187, 443, 123, 379, 251, 507,
  7, 263, 135, 391, 71, 327, 199, 455, 39, 295, 167, 423, 103, 359, 231, 487,
  23, 279, 151, 407, 87, 343, 215, 471, 55, 311, 183, 439, 119, 375, 247, 503,
  15, 271, 143, 399, 79, 335, 207, 463, 47, 303, 175, 431, 111, 367, 239, 495,
  31, 287, 159, 415, 95, 351, 223, 479, 63, 319, 191, 447, 127, 383, 255, 511,
};
static constexpr float DEV43_TWIDDLE_RE_512[256] = {
  1.0f, 0.999924702f, 0.999698819f, 0.999322385f, 0.998795456f, 0.998118113f, 0.997290457f, 0.996312612f,
  0.995184727f, 0.99390697f, 0.992479535f, 0.990902635f, 0.98917651f, 0.987301418f, 0.985277642f, 0.983105487f,
  0.98078528f, 0.978317371f, 0.97570213f, 0.972939952f, 0.970031253f, 0.966976471f, 0.963776066f, 0.960430519f,
  0.956940336f, 0.95330604f, 0.949528181f, 0.945607325f, 0.941544065f, 0.937339012f, 0.932992799f, 0.92850608f,
  0.923879533f, 0.919113852f, 0.914209756f, 0.909167983f, 0.903989293f, 0.898674466f, 0.893224301f, 0.88763962f,
  0.881921264f, 0.876070094f, 0.870086991f, 0.863972856f, 0.85772861f, 0.851355193f, 0.844853565f, 0.838224706f,
  0.831469612f, 0.824589303f, 0.817584813f, 0.810457198f, 0.803207531f, 0.795836905f, 0.788346428f, 0.780737229f,
  0.773010453f, 0.765167266f, 0.757208847f, 0.749136395f, 0.740951125f, 0.732654272f, 0.724247083f, 0.715730825f,
  0.707106781f, 0.698376249f, 0.689540545f, 0.680600998f, 0.671558955f, 0.662415778f, 0.653172843f, 0.643831543f,
  0.634393284f, 0.624859488f, 0.615231591f, 0.605511041f, 0.595699304f, 0.585797857f, 0.575808191f, 0.565731811f,
  0.555570233f, 0.545324988f, 0.53499762f, 0.524589683f, 0.514102744f, 0.503538384f, 0.492898192f, 0.482183772f,
  0.471396737f, 0.460538711f, 0.44961133f, 0.438616239f, 0.427555093f, 0.41642956f, 0.405241314f, 0.39399204f,
  0.382683432f, 0.371317194f, 0.359895037f, 0.34841868f, 0.336889853f, 0.325310292f, 0.31368174f, 0.302005949f,
  0.290284677f, 0.278519689f, 0.266712757f, 0.25486566f, 0.24298018f, 0.231058108f, 0.21910124f, 0.207111376f,
  0.195090322f, 0.183039888f, 0.170961889f, 0.158858143f, 0.146730474f, 0.134580709f, 0.122410675f, 0.110222207f,
  0.0980171403f, 0.0857973123f, 0.0735645636f, 0.0613207363f, 0.0490676743f, 0.0368072229f, 0.0245412285f, 0.0122715383f,
  0.0f, -0.0122715383f, -0.0245412285f, -0.0368072229f, -0.0490676743f, -0.0613207363f, -0.0735645636f, -0.0857973123f,
  -0.0980171403f, -0.110222207f, -0.122410675f, -0.134580709f, -0.146730474f, -0.158858143f, -0.170961889f, -0.183039888f,
  -0.195090322f, -0.207111376f, -0.21910124f, -0.231058108f, -0.24298018f, -0.25486566f, -0.266712757f, -0.278519689f,
  -0.290284677f, -0.302005949f, -0.31368174f, -0.325310292f, -0.336889853f, -0.34841868f, -0.359895037f, -0.371317194f,
  -0.382683432f, -0.39399204f, -0.405241314f, -0.41642956f, -0.427555093f, -0.438616239f, -0.44961133f, -0.460538711f,
  -0.471396737f, -0.482183772f, -0.492898192f, -0.503538384f, -0.514102744f, -0.524589683f, -0.53499762f, -0.545324988f,
  -0.555570233f, -0.565731811f, -0.575808191f, -0.585797857f, -0.595699304f, -0.605511041f, -0.615231591f, -0.624859488f,
  -0.634393284f, -0.643831543f, -0.653172843f, -0.662415778f, -0.671558955f, -0.680600998f, -0.689540545f, -0.698376249f,
  -0.707106781f, -0.715730825f, -0.724247083f, -0.732654272f, -0.740951125f, -0.749136395f, -0.757208847f, -0.765167266f,
  -0.773010453f, -0.780737229f, -0.788346428f, -0.795836905f, -0.803207531f, -0.810457198f, -0.817584813f, -0.824589303f,
  -0.831469612f, -0.838224706f, -0.844853565f, -0.851355193f, -0.85772861f, -0.863972856f, -0.870086991f, -0.876070094f,
  -0.881921264f, -0.88763962f, -0.893224301f, -0.898674466f, -0.903989293f, -0.909167983f, -0.914209756f, -0.919113852f,
  -0.923879533f, -0.92850608f, -0.932992799f, -0.937339012f, -0.941544065f, -0.945607325f, -0.949528181f, -0.95330604f,
  -0.956940336f, -0.960430519f, -0.963776066f, -0.966976471f, -0.970031253f, -0.972939952f, -0.97570213f, -0.978317371f,
  -0.98078528f, -0.983105487f, -0.985277642f, -0.987301418f, -0.98917651f, -0.990902635f, -0.992479535f, -0.99390697f,
  -0.995184727f, -0.996312612f, -0.997290457f, -0.998118113f, -0.998795456f, -0.999322385f, -0.999698819f, -0.999924702f,
};
static constexpr float DEV43_TWIDDLE_IM_512[256] = {
  0.0f, -0.0122715383f, -0.0245412285f, -0.0368072229f, -0.0490676743f, -0.0613207363f, -0.0735645636f, -0.0857973123f,
  -0.0980171403f, -0.110222207f, -0.122410675f, -0.134580709f, -0.146730474f, -0.158858143f, -0.170961889f, -0.183039888f,
  -0.195090322f, -0.207111376f, -0.21910124f, -0.231058108f, -0.24298018f, -0.25486566f, -0.266712757f, -0.278519689f,
  -0.290284677f, -0.302005949f, -0.31368174f, -0.325310292f, -0.336889853f, -0.34841868f, -0.359895037f, -0.371317194f,
  -0.382683432f, -0.39399204f, -0.405241314f, -0.41642956f, -0.427555093f, -0.438616239f, -0.44961133f, -0.460538711f,
  -0.471396737f, -0.482183772f, -0.492898192f, -0.503538384f, -0.514102744f, -0.524589683f, -0.53499762f, -0.545324988f,
  -0.555570233f, -0.565731811f, -0.575808191f, -0.585797857f, -0.595699304f, -0.605511041f, -0.615231591f, -0.624859488f,
  -0.634393284f, -0.643831543f, -0.653172843f, -0.662415778f, -0.671558955f, -0.680600998f, -0.689540545f, -0.698376249f,
  -0.707106781f, -0.715730825f, -0.724247083f, -0.732654272f, -0.740951125f, -0.749136395f, -0.757208847f, -0.765167266f,
  -0.773010453f, -0.780737229f, -0.788346428f, -0.795836905f, -0.803207531f, -0.810457198f, -0.817584813f, -0.824589303f,
  -0.831469612f, -0.838224706f, -0.844853565f, -0.851355193f, -0.85772861f, -0.863972856f, -0.870086991f, -0.876070094f,
  -0.881921264f, -0.88763962f, -0.893224301f, -0.898674466f, -0.903989293f, -0.909167983f, -0.914209756f, -0.919113852f,
  -0.923879533f, -0.92850608f, -0.932992799f, -0.937339012f, -0.941544065f, -0.945607325f, -0.949528181f, -0.95330604f,
  -0.956940336f, -0.960430519f, -0.963776066f, -0.966976471f, -0.970031253f, -0.972939952f, -0.97570213f, -0.978317371f,
  -0.98078528f, -0.983105487f, -0.985277642f, -0.987301418f, -0.98917651f, -0.990902635f, -0.992479535f, -0.99390697f,
  -0.995184727f, -0.996312612f, -0.997290457f, -0.998118113f, -0.998795456f, -0.999322385f, -0.999698819f, -0.999924702f,
  -1.0f, -0.999924702f, -0.999698819f, -0.999322385f, -0.998795456f, -0.998118113f, -0.997290457f, -0.996312612f,
  -0.995184727f, -0.99390697f, -0.992479535f, -0.990902635f, -0.98917651f, -0.987301418f, -0.985277642f, -0.983105487f,
  -0.98078528f, -0.978317371f, -0.97570213f, -0.972939952f, -0.970031253f, -0.966976471f, -0.963776066f, -0.960430519f,
  -0.956940336f, -0.95330604f, -0.949528181f, -0.945607325f, -0.941544065f, -0.937339012f, -0.932992799f, -0.92850608f,
  -0.923879533f, -0.919113852f, -0.914209756f, -0.909167983f, -0.903989293f, -0.898674466f, -0.893224301f, -0.88763962f,
  -0.881921264f, -0.876070094f, -0.870086991f, -0.863972856f, -0.85772861f, -0.851355193f, -0.844853565f, -0.838224706f,
  -0.831469612f, -0.824589303f, -0.817584813f, -0.810457198f, -0.803207531f, -0.795836905f, -0.788346428f, -0.780737229f,
  -0.773010453f, -0.765167266f, -0.757208847f, -0.749136395f, -0.740951125f, -0.732654272f, -0.724247083f, -0.715730825f,
  -0.707106781f, -0.698376249f, -0.689540545f, -0.680600998f, -0.671558955f, -0.662415778f, -0.653172843f, -0.643831543f,
  -0.634393284f, -0.624859488f, -0.615231591f, -0.605511041f, -0.595699304f, -0.585797857f, -0.575808191f, -0.565731811f,
  -0.555570233f, -0.545324988f, -0.53499762f, -0.524589683f, -0.514102744f, -0.503538384f, -0.492898192f, -0.482183772f,
  -0.471396737f, -0.460538711f, -0.44961133f, -0.438616239f, -0.427555093f, -0.41642956f, -0.405241314f, -0.39399204f,
  -0.382683432f, -0.371317194f, -0.359895037f, -0.34841868f, -0.336889853f, -0.325310292f, -0.31368174f, -0.302005949f,
  -0.290284677f, -0.278519689f, -0.266712757f, -0.25486566f, -0.24298018f, -0.231058108f, -0.21910124f, -0.207111376f,
  -0.195090322f, -0.183039888f, -0.170961889f, -0.158858143f, -0.146730474f, -0.134580709f, -0.122410675f, -0.110222207f,
  -0.0980171403f, -0.0857973123f, -0.0735645636f, -0.0613207363f, -0.0490676743f, -0.0368072229f, -0.0245412285f, -0.0122715383f,
};


// v1.4.0-dev43 allocator-isolation checkpoint journal.
// RTC_NOINIT survives watchdog/software resets, so the next boot can report
// the last TFLM stage reached even when the logger/API died before flushing.
// This is intentionally tiny and lock-free: checkpoint writes must not call
// logging, allocate memory, enter FreeRTOS critical sections, or touch lwIP.
static constexpr uint32_t DEV43_RTC_MAGIC = 0x43525932u;  // "CRY2"
struct Dev24RtcCheckpoint {
  uint32_t magic;
  uint32_t stage;
  uint32_t detail;
  uint32_t seq;
};
RTC_NOINIT_ATTR static Dev24RtcCheckpoint dev43_rtc_checkpoint_;

static inline void dev43_mark_(uint32_t stage, uint32_t detail = 0) {
  if (dev43_rtc_checkpoint_.magic != DEV43_RTC_MAGIC) {
    dev43_rtc_checkpoint_.seq = 0;
  }
  dev43_rtc_checkpoint_.magic = DEV43_RTC_MAGIC;
  dev43_rtc_checkpoint_.stage = stage;
  dev43_rtc_checkpoint_.detail = detail;
  dev43_rtc_checkpoint_.seq++;
}

static void dev43_report_previous_checkpoint_() {
  if (dev43_rtc_checkpoint_.magic != DEV43_RTC_MAGIC) {
    ESP_LOGI(TAG, "Cry dev46 previous allocator checkpoint: none (cold boot/power cycle)");
    return;
  }
  ESP_LOGW(TAG,
           "Cry dev46 previous allocator checkpoint: stage=0x%08X detail=%u seq=%u; "
           "A000=before AllocateTensors, A1xx=kernel Init, B1xx=native Prepare, "
           "B900=MEAN Prepare, BA00=FC Prepare, C1xx=scratch handles, C2xx=AllocateVariables, "
           "D3xx=planner Init/CreatePlan, D4xx=CommitPlan, E1xx=ResetTempAllocations, D5xx=post-resizable-head GetMaximumMemorySize, F000=AllocateTensors returned",
           (unsigned) dev43_rtc_checkpoint_.stage,
           (unsigned) dev43_rtc_checkpoint_.detail,
           (unsigned) dev43_rtc_checkpoint_.seq);
}


// dev46: instrument the native LinearMemoryPlanner without copying or patching
// MicroAllocator::CommitStaticMemoryPlan().  The call sequence itself provides
// stable boundaries inside that private method:
//   D3xx: planner Init/AddBuffer => InitializeAllocationInfo + lifetimes passed
//   D4xx: GetOffsetForBuffer      => CreatePlan passed; CommitPlan in progress
//   D4A0: all offsets committed   => CommitPlan passed
//   D5xx: GetMaximumMemorySize    => temp cleanup + resizable-head release passed
// If AllocateTensors still returns kTfLiteError after D5A0, the only remaining
// upstream operation is ReserveNonPersistentOverlayMemory().
class Dev34TracingLinearPlanner final : public tflite::LinearMemoryPlanner {
 public:
  static void operator delete(void *) noexcept {}
  ~Dev34TracingLinearPlanner() override = default;

  TfLiteStatus Init(unsigned char *scratch_buffer, int scratch_buffer_size) override {
    add_count_ = 0;
    offset_count_ = 0;
    dev43_mark_(0xD300u, static_cast<uint32_t>(scratch_buffer_size));
    const TfLiteStatus status =
        tflite::LinearMemoryPlanner::Init(scratch_buffer, scratch_buffer_size);
    dev43_mark_(status == kTfLiteOk ? 0xD3A0u : 0xD3FFu,
                static_cast<uint32_t>(scratch_buffer_size));
    ESP_LOGI(TAG,
             "Cry dev46 CSP planner Init: status=%d arena=%p bytes=%d",
             (int) status, scratch_buffer, scratch_buffer_size);
    return status;
  }

  TfLiteStatus AddBuffer(int size, int first_time_used, int last_time_used) override {
    const uint32_t index = static_cast<uint32_t>(add_count_);
    dev43_mark_(0xD310u, index);
    const TfLiteStatus status = tflite::LinearMemoryPlanner::AddBuffer(
        size, first_time_used, last_time_used);
    if (status == kTfLiteOk) {
      ++add_count_;
      dev43_mark_(0xD311u, static_cast<uint32_t>(add_count_));
    } else {
      dev43_mark_(0xD3FFu, index);
    }
    return status;
  }


  TfLiteStatus GetOffsetForBuffer(int buffer_index, int *offset) override {
    dev43_mark_(0xD400u, static_cast<uint32_t>(buffer_index));
    const TfLiteStatus status =
        tflite::LinearMemoryPlanner::GetOffsetForBuffer(buffer_index, offset);
    if (status != kTfLiteOk) {
      dev43_mark_(0xD4FFu, static_cast<uint32_t>(buffer_index));
      return status;
    }
    ++offset_count_;
    dev43_mark_(0xD410u, static_cast<uint32_t>(offset_count_));
    if (offset_count_ == add_count_) {
      dev43_mark_(0xD4A0u, static_cast<uint32_t>(offset_count_));
      ESP_LOGI(TAG,
               "Cry dev46 CSP CommitPlan boundary: PASS offsets=%d buffers=%d last_offset=%d",
               offset_count_, add_count_, offset ? *offset : -1);
    }
    return status;
  }

  size_t GetMaximumMemorySize() override {
    dev43_mark_(0xD500u, static_cast<uint32_t>(offset_count_));
    const size_t maximum = tflite::LinearMemoryPlanner::GetMaximumMemorySize();
    dev43_mark_(0xD5A0u, static_cast<uint32_t>(maximum));
    ESP_LOGI(TAG,
             "Cry dev46 CSP cleanup boundary: PASS GetMaximumMemorySize=%u B buffers=%d offsets=%d",
             (unsigned) maximum, add_count_, offset_count_);
    return maximum;
  }

  int GetBufferCount() override {
    return tflite::LinearMemoryPlanner::GetBufferCount();
  }

  void PrintMemoryPlan() override {
    tflite::LinearMemoryPlanner::PrintMemoryPlan();
  }

  bool preserves_all_tensors() const override { return true; }

 private:
  int add_count_{0};
  int offset_count_{0};
};

// dev46: narrow FinishModelAllocation without patching the packaged TFLM.
// MicroAllocator::FinishModelAllocation() first invokes the private virtual
// AllocateScratchBufferHandles(), then private virtual CommitStaticMemoryPlan().
// C++ allows overriding private virtuals in a derived class.  We therefore
// preserve the upstream FinishModelAllocation implementation and instrument the
// first boundary only.  If C1A0 is reached and AllocateTensors still fails, the
// failure is downstream in CommitStaticMemoryPlan().  AllocateVariables() is a
// protected virtual called from inside CommitStaticMemoryPlan, so the C2xx
// checkpoint additionally tells us whether CreateAllocationInfo() and
// GetOfflinePlannedOffsets() have already succeeded.
class Dev34Allocator final : public tflite::MicroAllocator {
 public:
  Dev34Allocator(tflite::SingleArenaBufferAllocator *arena_allocator,
                 tflite::MicroMemoryPlanner *memory_planner)
      : tflite::MicroAllocator(arena_allocator, memory_planner) {}

  // TFLM intentionally makes MicroAllocator::operator delete private via
  // TF_LITE_REMOVE_VIRTUAL_DELETE.  Because the base destructor is virtual, a
  // derived class needs an accessible final deallocation function even though
  // this probe is placement-new'd in arena storage and is never deleted.
  // Supplying a no-op class-local delete keeps the derived destructor
  // well-formed without changing allocator lifetime or runtime behaviour.
  static void operator delete(void *) noexcept {}
  ~Dev34Allocator() override = default;

 private:
  TfLiteStatus AllocateScratchBufferHandles(
      tflite::ScratchBufferHandle **scratch_buffer_handles,
      size_t handle_count) override {
    dev43_mark_(0xC100u, static_cast<uint32_t>(handle_count));
    const size_t internal_before =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t bytes = sizeof(tflite::ScratchBufferHandle) * handle_count;

    if (handle_count == 0) {
      dev43_mark_(0xC1A0u, 0);
      ESP_LOGI(TAG,
               "Cry dev46 FMA scratch-handles: PASS count=0 bytes=0 ptr=null (no requests)");
      return kTfLiteOk;
    }

    // Base implementation uses the persistent arena allocator with
    // alignof(ScratchBufferHandle). AllocatePersistentBuffer() uses the arena
    // alignment, which is at least as strict on ESP32-S3, while keeping the
    // same persistent-tail allocation semantics.  Importantly, upstream returns
    // kTfLiteOk even if the pointer is null; preserve that status semantics and
    // only record the pointer state.
    *scratch_buffer_handles = reinterpret_cast<tflite::ScratchBufferHandle *>(
        AllocatePersistentBuffer(bytes));
    const size_t internal_after =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const bool nonnull = (*scratch_buffer_handles != nullptr);
    dev43_mark_(0xC1A0u, nonnull ? static_cast<uint32_t>(bytes) : 0xFFFFFFFFu);
    ESP_LOGI(TAG,
             "Cry dev46 FMA scratch-handles: PASS count=%u bytes=%u ptr=%p nonnull=%s internal=%u->%u delta=%d B",
             (unsigned) handle_count, (unsigned) bytes, *scratch_buffer_handles,
             nonnull ? "YES" : "NO", (unsigned) internal_before,
             (unsigned) internal_after,
             (int) internal_after - (int) internal_before);
    return kTfLiteOk;
  }

  TfLiteStatus ResetTempAllocations() override {
    // dev46: production-like path. dev37 proved the temporary-tensor lifecycle
    // is now correct, so remove RTC/heap probes and per-call logging from the
    // hot inference path. Keep the override only to preserve the exact allocator
    // stack that made AllocateTensors reliable.
    return tflite::MicroAllocator::ResetTempAllocations();
  }

  TfLiteStatus AllocateVariables(
      const tflite::SubGraph *subgraph, TfLiteEvalTensor *eval_tensors,
      const int32_t *offline_planner_offsets) override {
    dev43_mark_(0xC200u,
                subgraph && subgraph->tensors()
                    ? static_cast<uint32_t>(subgraph->tensors()->size())
                    : 0u);
    size_t variables = 0;
    if (subgraph && subgraph->tensors()) {
      for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
        const auto *t = subgraph->tensors()->Get(i);
        if (t != nullptr && t->is_variable()) ++variables;
      }
    }
    const TfLiteStatus status = tflite::MicroAllocator::AllocateVariables(
        subgraph, eval_tensors, offline_planner_offsets);
    dev43_mark_(status == kTfLiteOk ? 0xC2A0u : 0xC2FFu,
                static_cast<uint32_t>(variables));
    ESP_LOGI(TAG,
             "Cry dev46 FMA AllocateVariables: status=%d tensors=%u variables=%u offline_offsets=%p",
             (int) status,
             (unsigned) (subgraph && subgraph->tensors()
                             ? subgraph->tensors()->size()
                             : 0),
             (unsigned) variables, offline_planner_offsets);
    return status;
  }

};

static const char *builtin_op_name_(tflite::BuiltinOperator op) {
  switch (op) {
    case tflite::BuiltinOperator_ADD: return "ADD";
    case tflite::BuiltinOperator_AVERAGE_POOL_2D: return "AVERAGE_POOL_2D";
    case tflite::BuiltinOperator_CONV_2D: return "CONV_2D";
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: return "DEPTHWISE_CONV_2D";
    case tflite::BuiltinOperator_FULLY_CONNECTED: return "FULLY_CONNECTED";
    case tflite::BuiltinOperator_RESHAPE: return "RESHAPE";
    case tflite::BuiltinOperator_SOFTMAX: return "SOFTMAX";
    case tflite::BuiltinOperator_MUL: return "MUL";
    case tflite::BuiltinOperator_RELU: return "RELU";
    case tflite::BuiltinOperator_RELU6: return "RELU6";
    case tflite::BuiltinOperator_QUANTIZE: return "QUANTIZE";
    case tflite::BuiltinOperator_DEQUANTIZE: return "DEQUANTIZE";
    case tflite::BuiltinOperator_LOGISTIC: return "LOGISTIC";
    case tflite::BuiltinOperator_MEAN: return "MEAN";
    case tflite::BuiltinOperator_PAD: return "PAD";
    case tflite::BuiltinOperator_PADV2: return "PADV2";
    case tflite::BuiltinOperator_MAX_POOL_2D: return "MAX_POOL_2D";
    case tflite::BuiltinOperator_CONCATENATION: return "CONCATENATION";
    case tflite::BuiltinOperator_SHAPE: return "SHAPE";
    case tflite::BuiltinOperator_STRIDED_SLICE: return "STRIDED_SLICE";
    case tflite::BuiltinOperator_SQUEEZE: return "SQUEEZE";
    case tflite::BuiltinOperator_TRANSPOSE: return "TRANSPOSE";
    default: return "OTHER";
  }
}

static const char *tensor_type_name_(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_FLOAT32: return "FLOAT32";
    case tflite::TensorType_INT32: return "INT32";
    case tflite::TensorType_UINT8: return "UINT8";
    case tflite::TensorType_INT64: return "INT64";
    case tflite::TensorType_BOOL: return "BOOL";
    case tflite::TensorType_INT16: return "INT16";
    case tflite::TensorType_INT8: return "INT8";
    case tflite::TensorType_FLOAT16: return "FLOAT16";
    default: return "OTHER";
  }
}

static void format_tensor_shape_(const flatbuffers::Vector<int32_t> *shape,
                                 char *dst, size_t dst_size) {
  if (dst == nullptr || dst_size == 0) return;
  if (shape == nullptr || shape->size() == 0) {
    std::snprintf(dst, dst_size, "[]");
    return;
  }

  size_t used = 0;
  int n = std::snprintf(dst, dst_size, "[");
  if (n < 0) return;
  used = static_cast<size_t>(n);

  for (unsigned i = 0; i < shape->size() && used + 2 < dst_size; i++) {
    n = std::snprintf(dst + used, dst_size - used, "%s%ld",
                      i ? "," : "", static_cast<long>(shape->Get(i)));
    if (n < 0) break;
    const size_t wrote = static_cast<size_t>(n);
    if (wrote >= dst_size - used) {
      used = dst_size - 1;
      break;
    }
    used += wrote;
  }

  if (used + 2 <= dst_size) {
    std::snprintf(dst + used, dst_size - used, "]");
  } else {
    dst[dst_size - 1] = '\0';
  }
}


static uint32_t dev43_read_le32_(const uint8_t *p, size_t remaining) {
  if (p == nullptr || remaining < 4) return 0;
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static void dev43_log_model_metadata_(const tflite::Model *model) {
  if (model == nullptr) return;
  const auto *metadata = model->metadata();
  const auto *buffers = model->buffers();
  if (metadata == nullptr || metadata->size() == 0) {
    ESP_LOGI(TAG, "Cry dev46 metadata audit: count=0; OfflineMemoryAllocation absent");
    return;
  }

  ESP_LOGI(TAG, "Cry dev46 metadata audit: count=%u", (unsigned) metadata->size());
  bool offline_seen = false;
  for (unsigned i = 0; i < metadata->size(); i++) {
    const auto *m = metadata->Get(i);
    if (m == nullptr) continue;
    const char *name = m->name() ? m->name()->c_str() : "";
    const uint32_t bi = m->buffer();
    const uint8_t *raw = nullptr;
    size_t bytes = 0;
    if (buffers != nullptr && bi < buffers->size()) {
      const auto *b = buffers->Get(bi);
      if (b != nullptr && b->data() != nullptr && b->data()->size() > 0) {
        raw = b->data()->Data();
        bytes = b->data()->size();
      }
    }
    const bool offline = std::strcmp(name, "OfflineMemoryAllocation") == 0;
    offline_seen = offline_seen || offline;
    ESP_LOGI(TAG,
             "Cry dev46 META[%u]: name='%s' buffer=%u bytes=%u offline_memory=%s",
             i, name, (unsigned) bi, (unsigned) bytes, offline ? "YES" : "NO");

    if (raw != nullptr && bytes > 0) {
      char hex[3 * 32 + 1]{};
      size_t used = 0;
      const size_t dump_n = std::min<size_t>(bytes, 32);
      for (size_t j = 0; j < dump_n && used + 4 < sizeof(hex); j++) {
        int n = std::snprintf(hex + used, sizeof(hex) - used, "%s%02X",
                              j ? " " : "", raw[j]);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(hex) - used) break;
        used += static_cast<size_t>(n);
      }
      ESP_LOGI(TAG, "Cry dev46 META[%u] raw[0..%u]=%s%s", i,
               (unsigned) (dump_n ? dump_n - 1 : 0), hex,
               bytes > dump_n ? " ..." : "");

      const size_t words = std::min<size_t>(bytes / 4, 16);
      for (size_t w = 0; w < words; w++) {
        const uint32_t v = dev43_read_le32_(raw + w * 4, bytes - w * 4);
        ESP_LOGI(TAG, "Cry dev46 META[%u] LE32[%u]=%u (0x%08X)", i,
                 (unsigned) w, (unsigned) v, (unsigned) v);
      }
    }
  }
  ESP_LOGI(TAG, "Cry dev46 metadata audit summary: OfflineMemoryAllocation=%s",
           offline_seen ? "PRESENT" : "ABSENT");
}

static void log_model_graph_(const tflite::Model *model) {
  if (model == nullptr) return;

  const auto *subgraphs = model->subgraphs();
  const auto *buffers = model->buffers();
  if (subgraphs == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 graph preflight: model has no subgraphs");
    return;
  }

  ESP_LOGI(TAG, "Cry dev46 graph preflight: subgraphs=%u buffers=%u",
           (unsigned) subgraphs->size(),
           (unsigned) (buffers ? buffers->size() : 0));

  for (unsigned sg_i = 0; sg_i < subgraphs->size(); sg_i++) {
    const auto *sg = subgraphs->Get(sg_i);
    if (sg == nullptr) continue;

    const auto *tensors = sg->tensors();
    const auto *ops = sg->operators();
    const auto *sg_inputs = sg->inputs();
    const auto *sg_outputs = sg->outputs();

    ESP_LOGI(TAG,
             "Cry dev46 subgraph[%u]: tensors=%u operators=%u inputs=%u outputs=%u",
             sg_i,
             (unsigned) (tensors ? tensors->size() : 0),
             (unsigned) (ops ? ops->size() : 0),
             (unsigned) (sg_inputs ? sg_inputs->size() : 0),
             (unsigned) (sg_outputs ? sg_outputs->size() : 0));

    if (tensors != nullptr) {
      for (unsigned t_i = 0; t_i < tensors->size(); t_i++) {
        const auto *tensor = tensors->Get(t_i);
        if (tensor == nullptr) continue;

        char shape[96]{};
        format_tensor_shape_(tensor->shape(), shape, sizeof(shape));

        size_t const_bytes = 0;
        if (buffers != nullptr && tensor->buffer() < buffers->size()) {
          const auto *buffer = buffers->Get(tensor->buffer());
          if (buffer != nullptr && buffer->data() != nullptr)
            const_bytes = buffer->data()->size();
        }

        float q_scale = NAN;
        long long q_zp = 0;
        long q_dim = -1;
        const auto *q = tensor->quantization();
        if (q != nullptr) {
          if (q->scale() != nullptr && q->scale()->size() > 0)
            q_scale = q->scale()->Get(0);
          if (q->zero_point() != nullptr && q->zero_point()->size() > 0)
            q_zp = static_cast<long long>(q->zero_point()->Get(0));
          q_dim = static_cast<long>(q->quantized_dimension());
        }

        const char *name =
            (tensor->name() != nullptr) ? tensor->name()->c_str() : "";

        ESP_LOGI(TAG,
                 "Cry tensor[%u]: type=%s(%d) shape=%s buffer=%u const=%uB "
                 "q_scale=%.9g q_zp=%lld q_dim=%ld name='%s'",
                 t_i,
                 tensor_type_name_(tensor->type()),
                 (int) tensor->type(),
                 shape,
                 (unsigned) tensor->buffer(),
                 (unsigned) const_bytes,
                 q_scale,
                 q_zp,
                 q_dim,
                 name);
      }
    }

    if (ops != nullptr) {
      const auto *opcodes = model->operator_codes();
      for (unsigned op_i = 0; op_i < ops->size(); op_i++) {
        const auto *op = ops->Get(op_i);
        if (op == nullptr) continue;

        int builtin = -1;
        int version = -1;
        const char *op_name = "UNKNOWN";
        if (opcodes != nullptr && op->opcode_index() < opcodes->size()) {
          const auto *code = opcodes->Get(op->opcode_index());
          if (code != nullptr) {
            const auto op_code = code->builtin_code();
            builtin = static_cast<int>(op_code);
            version = static_cast<int>(code->version());
            op_name = builtin_op_name_(op_code);
          }
        }

        char in_buf[128]{};
        char out_buf[128]{};
        const auto *inputs = op->inputs();
        const auto *outputs = op->outputs();

        size_t in_used = 0;
        int n = std::snprintf(in_buf, sizeof(in_buf), "[");
        if (n > 0) in_used = static_cast<size_t>(n);
        if (inputs != nullptr) {
          for (unsigned i = 0; i < inputs->size() && in_used + 4 < sizeof(in_buf); i++) {
            n = std::snprintf(in_buf + in_used, sizeof(in_buf) - in_used,
                              "%s%ld", i ? "," : "",
                              static_cast<long>(inputs->Get(i)));
            if (n < 0 || static_cast<size_t>(n) >= sizeof(in_buf) - in_used) break;
            in_used += static_cast<size_t>(n);
          }
        }
        if (in_used + 2 <= sizeof(in_buf))
          std::snprintf(in_buf + in_used, sizeof(in_buf) - in_used, "]");

        size_t out_used = 0;
        n = std::snprintf(out_buf, sizeof(out_buf), "[");
        if (n > 0) out_used = static_cast<size_t>(n);
        if (outputs != nullptr) {
          for (unsigned i = 0; i < outputs->size() && out_used + 4 < sizeof(out_buf); i++) {
            n = std::snprintf(out_buf + out_used, sizeof(out_buf) - out_used,
                              "%s%ld", i ? "," : "",
                              static_cast<long>(outputs->Get(i)));
            if (n < 0 || static_cast<size_t>(n) >= sizeof(out_buf) - out_used) break;
            out_used += static_cast<size_t>(n);
          }
        }
        if (out_used + 2 <= sizeof(out_buf))
          std::snprintf(out_buf + out_used, sizeof(out_buf) - out_used, "]");

        ESP_LOGI(TAG,
                 "Cry node[%u]: opcode_index=%u builtin=%d name=%s version=%d "
                 "inputs=%s outputs=%s builtin_options=%d",
                 op_i,
                 (unsigned) op->opcode_index(),
                 builtin,
                 op_name,
                 version,
                 in_buf,
                 out_buf,
                 (int) op->builtin_options_type());
      }
    }
  }
}


// ---------------------------------------------------------------------------
// v1.4.0-dev43: per-kernel Init/Prepare wrappers.
//
// MicroInterpreter::AllocateTensors() internally runs:
//   StartModelAllocation -> registration parsing -> InitSubgraphs ->
//   PrepareSubgraphs -> FinishModelAllocation.
// The public resolver API allows us to register copies of the real kernels.
// Wrapping init()/prepare() therefore tells us how far AllocateTensors() gets
// without depending on private MicroInterpreter / MicroAllocator internals.
// ---------------------------------------------------------------------------

using Dev15InitFn = void *(*)(TfLiteContext *, const char *, size_t);
using Dev15PrepareFn = TfLiteStatus (*)(TfLiteContext *, TfLiteNode *);
using Dev43InvokeFn = TfLiteStatus (*)(TfLiteContext *, TfLiteNode *);

static Dev15InitFn dev43_orig_conv_init_{nullptr};
static Dev15InitFn dev43_orig_depthwise_init_{nullptr};

static Dev15PrepareFn dev43_orig_conv_prepare_{nullptr};
static Dev15PrepareFn dev43_orig_depthwise_prepare_{nullptr};
static Dev43InvokeFn dev43_orig_conv_invoke_{nullptr};
static Dev43InvokeFn dev43_orig_depthwise_invoke_{nullptr};

// dev46: zero-allocation per-node Invoke profiler.  TFLM invokes graph nodes
// serially on the ML task, so the callback order is the exact graph order.
// We aggregate five successful inferences and log only after Invoke returns.
static constexpr size_t DEV43_MAX_PROFILE_NODES = 16;
struct Dev43NodePerf {
  const char *op{nullptr};
  uint64_t total_us{0};
  uint32_t calls{0};
  uint32_t min_us{UINT32_MAX};
  uint32_t max_us{0};
};
static Dev43NodePerf dev43_node_perf_[DEV43_MAX_PROFILE_NODES]{};
static uint32_t dev43_invoke_node_seq_{0};
static uint32_t dev43_profile_samples_{0};
static uint64_t dev43_profile_invoke_total_us_{0};
static uint32_t dev43_profile_node_count_max_{0};

static void dev43_profile_begin_inference_() {
  dev43_invoke_node_seq_ = 0;
}

static TfLiteStatus dev43_profile_call_(const char *op, Dev43InvokeFn fn,
                                        TfLiteContext *context, TfLiteNode *node) {
  const uint32_t idx = dev43_invoke_node_seq_++;
  const int64_t start_us = esp_timer_get_time();
  const TfLiteStatus status = fn != nullptr ? fn(context, node) : kTfLiteError;
  const uint32_t elapsed_us = static_cast<uint32_t>(
      std::max<int64_t>(0, esp_timer_get_time() - start_us));
  if (idx < DEV43_MAX_PROFILE_NODES) {
    auto &p = dev43_node_perf_[idx];
    if (p.op == nullptr) p.op = op;
    p.total_us += elapsed_us;
    p.calls++;
    p.min_us = std::min(p.min_us, elapsed_us);
    p.max_us = std::max(p.max_us, elapsed_us);
  }
  dev43_profile_node_count_max_ = std::max(dev43_profile_node_count_max_, idx + 1);
  return status;
}

static void dev43_profile_commit_inference_(uint32_t invoke_us) {
  dev43_profile_samples_++;
  dev43_profile_invoke_total_us_ += invoke_us;
}

static void dev43_profile_log_and_reset_(uint32_t inference_number) {
  if (dev43_profile_samples_ < 5) return;
  const uint32_t n = dev43_profile_samples_;
  uint64_t accounted_us = 0;
  const uint32_t nodes = std::min<uint32_t>(dev43_profile_node_count_max_, DEV43_MAX_PROFILE_NODES);
  for (uint32_t i = 0; i < nodes; i++) accounted_us += dev43_node_perf_[i].total_us;
  const uint32_t invoke_avg_us = n ? static_cast<uint32_t>(dev43_profile_invoke_total_us_ / n) : 0;
  const uint32_t accounted_avg_us = n ? static_cast<uint32_t>(accounted_us / n) : 0;
  const char *sched_label = "P5";
  ESP_LOGI(TAG,
           "Cry dev46 NODEPROF[%s]: n=%u inference_count=%u nodes=%u invoke=%ums avg accounted=%ums avg coverage=%.1f%%",
           sched_label,
           (unsigned) n, (unsigned) inference_number, (unsigned) nodes,
           (unsigned) ((invoke_avg_us + 500) / 1000),
           (unsigned) ((accounted_avg_us + 500) / 1000),
           invoke_avg_us ? (100.0 * (double) accounted_avg_us / (double) invoke_avg_us) : 0.0);

  for (uint32_t base = 0; base < nodes; base += 3) {
    char line[384];
    size_t used = 0;
    const int n0 = std::snprintf(line, sizeof(line), "Cry dev46 NODE[%u-%u]:",
                                 (unsigned) base,
                                 (unsigned) std::min<uint32_t>(nodes - 1, base + 2));
    if (n0 > 0) used = std::min<size_t>((size_t) n0, sizeof(line) - 1);
    for (uint32_t i = base; i < nodes && i < base + 3 && used < sizeof(line) - 1; i++) {
      const auto &p = dev43_node_perf_[i];
      const uint32_t avg_us = p.calls ? static_cast<uint32_t>(p.total_us / p.calls) : 0;
      const double share = invoke_avg_us ? 100.0 * (double) avg_us / (double) invoke_avg_us : 0.0;
      const int nw = std::snprintf(line + used, sizeof(line) - used,
          " %u:%s=%ums(%.1f%%,%u-%ums)", (unsigned) i,
          p.op ? p.op : "?", (unsigned) ((avg_us + 500) / 1000), share,
          (unsigned) (((p.min_us == UINT32_MAX ? 0 : p.min_us) + 500) / 1000),
          (unsigned) ((p.max_us + 500) / 1000));
      if (nw <= 0) break;
      used += std::min<size_t>((size_t) nw, sizeof(line) - used - 1);
    }
    ESP_LOGI(TAG, "%s", line);
  }

  for (auto &p : dev43_node_perf_) p = Dev43NodePerf{};
  dev43_profile_samples_ = 0;
  dev43_profile_invoke_total_us_ = 0;
  dev43_profile_node_count_max_ = 0;
}

static uint32_t dev43_init_seq_{0};
static uint32_t dev43_prepare_seq_{0};

// dev46: intercept native kernel scratch requests while Prepare() is running.
// This uses the public TfLiteContext callback, so it does not depend on private
// MicroAllocator headers.  The original callback is restored immediately after
// every wrapped Prepare().
using Dev15ScratchRequestFn = TfLiteStatus (*)(TfLiteContext *, size_t, int *);
static Dev15ScratchRequestFn dev43_orig_scratch_request_{nullptr};
static uint32_t dev43_scratch_current_node_{0xFFFFFFFFu};
static const char *dev43_scratch_current_op_{"none"};
static constexpr size_t DEV43_MAX_SCRATCH_REQUESTS = 32;

struct Dev15ScratchRequest {
  uint32_t node{0};
  const char *op{nullptr};
  size_t bytes{0};
  int index{-1};
  TfLiteStatus status{kTfLiteError};
};

static Dev15ScratchRequest dev43_scratch_requests_[DEV43_MAX_SCRATCH_REQUESTS]{};
static size_t dev43_scratch_count_{0};
static size_t dev43_scratch_total_bytes_{0};
static size_t dev43_scratch_max_bytes_{0};
static size_t dev43_scratch_overflow_{0};
static tflite::RecordingMicroInterpreter *dev43_active_interpreter_{nullptr};

static const tflite::Model *dev43_active_model_{nullptr};


// dev46: scalar snapshot of the exact pre-Finish planner candidates.  The
// snapshot is captured during the final FC Prepare while TfLiteContext is
// valid, but it contains no pointers into the interpreter/arena.  The real
// GreedyMemoryPlanner replay is deliberately deferred until AllocateTensors()
// has returned, so the replay can no longer perturb FinishModelAllocation().
struct Dev30ReplayRecord {
  int bytes{0};
  int first{0};
  int last{0};
  int source{-1};      // tensor index or scratch slot
  bool scratch{false};
};
static constexpr size_t DEV43_MAX_REPLAY_RECORDS = 32;
static Dev30ReplayRecord dev43_replay_records_[DEV43_MAX_REPLAY_RECORDS]{};
static size_t dev43_replay_count_{0};
static bool dev43_replay_ready_{false};
static size_t dev43_replay_combined_peak_{0};


static size_t dev43_eval_type_size_(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return 4;
    case kTfLiteInt32: return 4;
    case kTfLiteUInt8: return 1;
    case kTfLiteInt64: return 8;
    case kTfLiteBool: return 1;
    case kTfLiteInt16: return 2;
    case kTfLiteInt8: return 1;
    case kTfLiteFloat16: return 2;
    default: return 0;
  }
}

static const char *dev43_eval_type_name_(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return "FLOAT32";
    case kTfLiteInt32: return "INT32";
    case kTfLiteUInt8: return "UINT8";
    case kTfLiteInt64: return "INT64";
    case kTfLiteBool: return "BOOL";
    case kTfLiteInt16: return "INT16";
    case kTfLiteInt8: return "INT8";
    case kTfLiteFloat16: return "FLOAT16";
    default: return "OTHER";
  }
}

static void dev43_format_eval_shape_(const TfLiteIntArray *dims, char *dst,
                                     size_t dst_size) {
  if (dst == nullptr || dst_size == 0) return;
  if (dims == nullptr) {
    std::snprintf(dst, dst_size, "<null>");
    return;
  }
  size_t used = 0;
  int n = std::snprintf(dst, dst_size, "[");
  if (n < 0) return;
  used = std::min<size_t>((size_t) n, dst_size - 1);
  for (int i = 0; i < dims->size && used + 2 < dst_size; i++) {
    n = std::snprintf(dst + used, dst_size - used, "%s%d", i ? "," : "",
                      dims->data[i]);
    if (n < 0) break;
    if ((size_t) n >= dst_size - used) {
      used = dst_size - 1;
      break;
    }
    used += (size_t) n;
  }
  if (used + 2 <= dst_size) std::snprintf(dst + used, dst_size - used, "]");
  else dst[dst_size - 1] = '\0';
}

static size_t dev43_eval_tensor_bytes_(const TfLiteEvalTensor *tensor) {
  if (tensor == nullptr || tensor->dims == nullptr) return 0;
  size_t elements = 1;
  for (int i = 0; i < tensor->dims->size; i++) {
    const int d = tensor->dims->data[i];
    if (d < 0) return 0;
    elements *= static_cast<size_t>(d);
  }
  const size_t type_size = dev43_eval_type_size_(tensor->type);
  return elements * type_size;
}

static bool dev43_model_tensor_is_const_(const tflite::SubGraph *sg,
                                         const tflite::Model *model,
                                         int tensor_index,
                                         uint32_t *buffer_index_out,
                                         size_t *const_bytes_out) {
  if (buffer_index_out) *buffer_index_out = 0;
  if (const_bytes_out) *const_bytes_out = 0;
  if (sg == nullptr || model == nullptr || sg->tensors() == nullptr ||
      tensor_index < 0 || tensor_index >= (int) sg->tensors()->size()) return false;
  const auto *ft = sg->tensors()->Get(tensor_index);
  if (ft == nullptr) return false;
  const uint32_t bi = ft->buffer();
  if (buffer_index_out) *buffer_index_out = bi;
  const auto *buffers = model->buffers();
  if (buffers == nullptr || bi >= buffers->size()) return false;
  const auto *buf = buffers->Get(bi);
  const size_t bytes = (buf && buf->data()) ? buf->data()->size() : 0;
  if (const_bytes_out) *const_bytes_out = bytes;
  return bytes > 0;
}

static void dev43_compute_lifetimes_(const tflite::SubGraph *sg, int *first,
                                     int *last, int count) {
  for (int i = 0; i < count; i++) {
    first[i] = -1;
    last[i] = -1;
  }
  if (sg == nullptr) return;
  const auto *ops = sg->operators();
  const int op_count = ops ? (int) ops->size() : 0;

  if (sg->inputs()) {
    for (unsigned i = 0; i < sg->inputs()->size(); i++) {
      const int t = sg->inputs()->Get(i);
      if (t >= 0 && t < count) first[t] = 0;
    }
  }

  if (ops) {
    for (int op_i = 0; op_i < op_count; op_i++) {
      const auto *op = ops->Get(op_i);
      if (op == nullptr) continue;
      if (op->inputs()) {
        for (unsigned j = 0; j < op->inputs()->size(); j++) {
          const int t = op->inputs()->Get(j);
          if (t < 0 || t >= count) continue;
          if (first[t] < 0) first[t] = op_i;
          last[t] = std::max(last[t], op_i);
        }
      }
      if (op->outputs()) {
        for (unsigned j = 0; j < op->outputs()->size(); j++) {
          const int t = op->outputs()->Get(j);
          if (t < 0 || t >= count) continue;
          if (first[t] < 0) first[t] = op_i;
          last[t] = std::max(last[t], op_i);
        }
      }
    }
  }

  if (sg->outputs()) {
    for (unsigned i = 0; i < sg->outputs()->size(); i++) {
      const int t = sg->outputs()->Get(i);
      if (t >= 0 && t < count) {
        if (first[t] < 0) first[t] = op_count;
        last[t] = std::max(last[t], op_count);
      }
    }
  }
}


struct Dev20ShadowBuffer {
  const char *kind{nullptr};
  int source_index{-1};
  int node{-1};
  size_t raw_bytes{0};
  size_t aligned_bytes{0};
  int first{-1};
  int last{-1};
  int planner_index{-1};
};

static void dev43_run_shadow_greedy_(TfLiteContext *context,
                                     const int *first,
                                     const int *last,
                                     int tensor_count) {
  if (context == nullptr || context->GetEvalTensor == nullptr ||
      dev43_active_model_ == nullptr || dev43_active_model_->subgraphs() == nullptr ||
      dev43_active_model_->subgraphs()->size() == 0) {
    ESP_LOGE(TAG, "Cry dev46 SHADOW planner: context/model unavailable");
    return;
  }

  const auto *sg = dev43_active_model_->subgraphs()->Get(0);
  if (sg == nullptr || sg->tensors() == nullptr) return;

  constexpr int MAX_SHADOW_BUFFERS = 96;
  Dev20ShadowBuffer buffers[MAX_SHADOW_BUFFERS]{};
  int buffer_count = 0;
  int tensor_buffers = 0;
  int scratch_buffers = 0;

  for (int i = 0; i < tensor_count && buffer_count < MAX_SHADOW_BUFFERS; i++) {
    uint32_t buffer_index = 0;
    size_t const_bytes = 0;
    if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i,
                                     &buffer_index, &const_bytes)) continue;
    const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
    const size_t raw_bytes = dev43_eval_tensor_bytes_(et);
    if (raw_bytes == 0 || first[i] < 0 || last[i] < 0) continue;
    auto &b = buffers[buffer_count++];
    b.kind = "tensor";
    b.source_index = i;
    b.node = -1;
    b.raw_bytes = raw_bytes;
    b.aligned_bytes = tflite::AlignSizeUp(raw_bytes, tflite::MicroArenaBufferAlignment());
    b.first = first[i];
    b.last = last[i];
    tensor_buffers++;
  }

  for (size_t i = 0; i < dev43_scratch_count_ && buffer_count < MAX_SHADOW_BUFFERS; i++) {
    const auto &r = dev43_scratch_requests_[i];
    if (r.status != kTfLiteOk || r.bytes == 0 || r.index < 0) continue;
    auto &b = buffers[buffer_count++];
    b.kind = "scratch";
    b.source_index = r.index;
    b.node = static_cast<int>(r.node);
    b.raw_bytes = r.bytes;
    b.aligned_bytes = tflite::AlignSizeUp(r.bytes, tflite::MicroArenaBufferAlignment());
    b.first = static_cast<int>(r.node);
    b.last = static_cast<int>(r.node);
    scratch_buffers++;
  }

  const size_t per_buffer = tflite::GreedyMemoryPlanner::per_buffer_size();
  const size_t planner_scratch_bytes = std::max<size_t>(4096, per_buffer * (buffer_count + 8));
  auto *planner_scratch = static_cast<uint8_t *>(
      heap_caps_malloc(planner_scratch_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (planner_scratch == nullptr) {
    ESP_LOGE(TAG,
             "Cry dev46 SHADOW planner: scratch allocation failed bytes=%u buffers=%d",
             (unsigned) planner_scratch_bytes, buffer_count);
    return;
  }

  // dev46 intentionally yields before the diagnostic planner. dev19 proved that
  // synchronous log bursts during AllocateTensors() can starve lwIP/API on CPU1.
  taskYIELD();

  tflite::GreedyMemoryPlanner planner;
  const TfLiteStatus init_status = planner.Init(planner_scratch,
                                                static_cast<int>(planner_scratch_bytes));
  if (init_status != kTfLiteOk) {
    ESP_LOGE(TAG,
             "Cry dev46 SHADOW init failed: buffers=%d scratch=%u B status=%d",
             buffer_count, (unsigned) planner_scratch_bytes, (int) init_status);
    heap_caps_free(planner_scratch);
    return;
  }

  int failed_index = -1;
  TfLiteStatus failed_status = kTfLiteOk;
  for (int i = 0; i < buffer_count; i++) {
    auto &b = buffers[i];
    const TfLiteStatus st = planner.AddBuffer(static_cast<int>(b.aligned_bytes),
                                              b.first, b.last);
    b.planner_index = i;
    if (st != kTfLiteOk) {
      failed_index = i;
      failed_status = st;
      break;
    }
  }

  size_t maximum = 0;
  bool overlap = false;
  int offset_failures = 0;
  int min_offset = INT32_MAX;
  int max_end = 0;
  if (failed_index < 0) {
    maximum = planner.GetMaximumMemorySize();
    overlap = planner.DoAnyBuffersOverlap();
    for (int i = 0; i < buffer_count; i++) {
      int offset = -1;
      const TfLiteStatus st = planner.GetOffsetForBuffer(i, &offset);
      if (st != kTfLiteOk || offset < 0) {
        offset_failures++;
        continue;
      }
      min_offset = std::min(min_offset, offset);
      max_end = std::max(max_end, offset + static_cast<int>(buffers[i].aligned_bytes));
    }
  }

  taskYIELD();

  if (failed_index >= 0) {
    const auto &b = buffers[failed_index];
    ESP_LOGE(TAG,
             "Cry dev46 SHADOW FAILED: buffers=%d tensor=%d scratch=%d at=%d kind=%s source=%d node=%d bytes=%u life=[%d,%d] status=%d",
             buffer_count, tensor_buffers, scratch_buffers, failed_index,
             b.kind ? b.kind : "?", b.source_index, b.node,
             (unsigned) b.aligned_bytes, b.first, b.last, (int) failed_status);
  } else {
    ESP_LOGI(TAG,
             "Cry dev46 SHADOW result: buffers=%d tensor=%d scratch=%d maximum=%u B overlap=%s offset_failures=%d span=[%d,%d] planner_scratch=%u B",
             buffer_count, tensor_buffers, scratch_buffers, (unsigned) maximum,
             overlap ? "YES" : "NO", offset_failures,
             min_offset == INT32_MAX ? -1 : min_offset, max_end,
             (unsigned) planner_scratch_bytes);
  }

  heap_caps_free(planner_scratch);
}

static void dev43_log_runtime_eval_tensors_(TfLiteContext *context,
                                            const char *phase) {
  if (context == nullptr || context->GetEvalTensor == nullptr ||
      dev43_active_model_ == nullptr || dev43_active_model_->subgraphs() == nullptr ||
      dev43_active_model_->subgraphs()->size() == 0) {
    ESP_LOGE(TAG, "Cry dev46 runtime tensor audit (%s): context/model unavailable",
             phase ? phase : "unknown");
    return;
  }
  const auto *sg = dev43_active_model_->subgraphs()->Get(0);
  if (sg == nullptr || sg->tensors() == nullptr) return;
  const int count = (int) sg->tensors()->size();
  constexpr int MAX_TENSORS = 64;
  if (count > MAX_TENSORS) {
    ESP_LOGE(TAG, "Cry dev46 runtime tensor audit: tensor count=%d exceeds cap=%d",
             count, MAX_TENSORS);
    return;
  }

  int first[MAX_TENSORS];
  int last[MAX_TENSORS];
  dev43_compute_lifetimes_(sg, first, last, count);

  size_t planned_sum = 0;
  size_t planned_peak_naive = 0;
  size_t largest_tensor = 0;
  int planned_count = 0;
  int largest_tensor_index = -1;
  const int op_count = sg->operators() ? (int) sg->operators()->size() : 0;

  for (int i = 0; i < count; i++) {
    uint32_t buffer_index = 0;
    size_t const_bytes = 0;
    if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i,
                                     &buffer_index, &const_bytes)) continue;
    const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
    const size_t bytes = dev43_eval_tensor_bytes_(et);
    if (bytes == 0) continue;
    planned_sum += bytes;
    planned_count++;
    if (bytes > largest_tensor) {
      largest_tensor = bytes;
      largest_tensor_index = i;
    }
  }

  for (int step = 0; step <= op_count; step++) {
    size_t active = 0;
    for (int i = 0; i < count; i++) {
      uint32_t bi = 0;
      size_t cb = 0;
      if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i, &bi, &cb)) continue;
      if (first[i] < 0 || last[i] < 0 || step < first[i] || step > last[i]) continue;
      const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
      active += dev43_eval_tensor_bytes_(et);
    }
    planned_peak_naive = std::max(planned_peak_naive, active);
  }

  ESP_LOGI(TAG,
           "Cry dev46 planner input: tensors=%d nonconst_sum=%u B naive_peak=%u B largest=tensor[%d]/%u B scratch=%u req/%u B max=%u B overflow=%u",
           planned_count, (unsigned) planned_sum, (unsigned) planned_peak_naive,
           largest_tensor_index, (unsigned) largest_tensor,
           (unsigned) dev43_scratch_count_, (unsigned) dev43_scratch_total_bytes_,
           (unsigned) dev43_scratch_max_bytes_, (unsigned) dev43_scratch_overflow_);

  // dev46: deterministic, logger-safe audit. dev26 proved that the complete
  // candidate set is small, but packing many records into one long log line
  // caused truncation exactly where scratch records should have appeared.
  // Emit fixed-size groups and add the per-node tensor+scratch working-set peak
  // without changing the allocator, kernels, resolver, arena, or quarantine.
  uint32_t audit_hash = 2166136261u;  // FNV-1a over canonical record strings.
  size_t aligned_tensor_sum = 0;
  size_t aligned_scratch_sum = 0;
  unsigned emitted_tensors = 0;
  unsigned emitted_scratch = 0;

  auto hash_bytes = [&](const char *text) {
    if (text == nullptr) return;
    for (const unsigned char *q = reinterpret_cast<const unsigned char *>(text); *q; ++q) {
      audit_hash ^= *q;
      audit_hash *= 16777619u;
    }
  };

  constexpr unsigned DEV43_RECORDS_PER_LINE = 3;
  constexpr size_t DEV43_LINE_BYTES = 256;
  char line[DEV43_LINE_BYTES];
  char record[96];
  size_t line_used = 0;
  unsigned line_records = 0;
  unsigned tensor_line_no = 0;
  unsigned scratch_line_no = 0;

  auto reset_line = [&]() {
    line_used = 0;
    line_records = 0;
    line[0] = '\0';
  };
  auto append_short = [&](const char *text) {
    if (text == nullptr || text[0] == '\0') return;
    const size_t n = strlen(text);
    if (line_used != 0 && line_used + 1 < DEV43_LINE_BYTES) line[line_used++] = ' ';
    const size_t room = DEV43_LINE_BYTES - 1 - line_used;
    const size_t copy_n = std::min(n, room);
    memcpy(line + line_used, text, copy_n);
    line_used += copy_n;
    line[line_used] = '\0';
    line_records++;
  };
  auto flush_tensor_line = [&]() {
    if (line_records == 0) return;
    ESP_LOGI(TAG, "Cry dev46 ALLOC_T[%u]: %s", tensor_line_no++, line);
    reset_line();
  };
  auto flush_scratch_line = [&]() {
    if (line_records == 0) return;
    ESP_LOGI(TAG, "Cry dev46 ALLOC_S[%u]: %s", scratch_line_no++, line);
    reset_line();
  };

  reset_line();
  for (int i = 0; i < count; i++) {
    uint32_t buffer_index = 0;
    size_t const_bytes = 0;
    if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i,
                                     &buffer_index, &const_bytes)) continue;
    const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
    const size_t bytes = dev43_eval_tensor_bytes_(et);
    if (bytes == 0 || first[i] < 0 || last[i] < 0) continue;
    const size_t aligned =
        tflite::AlignSizeUp(bytes, tflite::MicroArenaBufferAlignment());
    aligned_tensor_sum += aligned;
    emitted_tensors++;
    snprintf(record, sizeof(record), "T:%d:%u:%u:%d:%d",
             i, (unsigned) bytes, (unsigned) aligned, first[i], last[i]);
    hash_bytes(record);
    append_short(record);
    if (line_records >= DEV43_RECORDS_PER_LINE) flush_tensor_line();
  }
  flush_tensor_line();

  reset_line();
  for (size_t i = 0; i < dev43_scratch_count_; i++) {
    const auto &r = dev43_scratch_requests_[i];
    const size_t aligned =
        tflite::AlignSizeUp(r.bytes, tflite::MicroArenaBufferAlignment());
    aligned_scratch_sum += aligned;
    emitted_scratch++;
    snprintf(record, sizeof(record), "S:%u:%d:%u:%u:%u:%d",
             (unsigned) i, r.index, (unsigned) r.node, (unsigned) r.bytes,
             (unsigned) aligned, (int) r.status);
    hash_bytes(record);
    append_short(record);
    if (line_records >= DEV43_RECORDS_PER_LINE) flush_scratch_line();
  }
  flush_scratch_line();

  // Compute the exact candidate working set at each execution step using the
  // same aligned sizes that the arena planner sees. Scratch buffers requested
  // by a node are live only for that node, so multiple requests for one node
  // are summed for the same step.
  size_t combined_peak = 0;
  size_t combined_peak_tensor = 0;
  size_t combined_peak_scratch = 0;
  int combined_peak_step = -1;
  unsigned peak_line_no = 0;
  reset_line();
  for (int step = 0; step <= op_count; step++) {
    size_t tensor_active_aligned = 0;
    size_t scratch_active_aligned = 0;

    for (int i = 0; i < count; i++) {
      uint32_t bi = 0;
      size_t cb = 0;
      if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i, &bi, &cb)) continue;
      if (first[i] < 0 || last[i] < 0 || step < first[i] || step > last[i]) continue;
      const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
      const size_t bytes = dev43_eval_tensor_bytes_(et);
      if (bytes == 0) continue;
      tensor_active_aligned +=
          tflite::AlignSizeUp(bytes, tflite::MicroArenaBufferAlignment());
    }

    for (size_t i = 0; i < dev43_scratch_count_; i++) {
      const auto &r = dev43_scratch_requests_[i];
      if ((int) r.node != step) continue;
      scratch_active_aligned +=
          tflite::AlignSizeUp(r.bytes, tflite::MicroArenaBufferAlignment());
    }

    const size_t combined = tensor_active_aligned + scratch_active_aligned;
    if (combined > combined_peak) {
      combined_peak = combined;
      combined_peak_tensor = tensor_active_aligned;
      combined_peak_scratch = scratch_active_aligned;
      combined_peak_step = step;
    }

    snprintf(record, sizeof(record), "P:%d:T%u:S%u:C%u",
             step, (unsigned) tensor_active_aligned,
             (unsigned) scratch_active_aligned, (unsigned) combined);
    append_short(record);
    if (line_records >= DEV43_RECORDS_PER_LINE) {
      ESP_LOGI(TAG, "Cry dev46 PEAK[%u]: %s", peak_line_no++, line);
      reset_line();
    }
  }
  if (line_records != 0) {
    ESP_LOGI(TAG, "Cry dev46 PEAK[%u]: %s", peak_line_no++, line);
    reset_line();
  }

  ESP_LOGI(TAG,
           "Cry dev46 ALLOC_INFO summary: tensor=%u scratch=%u records=%u aligned_tensor_sum=%u B aligned_scratch_sum=%u B naive_tensor_peak=%u B scratch_max=%u B hash=0x%08X",
           emitted_tensors, emitted_scratch, emitted_tensors + emitted_scratch,
           (unsigned) aligned_tensor_sum, (unsigned) aligned_scratch_sum,
           (unsigned) planned_peak_naive, (unsigned) dev43_scratch_max_bytes_,
           (unsigned) audit_hash);
  ESP_LOGI(TAG,
           "Cry dev46 COMBINED_PEAK: step=%d tensor=%u B scratch=%u B combined=%u B arena=%u B",
           combined_peak_step, (unsigned) combined_peak_tensor,
           (unsigned) combined_peak_scratch, (unsigned) combined_peak,
           (unsigned) TENSOR_ARENA_BYTES);
  ESP_LOGI(TAG,
           "Cry dev46 formats: ALLOC_T=T:index:bytes:aligned:first:last; ALLOC_S=S:slot:index:node:bytes:aligned:status; PEAK=P:step:Ttensor:Sscratch:Ccombined");

  // dev46: capture ONLY scalar planner inputs here.  No heap allocation, no
  // GreedyMemoryPlanner construction and no logging-heavy replay is allowed
  // inside Prepare/AllocateTensors.  This is the key isolation change vs dev29.
  dev43_replay_count_ = 0;
  dev43_replay_ready_ = false;
  dev43_replay_combined_peak_ = combined_peak;

  for (int i = 0; i < count && dev43_replay_count_ < DEV43_MAX_REPLAY_RECORDS; i++) {
    uint32_t bi = 0;
    size_t cb = 0;
    if (dev43_model_tensor_is_const_(sg, dev43_active_model_, i, &bi, &cb)) continue;
    if (first[i] < 0 || last[i] < 0) continue;
    const TfLiteEvalTensor *et = context->GetEvalTensor(context, i);
    const size_t bytes = dev43_eval_tensor_bytes_(et);
    if (bytes == 0) continue;
    auto &r = dev43_replay_records_[dev43_replay_count_++];
    r.bytes = (int) tflite::AlignSizeUp(bytes, tflite::MicroArenaBufferAlignment());
    r.first = first[i];
    r.last = last[i];
    r.source = i;
    r.scratch = false;
  }
  for (size_t i = 0; i < dev43_scratch_count_ && dev43_replay_count_ < DEV43_MAX_REPLAY_RECORDS; i++) {
    const auto &sr = dev43_scratch_requests_[i];
    auto &r = dev43_replay_records_[dev43_replay_count_++];
    r.bytes = (int) tflite::AlignSizeUp(sr.bytes, tflite::MicroArenaBufferAlignment());
    r.first = (int) sr.node;
    r.last = (int) sr.node;
    r.source = (int) i;
    r.scratch = true;
  }
  dev43_replay_ready_ =
      dev43_replay_count_ == (size_t) (emitted_tensors + emitted_scratch);
  ESP_LOGI(TAG,
           "Cry dev46 REPLAY_SNAPSHOT: ready=%s records=%u expected=%u combined_peak=%u B; native replay deferred until AllocateTensors returns",
           dev43_replay_ready_ ? "YES" : "NO", (unsigned) dev43_replay_count_,
           (unsigned) (emitted_tensors + emitted_scratch),
           (unsigned) dev43_replay_combined_peak_);

}

static void dev43_run_native_replay_snapshot_() {
  dev43_mark_(0xF200u, (uint32_t) dev43_replay_count_);
  if (!dev43_replay_ready_ || dev43_replay_count_ == 0) {
    ESP_LOGE(TAG,
             "Cry dev46 NATIVE_REPLAY FAIL: snapshot_not_ready records=%u",
             (unsigned) dev43_replay_count_);
    dev43_mark_(0xF2FFu, 1u);
    return;
  }

  constexpr size_t DEV43_NATIVE_PLANNER_SCRATCH = 4096;
  auto *planner_scratch = static_cast<unsigned char *>(heap_caps_malloc(
      DEV43_NATIVE_PLANNER_SCRATCH, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (planner_scratch == nullptr) {
    ESP_LOGE(TAG,
             "Cry dev46 NATIVE_REPLAY FAIL: planner scratch allocation failed (%u B)",
             (unsigned) DEV43_NATIVE_PLANNER_SCRATCH);
    dev43_mark_(0xF2FFu, 2u);
    return;
  }

  tflite::GreedyMemoryPlanner planner;
  TfLiteStatus status = planner.Init(planner_scratch, DEV43_NATIVE_PLANNER_SCRATCH);
  int added = 0;
  int failed_record = -1;
  for (size_t i = 0; i < dev43_replay_count_ && status == kTfLiteOk; i++) {
    const auto &r = dev43_replay_records_[i];
    const TfLiteStatus st = planner.AddBuffer(r.bytes, r.first, r.last);
    if (st != kTfLiteOk) {
      status = st;
      failed_record = (int) i;
      ESP_LOGE(TAG,
               "Cry dev46 NATIVE_REPLAY AddBuffer FAIL: record=%u kind=%s source=%d bytes=%d life=[%d,%d] status=%d",
               (unsigned) i, r.scratch ? "scratch" : "tensor", r.source,
               r.bytes, r.first, r.last, (int) st);
      break;
    }
    added++;
  }

  if (status == kTfLiteOk) {
    const size_t maximum = planner.GetMaximumMemorySize();
    const bool overlap = planner.DoAnyBuffersOverlap();
    int offset_failures = 0;
    int max_end = 0;
    for (int i = 0; i < planner.GetBufferCount(); i++) {
      int offset = -1;
      if (planner.GetOffsetForBuffer(i, &offset) != kTfLiteOk) {
        offset_failures++;
        continue;
      }
      const int end = offset + dev43_replay_records_[i].bytes;
      if (end > max_end) max_end = end;
    }
    ESP_LOGI(TAG,
             "Cry dev46 NATIVE_REPLAY PASS: buffers=%d records=%u maximum=%u B span_end=%d overlap=%s offset_failures=%d planner_scratch=%u B arena=%u B",
             planner.GetBufferCount(), (unsigned) dev43_replay_count_,
             (unsigned) maximum, max_end, overlap ? "YES" : "NO",
             offset_failures, (unsigned) DEV43_NATIVE_PLANNER_SCRATCH,
             (unsigned) TENSOR_ARENA_BYTES);
    ESP_LOGI(TAG,
             "Cry dev46 NATIVE_REPLAY VERDICT: native_max=%u B manual_combined_peak=%u B; PASS + AllocateTensors status=1 isolates the fault to AllocationInfoBuilder/native handoff/commit around FinishModelAllocation, not model capacity or Greedy placement",
             (unsigned) maximum, (unsigned) dev43_replay_combined_peak_);
    dev43_mark_(0xF201u, (uint32_t) maximum);
  } else {
    ESP_LOGE(TAG,
             "Cry dev46 NATIVE_REPLAY FAIL: status=%d added=%d failed_record=%d records=%u",
             (int) status, added, failed_record, (unsigned) dev43_replay_count_);
    dev43_mark_(0xF2FFu, (uint32_t) (failed_record + 16));
  }

  heap_caps_free(planner_scratch);
}

static void dev43_reset_scratch_audit_() {
  dev43_scratch_count_ = 0;
  dev43_scratch_total_bytes_ = 0;
  dev43_scratch_max_bytes_ = 0;
  dev43_scratch_overflow_ = 0;
  dev43_scratch_current_node_ = 0xFFFFFFFFu;
  dev43_scratch_current_op_ = "none";
  dev43_orig_scratch_request_ = nullptr;
}

static void dev43_log_simple_arena_(const char *phase) {
  if (dev43_active_interpreter_ == nullptr) return;
  const auto &rec = dev43_active_interpreter_->GetMicroAllocator();
  const auto *arena = rec.GetSimpleMemoryAllocator();
  if (arena == nullptr) {
    ESP_LOGI(TAG, "Cry dev46 arena checkpoint (%s): unavailable",
             phase ? phase : "unknown");
    return;
  }
  ESP_LOGI(TAG,
           "Cry dev46 arena checkpoint (%s): used=%u B head=%u B tail=%u B free_gap=%d B",
           phase ? phase : "unknown",
           (unsigned) arena->GetUsedBytes(),
           (unsigned) arena->GetNonPersistentUsedBytes(),
           (unsigned) arena->GetPersistentUsedBytes(),
           (int) TENSOR_ARENA_BYTES -
               (int) arena->GetNonPersistentUsedBytes() -
               (int) arena->GetPersistentUsedBytes());
}

static TfLiteStatus dev43_scratch_request_(TfLiteContext *context,
                                           size_t bytes,
                                           int *buffer_idx) {
  int local_index = -1;
  int *idx_out = buffer_idx != nullptr ? buffer_idx : &local_index;
  TfLiteStatus status = kTfLiteError;
  if (dev43_orig_scratch_request_ != nullptr)
    status = dev43_orig_scratch_request_(context, bytes, idx_out);

  const int assigned_index = *idx_out;
  if (dev43_scratch_count_ < DEV43_MAX_SCRATCH_REQUESTS) {
    auto &r = dev43_scratch_requests_[dev43_scratch_count_++];
    r.node = dev43_scratch_current_node_;
    r.op = dev43_scratch_current_op_;
    r.bytes = bytes;
    r.index = assigned_index;
    r.status = status;
  } else {
    dev43_scratch_overflow_++;
  }
  dev43_scratch_total_bytes_ += bytes;
  dev43_scratch_max_bytes_ = std::max(dev43_scratch_max_bytes_, bytes);

  return status;
}

static void dev43_log_scratch_summary_(const char *phase) {
  ESP_LOGI(TAG,
           "Cry dev46 SCRATCH summary (%s): requests=%u total=%u B max=%u B overflow=%u",
           phase ? phase : "unknown", (unsigned) dev43_scratch_count_,
           (unsigned) dev43_scratch_total_bytes_,
           (unsigned) dev43_scratch_max_bytes_,
           (unsigned) dev43_scratch_overflow_);
}

static void dev43_reset_stage_counters_() {
  dev43_init_seq_ = 0;
  dev43_prepare_seq_ = 0;
  dev43_reset_scratch_audit_();
}

static void *dev43_call_init_(const char *op_name, Dev15InitFn original,
                             TfLiteContext *context, const char *buffer,
                             size_t length) {
  (void) op_name;
  const uint32_t seq = dev43_init_seq_++;
  dev43_mark_(0xA100u | (seq & 0xFFu), seq);
  void *result = original != nullptr ? original(context, buffer, length) : nullptr;
  dev43_mark_(0xA180u | (seq & 0x7Fu), seq);
  return result;
}

static void dev43_log_conv_contract_(TfLiteContext *context, TfLiteNode *node,
                                      bool depthwise, uint32_t graph_node,
                                      size_t scratch_begin);

static TfLiteStatus dev43_call_prepare_(const char *op_name,
                                      Dev15PrepareFn original,
                                      TfLiteContext *context,
                                      TfLiteNode *node) {
  const uint32_t seq = dev43_prepare_seq_++;
  const TfLiteStatus status =
      original != nullptr ? original(context, node) : kTfLiteOk;
  if (status != kTfLiteOk) {
    ESP_LOGE(TAG, "Cry dev46 PREPARE[%u] FAILED op=%s status=%d",
             (unsigned) seq, op_name ? op_name : "?", (int) status);
  }
  return status;
}

static TfLiteStatus dev43_conv_prepare_(TfLiteContext *context, TfLiteNode *node) {
  return dev43_call_prepare_("CONV_2D", dev43_orig_conv_prepare_, context, node);
}

static TfLiteStatus dev43_depthwise_prepare_(TfLiteContext *context,
                                            TfLiteNode *node) {
  return dev43_call_prepare_("DEPTHWISE_CONV_2D",
                            dev43_orig_depthwise_prepare_, context, node);
}

static TfLiteStatus dev43_conv_invoke_(TfLiteContext *context, TfLiteNode *node) {
  return dev43_orig_conv_invoke_ != nullptr ? dev43_orig_conv_invoke_(context, node) : kTfLiteError;
}

static TfLiteStatus dev43_depthwise_invoke_(TfLiteContext *context, TfLiteNode *node) {
  return dev43_orig_depthwise_invoke_ != nullptr ? dev43_orig_depthwise_invoke_(context, node) : kTfLiteError;
}

static void dev43_format_runtime_dims_(const TfLiteIntArray *dims,
                                      char *dst, size_t dst_size) {
  if (dst == nullptr || dst_size == 0) return;
  if (dims == nullptr) {
    std::snprintf(dst, dst_size, "<null>");
    return;
  }
  size_t used = 0;
  int n = std::snprintf(dst, dst_size, "[");
  if (n < 0) return;
  used = std::min<size_t>((size_t) n, dst_size - 1);
  for (int i = 0; i < dims->size && used + 2 < dst_size; i++) {
    n = std::snprintf(dst + used, dst_size - used, "%s%d",
                      i ? "," : "", dims->data[i]);
    if (n < 0) break;
    used += std::min<size_t>((size_t) n, dst_size - used - 1);
  }
  if (used + 2 <= dst_size)
    std::snprintf(dst + used, dst_size - used, "]");
}

static void dev43_log_conv_tensor_short_(const char *label, const TfLiteTensor *tensor,
                                          char *dst, size_t dst_size) {
  if (dst == nullptr || dst_size == 0) return;
  if (tensor == nullptr) {
    std::snprintf(dst, dst_size, "%s=<null>", label ? label : "tensor");
    return;
  }
  char dims[64];
  dev43_format_runtime_dims_(tensor->dims, dims, sizeof(dims));
  int scales = 0;
  int qdim = -1;
  if (tensor->quantization.type == kTfLiteAffineQuantization &&
      tensor->quantization.params != nullptr) {
    const auto *aff = reinterpret_cast<const TfLiteAffineQuantization *>(tensor->quantization.params);
    qdim = aff->quantized_dimension;
    if (aff->scale != nullptr) scales = aff->scale->size;
  }
  std::snprintf(dst, dst_size, "%s=%s/t%d/%uB/s=%.7g/zp=%ld/qs=%d/qd=%d",
                label ? label : "tensor", dims, (int) tensor->type,
                (unsigned) tensor->bytes, (double) tensor->params.scale,
                (long) tensor->params.zero_point, scales, qdim);
}

static void dev43_log_conv_contract_(TfLiteContext *context, TfLiteNode *node,
                                      bool depthwise, uint32_t graph_node,
                                      size_t scratch_begin) {
  if (context == nullptr || node == nullptr) return;
  tflite::MicroContext *micro_context = tflite::GetMicroContext(context);
  if (micro_context == nullptr) {
    ESP_LOGW(TAG, "Cry dev46 CONV_AUDIT node=%u op=%s MicroContext unavailable",
             (unsigned) graph_node, depthwise ? "DW" : "CONV");
    return;
  }

  TfLiteTensor *input = node->inputs && node->inputs->size > 0
                            ? micro_context->AllocateTempInputTensor(node, 0) : nullptr;
  TfLiteTensor *filter = node->inputs && node->inputs->size > 1
                             ? micro_context->AllocateTempInputTensor(node, 1) : nullptr;
  TfLiteTensor *bias = node->inputs && node->inputs->size > 2
                           ? micro_context->AllocateTempInputTensor(node, 2) : nullptr;
  TfLiteTensor *output = node->outputs && node->outputs->size > 0
                             ? micro_context->AllocateTempOutputTensor(node, 0) : nullptr;

  char in_s[128], filt_s[128], bias_s[128], out_s[128];
  dev43_log_conv_tensor_short_("in", input, in_s, sizeof(in_s));
  dev43_log_conv_tensor_short_("filter", filter, filt_s, sizeof(filt_s));
  dev43_log_conv_tensor_short_("bias", bias, bias_s, sizeof(bias_s));
  dev43_log_conv_tensor_short_("out", output, out_s, sizeof(out_s));

  size_t scratch_bytes = 0;
  const size_t scratch_end = dev43_scratch_count_;
  for (size_t i = scratch_begin; i < scratch_end && i < DEV43_MAX_SCRATCH_REQUESTS; i++)
    scratch_bytes += dev43_scratch_requests_[i].bytes;
  const size_t scratch_reqs = scratch_end >= scratch_begin ? scratch_end - scratch_begin : 0;

  if (!depthwise) {
    const auto *p = reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);
    ESP_LOGI(TAG,
             "Cry dev46 CONV_AUDIT node=%u op=CONV %s %s %s %s stride=%dx%d dilation=%dx%d padding=%d activation=%d scratch=%u/%uB",
             (unsigned) graph_node, in_s, filt_s, bias_s, out_s,
             p ? p->stride_height : -1, p ? p->stride_width : -1,
             p ? p->dilation_height_factor : -1, p ? p->dilation_width_factor : -1,
             p ? (int) p->padding : -1, p ? (int) p->activation : -1,
             (unsigned) scratch_reqs, (unsigned) scratch_bytes);
  } else {
    const auto *p = reinterpret_cast<const TfLiteDepthwiseConvParams *>(node->builtin_data);
    ESP_LOGI(TAG,
             "Cry dev46 CONV_AUDIT node=%u op=DW %s %s %s %s stride=%dx%d dilation=%dx%d depth_mult=%d padding=%d activation=%d scratch=%u/%uB",
             (unsigned) graph_node, in_s, filt_s, bias_s, out_s,
             p ? p->stride_height : -1, p ? p->stride_width : -1,
             p ? p->dilation_height_factor : -1, p ? p->dilation_width_factor : -1,
             p ? p->depth_multiplier : -1, p ? (int) p->padding : -1,
             p ? (int) p->activation : -1,
             (unsigned) scratch_reqs, (unsigned) scratch_bytes);
  }

  if (input) micro_context->DeallocateTempTfLiteTensor(input);
  if (filter) micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias) micro_context->DeallocateTempTfLiteTensor(bias);
  if (output) micro_context->DeallocateTempTfLiteTensor(output);
}

static void dev43_log_runtime_tensor_(const char *label,
                                     const TfLiteTensor *tensor) {
  if (tensor == nullptr) {
    ESP_LOGI(TAG, "Cry dev46 FC %s: <null tensor>", label);
    return;
  }

  char dims[96];
  dev43_format_runtime_dims_(tensor->dims, dims, sizeof(dims));

  int scale_count = 0;
  int zero_point_count = 0;
  int quantized_dimension = -1;
  float scale_first = NAN;
  float scale_last = NAN;
  int32_t zp_first = 0;
  int32_t zp_last = 0;

  if (tensor->quantization.type == kTfLiteAffineQuantization &&
      tensor->quantization.params != nullptr) {
    const auto *affine = reinterpret_cast<const TfLiteAffineQuantization *>(
        tensor->quantization.params);
    quantized_dimension = affine->quantized_dimension;
    if (affine->scale != nullptr) {
      scale_count = affine->scale->size;
      if (scale_count > 0) {
        scale_first = affine->scale->data[0];
        scale_last = affine->scale->data[scale_count - 1];
      }
    }
    if (affine->zero_point != nullptr) {
      zero_point_count = affine->zero_point->size;
      if (zero_point_count > 0) {
        zp_first = affine->zero_point->data[0];
        zp_last = affine->zero_point->data[zero_point_count - 1];
      }
    }
  }

  ESP_LOGI(TAG,
           "Cry dev46 FC %s: type=%d dims=%s bytes=%u alloc=%d "
           "params(scale=%.10g zp=%ld) quant(type=%d scales=%d zps=%d qdim=%d "
           "scale_first=%.10g scale_last=%.10g zp_first=%ld zp_last=%ld)",
           label, (int) tensor->type, dims, (unsigned) tensor->bytes,
           (int) tensor->allocation_type,
           (double) tensor->params.scale, (long) tensor->params.zero_point,
           (int) tensor->quantization.type, scale_count, zero_point_count,
           quantized_dimension, (double) scale_first, (double) scale_last,
           (long) zp_first, (long) zp_last);
}


// v1.4.0-dev43 Invoke cache.  Prepare owns all full TfLiteTensor access and
// copies the tiny amount of quantization metadata needed by Invoke.  Runtime
// Invoke then touches only TfLiteEvalTensor, so it cannot leave temporary
// TfLiteTensor objects live in the MicroAllocator temp chain.
struct Dev38MeanInvokeData {
  bool ready{false};
  int32_t input_zero_point{0};
  int32_t output_zero_point{0};
  float input_scale{0.0f};
  float output_scale{0.0f};
};
static Dev38MeanInvokeData dev43_mean_invoke_data_;

static constexpr int DEV43_FC_MAX_OUTPUTS = 521;
struct Dev38FcInvokeData {
  bool ready{false};
  int32_t input_zero_point{0};
  int32_t output_zero_point{0};
  float input_scale{0.0f};
  float output_scale{0.0f};
  int32_t activation_min{-128};
  int32_t activation_max{127};
  int output_depth{0};
  float *weight_scales{nullptr};
};
static Dev38FcInvokeData dev43_fc_invoke_data_;

// ---------------------------------------------------------------------------
// v1.4.0-dev43: controlled INT8 MEAN / global-average compatibility kernel.
//
// dev10 proved that execution reaches the final FC, which strongly suggests
// native MEAN Prepare already returns kTfLiteOk.  dev43 nevertheless replaces
// this single node with a tiny, fully observable implementation.  If all 11
// Prepare callbacks now return kTfLiteOk and AllocateTensors still fails, the
// remaining suspect is FinishModelAllocation / memory planning rather than an
// operator contract.
// ---------------------------------------------------------------------------

static void dev43_log_mean_contract_(TfLiteContext *context, TfLiteNode *node) {
  const int inputs = node && node->inputs ? node->inputs->size : -1;
  const int outputs = node && node->outputs ? node->outputs->size : -1;
  ESP_LOGI(TAG,
           "Cry dev46 MEAN contract: inputs=%d outputs=%d user_data=%p builtin_data=%p",
           inputs, outputs, node ? node->user_data : nullptr,
           node ? node->builtin_data : nullptr);
  if (context == nullptr || node == nullptr) return;

  const TfLiteTensor *input = inputs > 0 ? tflite::GetInput(context, node, 0) : nullptr;
  const TfLiteTensor *axis = inputs > 1 ? tflite::GetInput(context, node, 1) : nullptr;
  TfLiteTensor *output = outputs > 0 ? tflite::GetOutput(context, node, 0) : nullptr;

  dev43_log_runtime_tensor_("MEAN input", input);
  dev43_log_runtime_tensor_("MEAN axis", axis);
  dev43_log_runtime_tensor_("MEAN output", output);

  if (axis != nullptr && axis->type == kTfLiteInt32 && axis->data.i32 != nullptr &&
      axis->dims != nullptr && axis->dims->size == 1) {
    char axes[64];
    size_t used = 0;
    int n = std::snprintf(axes, sizeof(axes), "[");
    if (n > 0) used = std::min<size_t>((size_t) n, sizeof(axes) - 1);
    const int count = axis->dims->data[0];
    for (int i = 0; i < count && used + 3 < sizeof(axes); i++) {
      n = std::snprintf(axes + used, sizeof(axes) - used, "%s%ld",
                        i ? "," : "", (long) axis->data.i32[i]);
      if (n < 0) break;
      used += std::min<size_t>((size_t) n, sizeof(axes) - used - 1);
    }
    if (used + 2 <= sizeof(axes)) std::snprintf(axes + used, sizeof(axes) - used, "]");
    ESP_LOGI(TAG, "Cry dev46 MEAN axes=%s", axes);
  }

  if (node->builtin_data != nullptr) {
    const auto *params = reinterpret_cast<const TfLiteReducerParams *>(node->builtin_data);
    ESP_LOGI(TAG, "Cry dev46 MEAN params: keep_dims=%d", (int) params->keep_dims);
  }
}

static TfLiteStatus dev43_mean_prepare_(TfLiteContext *context, TfLiteNode *node) {
  const uint32_t seq = dev43_prepare_seq_++;
  dev43_mark_(0xB900u, seq);

  if (context == nullptr || node == nullptr || node->inputs == nullptr ||
      node->outputs == nullptr || node->inputs->size != 2 ||
      node->outputs->size != 1) {
    ESP_LOGE(TAG, "Cry dev46 MEAN Prepare: invalid node contract");
    return kTfLiteError;
  }

  // dev46: full TfLiteTensor objects allocated during Prepare are TEMP arena
  // objects in TFLM.  Allocate them through MicroContext and always release
  // them before returning; leaving even one live makes the allocator's
  // subsequent ResetTempAllocations() report kTfLiteError.
  tflite::MicroContext *micro_context = tflite::GetMicroContext(context);
  if (micro_context == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 MEAN Prepare: MicroContext unavailable");
    return kTfLiteError;
  }

  TfLiteTensor *input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor *axis = micro_context->AllocateTempInputTensor(node, 1);
  TfLiteTensor *output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteStatus status = kTfLiteOk;

  if (input == nullptr || axis == nullptr || output == nullptr ||
      input->type != kTfLiteInt8 || axis->type != kTfLiteInt32 ||
      output->type != kTfLiteInt8 || input->dims == nullptr ||
      axis->dims == nullptr || output->dims == nullptr || axis->data.i32 == nullptr ||
      input->dims->size != 4 || axis->dims->size != 1 ||
      output->dims->size != 2) {
    ESP_LOGE(TAG, "Cry dev46 MEAN Prepare: unsupported tensor types/dimensions");
    status = kTfLiteError;
  }

  if (status == kTfLiteOk) {
    const int batches = input->dims->data[0];
    const int height = input->dims->data[1];
    const int width = input->dims->data[2];
    const int channels = input->dims->data[3];
    const int axis_count = axis->dims->data[0];
    const bool axes_ok = axis_count == 2 &&
                         ((axis->data.i32[0] == 1 && axis->data.i32[1] == 2) ||
                          (axis->data.i32[0] == 2 && axis->data.i32[1] == 1));
    if (batches != 1 || height <= 0 || width <= 0 || channels <= 0 || !axes_ok ||
        output->dims->data[0] != batches || output->dims->data[1] != channels) {
      ESP_LOGE(TAG,
               "Cry dev46 MEAN Prepare: expected [1,H,W,C] reduce axes {1,2} -> [1,C]; input=[%d,%d,%d,%d] axis_count=%d output=[%d,%d]",
               batches, height, width, channels, axis_count,
               output->dims->data[0], output->dims->data[1]);
      status = kTfLiteError;
    }
  }

  if (status == kTfLiteOk) {
    const auto *params = reinterpret_cast<const TfLiteReducerParams *>(node->builtin_data);
    if (params == nullptr || params->keep_dims) {
      ESP_LOGE(TAG, "Cry dev46 MEAN Prepare: keep_dims must be false");
      status = kTfLiteError;
    } else if (!(input->params.scale > 0.0f) || !(output->params.scale > 0.0f)) {
      ESP_LOGE(TAG, "Cry dev46 MEAN Prepare: invalid quantization scales");
      status = kTfLiteError;
    }
  }

  if (status == kTfLiteOk) {
    dev43_mean_invoke_data_.input_zero_point = input->params.zero_point;
    dev43_mean_invoke_data_.output_zero_point = output->params.zero_point;
    dev43_mean_invoke_data_.input_scale = input->params.scale;
    dev43_mean_invoke_data_.output_scale = output->params.scale;
    dev43_mean_invoke_data_.ready = true;
  } else {
    dev43_mean_invoke_data_.ready = false;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(output);

  dev43_mark_(status == kTfLiteOk ? 0xB980u : 0xB9FFu, seq);
  return status;
}

static TfLiteStatus dev43_mean_invoke_impl_(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  if (!dev43_mean_invoke_data_.ready || input == nullptr || output == nullptr ||
      input->data.int8 == nullptr || output->data.int8 == nullptr ||
      input->dims == nullptr || output->dims == nullptr || input->dims->size != 4 ||
      output->dims->size != 2) {
    ESP_LOGE(TAG, "Cry dev46 MEAN Invoke: eval tensor/cache unavailable");
    return kTfLiteError;
  }

  const int batches = input->dims->data[0];
  const int height = input->dims->data[1];
  const int width = input->dims->data[2];
  const int channels = input->dims->data[3];
  const int spatial = height * width;
  if (batches != 1 || spatial <= 0 || channels <= 0 ||
      output->dims->data[0] != 1 || output->dims->data[1] != channels)
    return kTfLiteError;

  const int32_t in_zp = dev43_mean_invoke_data_.input_zero_point;
  const int32_t out_zp = dev43_mean_invoke_data_.output_zero_point;
  const double multiplier =
      (double) dev43_mean_invoke_data_.input_scale /
      ((double) spatial * (double) dev43_mean_invoke_data_.output_scale);
  if (!(multiplier > 0.0)) return kTfLiteError;

  const int8_t *in = input->data.int8;
  int8_t *out = output->data.int8;
  for (int c = 0; c < channels; c++) {
    int32_t sum_centered = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int idx = ((y * width) + x) * channels + c;
        sum_centered += (int32_t) in[idx] - in_zp;
      }
    }
    int32_t q = (int32_t) std::lround((double) sum_centered * multiplier) + out_zp;
    q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
    out[c] = static_cast<int8_t>(q);
  }

  static bool first_invoke = true;
  if (first_invoke) {
    first_invoke = false;
    ESP_LOGI(TAG,
             "Cry dev46 custom MEAN first EvalTensor-only Invoke OK: spatial=%d channels=%d",
             spatial, channels);
  }
  return kTfLiteOk;
}

static TfLiteStatus dev43_mean_invoke_(TfLiteContext *context, TfLiteNode *node) {
  return dev43_mean_invoke_impl_(context, node);
}

static void dev43_log_fc_contract_(TfLiteContext *context, TfLiteNode *node) {
  const int inputs = node && node->inputs ? node->inputs->size : -1;
  const int outputs = node && node->outputs ? node->outputs->size : -1;
  ESP_LOGI(TAG, "Cry dev46 FC contract: inputs=%d outputs=%d user_data=%p builtin_data=%p",
           inputs, outputs, node ? node->user_data : nullptr,
           node ? node->builtin_data : nullptr);

  if (context == nullptr || node == nullptr) return;

  const TfLiteTensor *input = inputs > 0 ? tflite::GetInput(context, node, 0) : nullptr;
  const TfLiteTensor *weights = inputs > 1 ? tflite::GetInput(context, node, 1) : nullptr;
  const TfLiteTensor *bias = inputs > 2 ? tflite::GetInput(context, node, 2) : nullptr;
  TfLiteTensor *output = outputs > 0 ? tflite::GetOutput(context, node, 0) : nullptr;

  dev43_log_runtime_tensor_("input", input);
  dev43_log_runtime_tensor_("weights", weights);
  dev43_log_runtime_tensor_("bias", bias);
  dev43_log_runtime_tensor_("output", output);

  if (node->builtin_data != nullptr) {
    const auto *params = reinterpret_cast<const TfLiteFullyConnectedParams *>(
        node->builtin_data);
    ESP_LOGI(TAG,
             "Cry dev46 FC params: activation=%d weights_format=%d keep_num_dims=%d "
             "asymmetric_quantize_inputs=%d",
             (int) params->activation, (int) params->weights_format,
             (int) params->keep_num_dims,
             (int) params->asymmetric_quantize_inputs);
  }

  if (input != nullptr && weights != nullptr && bias != nullptr) {
    const double expected_bias_scale =
        (double) input->params.scale * (double) weights->params.scale;
    const double actual_bias_scale = (double) bias->params.scale;
    const double ratio = expected_bias_scale != 0.0
                             ? actual_bias_scale / expected_bias_scale
                             : NAN;
    ESP_LOGI(TAG,
             "Cry dev46 FC quant check: input_scale*weight_scale=%.12g "
             "bias_scale=%.12g ratio=%.9g weights_zp=%ld",
             expected_bias_scale, actual_bias_scale, ratio,
             (long) weights->params.zero_point);
  }
}

static TfLiteStatus dev43_fc_prepare_(TfLiteContext *context, TfLiteNode *node) {
  // dev46 keeps the dev35 custom per-channel FC contract, but fixes the
  // lifetime of full TfLiteTensor objects used during Prepare.
  const uint32_t seq = dev43_prepare_seq_++;
  dev43_mark_(0xBA00u, seq);
  if (context == nullptr || node == nullptr || node->inputs == nullptr ||
      node->outputs == nullptr || node->inputs->size != 3 ||
      node->outputs->size != 1) {
    ESP_LOGE(TAG, "Cry dev46 FC Prepare: invalid node contract");
    return kTfLiteError;
  }

  tflite::MicroContext *micro_context = tflite::GetMicroContext(context);
  if (micro_context == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 FC Prepare: MicroContext unavailable");
    return kTfLiteError;
  }

  TfLiteTensor *input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor *weights = micro_context->AllocateTempInputTensor(node, 1);
  TfLiteTensor *bias = micro_context->AllocateTempInputTensor(node, 2);
  TfLiteTensor *output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteStatus status = kTfLiteOk;

  if (input == nullptr || weights == nullptr || bias == nullptr || output == nullptr ||
      input->type != kTfLiteInt8 || weights->type != kTfLiteInt8 ||
      bias->type != kTfLiteInt32 || output->type != kTfLiteInt8 ||
      input->dims == nullptr || weights->dims == nullptr || bias->dims == nullptr ||
      output->dims == nullptr || input->dims->size != 2 || weights->dims->size != 2 ||
      bias->dims->size != 1 || output->dims->size != 2) {
    ESP_LOGE(TAG, "Cry dev46 FC Prepare: unsupported tensor types/dimensions");
    status = kTfLiteError;
  }

  int output_depth = 0;
  if (status == kTfLiteOk) {
    const int batches = input->dims->data[0];
    const int accum_depth = input->dims->data[1];
    output_depth = weights->dims->data[0];
    if (batches != 1 || accum_depth != weights->dims->data[1] ||
        bias->dims->data[0] != output_depth || output->dims->data[0] != batches ||
        output->dims->data[1] != output_depth) {
      ESP_LOGE(TAG,
               "Cry dev46 FC Prepare: shape mismatch input=[%d,%d] weights=[%d,%d] bias=%d output=[%d,%d]",
               batches, accum_depth, weights->dims->data[0], weights->dims->data[1],
               bias->dims->data[0], output->dims->data[0], output->dims->data[1]);
      status = kTfLiteError;
    }
  }

  const TfLiteAffineQuantization *wq = nullptr;
  const TfLiteAffineQuantization *bq = nullptr;
  if (status == kTfLiteOk) {
    if (weights->quantization.type != kTfLiteAffineQuantization ||
        weights->quantization.params == nullptr ||
        bias->quantization.type != kTfLiteAffineQuantization ||
        bias->quantization.params == nullptr) {
      ESP_LOGE(TAG, "Cry dev46 FC Prepare: affine quantization required");
      status = kTfLiteError;
    } else {
      wq = reinterpret_cast<const TfLiteAffineQuantization *>(weights->quantization.params);
      bq = reinterpret_cast<const TfLiteAffineQuantization *>(bias->quantization.params);
    }
  }

  if (status == kTfLiteOk) {
    if (wq->scale == nullptr || wq->zero_point == nullptr || bq->scale == nullptr ||
        bq->zero_point == nullptr || wq->scale->size != output_depth ||
        wq->zero_point->size != output_depth || bq->scale->size != output_depth ||
        bq->zero_point->size != output_depth || wq->quantized_dimension != 0 ||
        bq->quantized_dimension != 0) {
      ESP_LOGE(TAG,
               "Cry dev46 FC Prepare: expected per-channel qdim=0 with %d scales/zps",
               output_depth);
      status = kTfLiteError;
    }
  }

  if (status == kTfLiteOk) {
    for (int c = 0; c < output_depth; c++) {
      if (wq->zero_point->data[c] != 0 || bq->zero_point->data[c] != 0 ||
          !(wq->scale->data[c] > 0.0f) || !(bq->scale->data[c] > 0.0f)) {
        ESP_LOGE(TAG, "Cry dev46 FC Prepare: invalid channel quantization at c=%d", c);
        status = kTfLiteError;
        break;
      }
      const double expected = (double) input->params.scale * (double) wq->scale->data[c];
      const double actual = (double) bq->scale->data[c];
      if (!(expected > 0.0) || std::fabs(actual / expected - 1.0) > 0.02) {
        ESP_LOGE(TAG,
                 "Cry dev46 FC Prepare: bias scale mismatch c=%d expected=%.10g actual=%.10g",
                 c, expected, actual);
        status = kTfLiteError;
        break;
      }
    }
  }

  if (status == kTfLiteOk) {
    const auto *params = reinterpret_cast<const TfLiteFullyConnectedParams *>(node->builtin_data);
    if (params == nullptr || params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault ||
        params->keep_num_dims || params->asymmetric_quantize_inputs) {
      ESP_LOGE(TAG, "Cry dev46 FC Prepare: unsupported FC options");
      status = kTfLiteError;
    }
  }

  if (status == kTfLiteOk) {
    if (output_depth > DEV43_FC_MAX_OUTPUTS) {
      ESP_LOGE(TAG, "Cry dev46 FC Prepare: output depth %d exceeds cache %d",
               output_depth, DEV43_FC_MAX_OUTPUTS);
      status = kTfLiteError;
      dev43_fc_invoke_data_.ready = false;
    } else {
      dev43_fc_invoke_data_.input_zero_point = input->params.zero_point;
      dev43_fc_invoke_data_.output_zero_point = output->params.zero_point;
      dev43_fc_invoke_data_.input_scale = input->params.scale;
      dev43_fc_invoke_data_.output_scale = output->params.scale;
      dev43_fc_invoke_data_.activation_min = -128;
      dev43_fc_invoke_data_.activation_max = 127;
      dev43_fc_invoke_data_.output_depth = output_depth;
      if (dev43_fc_invoke_data_.weight_scales == nullptr) {
        dev43_fc_invoke_data_.weight_scales = static_cast<float *>(
            context->AllocatePersistentBuffer(
                context, sizeof(float) * DEV43_FC_MAX_OUTPUTS));
      }
      if (dev43_fc_invoke_data_.weight_scales == nullptr) {
        ESP_LOGE(TAG, "Cry dev46 FC Prepare: persistent scale cache allocation failed");
        status = kTfLiteError;
        dev43_fc_invoke_data_.ready = false;
      }
      const auto *params = reinterpret_cast<const TfLiteFullyConnectedParams *>(node->builtin_data);
      if (params != nullptr) {
        if (params->activation == kTfLiteActRelu) {
          dev43_fc_invoke_data_.activation_min =
              std::max<int32_t>(-128, output->params.zero_point);
        } else if (params->activation == kTfLiteActRelu6) {
          dev43_fc_invoke_data_.activation_min =
              std::max<int32_t>(-128, output->params.zero_point);
          dev43_fc_invoke_data_.activation_max = std::min<int32_t>(
              127, (int32_t) std::lround(6.0 / output->params.scale) +
                       output->params.zero_point);
        }
      }
      if (status == kTfLiteOk) {
        for (int c = 0; c < output_depth; c++)
          dev43_fc_invoke_data_.weight_scales[c] = wq->scale->data[c];
        dev43_fc_invoke_data_.ready = true;
      }
    }
  } else {
    dev43_fc_invoke_data_.ready = false;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights);
  micro_context->DeallocateTempTfLiteTensor(bias);
  micro_context->DeallocateTempTfLiteTensor(output);

  if (status == kTfLiteOk) {
    // This remains the final Prepare node.  After all temporary TfLiteTensor
    // structs are released, capture the eval-tensor-only planner snapshot.
    dev43_mark_(0xBA80u, seq);
    dev43_log_runtime_eval_tensors_(context, "native_planner_input");
  } else {
    dev43_mark_(0xBAFFu, seq);
  }
  return status;
}

static TfLiteStatus dev43_fc_invoke_impl_(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *weights = tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bias = tflite::micro::GetEvalInput(context, node, 2);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  if (!dev43_fc_invoke_data_.ready || input == nullptr || weights == nullptr ||
      bias == nullptr || output == nullptr || input->data.int8 == nullptr ||
      weights->data.int8 == nullptr || bias->data.i32 == nullptr ||
      output->data.int8 == nullptr || input->dims == nullptr ||
      weights->dims == nullptr || output->dims == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 FC Invoke: eval tensor/cache unavailable");
    return kTfLiteError;
  }

  const int batches = input->dims->data[0];
  const int accum_depth = input->dims->data[1];
  const int output_depth = weights->dims->data[0];
  if (batches != 1 || output_depth != dev43_fc_invoke_data_.output_depth ||
      weights->dims->data[1] != accum_depth || output->dims->data[0] != batches ||
      output->dims->data[1] != output_depth)
    return kTfLiteError;

  const int32_t input_zp = dev43_fc_invoke_data_.input_zero_point;
  const int32_t output_zp = dev43_fc_invoke_data_.output_zero_point;
  const double input_scale = (double) dev43_fc_invoke_data_.input_scale;
  const double output_scale = (double) dev43_fc_invoke_data_.output_scale;
  if (!(input_scale > 0.0) || !(output_scale > 0.0)) return kTfLiteError;

  const int8_t *in = input->data.int8;
  const int8_t *w = weights->data.int8;
  const int32_t *b = bias->data.i32;
  int8_t *out = output->data.int8;

  for (int batch = 0; batch < batches; batch++) {
    for (int oc = 0; oc < output_depth; oc++) {
      int32_t acc = b[oc];
      const int8_t *in_row = in + batch * accum_depth;
      const int8_t *w_row = w + oc * accum_depth;
      for (int d = 0; d < accum_depth; d++) {
        acc += (int32_t(w_row[d])) * (int32_t(in_row[d]) - input_zp);
      }

      const double effective_scale =
          input_scale * (double) dev43_fc_invoke_data_.weight_scales[oc] / output_scale;
      int32_t q = (int32_t) std::lround((double) acc * effective_scale) + output_zp;
      q = std::max<int32_t>(dev43_fc_invoke_data_.activation_min,
                            std::min<int32_t>(dev43_fc_invoke_data_.activation_max, q));
      out[batch * output_depth + oc] = static_cast<int8_t>(q);
    }
  }

  static bool first_invoke = true;
  if (first_invoke) {
    first_invoke = false;
    ESP_LOGI(TAG,
             "Cry dev46 custom per-channel FC first EvalTensor-only Invoke OK: batches=%d depth=%d outputs=%d",
             batches, accum_depth, output_depth);
  }
  return kTfLiteOk;
}

static TfLiteStatus dev43_fc_invoke_(TfLiteContext *context, TfLiteNode *node) {
  return dev43_fc_invoke_impl_(context, node);
}


static void dev43_log_recording_allocator_(
    const tflite::RecordingMicroInterpreter *interpreter,
    const char *phase) {
  if (interpreter == nullptr) return;

  const auto &rec = interpreter->GetMicroAllocator();
  const auto *arena = rec.GetSimpleMemoryAllocator();
  if (arena == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 allocator snapshot (%s): simple arena allocator unavailable",
             phase ? phase : "unknown");
    return;
  }

  ESP_LOGI(TAG,
           "Cry dev46 allocator snapshot (%s): used=%u B head=%u B tail=%u B arena=%u B recording_overhead=%u B",
           phase ? phase : "unknown",
           (unsigned) arena->GetUsedBytes(),
           (unsigned) arena->GetNonPersistentUsedBytes(),
           (unsigned) arena->GetPersistentUsedBytes(),
           (unsigned) TENSOR_ARENA_BYTES,
           (unsigned) tflite::RecordingMicroAllocator::GetDefaultTailUsage());

  struct BucketName {
    tflite::RecordedAllocationType type;
    const char *name;
  };
  static const BucketName buckets[] = {
      {tflite::RecordedAllocationType::kTfLiteEvalTensorData, "eval_tensors"},
      {tflite::RecordedAllocationType::kPersistentTfLiteTensorData, "persistent_tflite_tensors"},
      {tflite::RecordedAllocationType::kPersistentTfLiteTensorQuantizationData, "persistent_quant"},
      {tflite::RecordedAllocationType::kPersistentBufferData, "persistent_buffers"},
      {tflite::RecordedAllocationType::kTfLiteTensorVariableBufferData, "variable_buffers"},
      {tflite::RecordedAllocationType::kNodeAndRegistrationArray, "nodes_registrations"},
      {tflite::RecordedAllocationType::kOpData, "op_runtime_data"},
  };

  for (const auto &bucket : buckets) {
    const auto a = rec.GetRecordedAllocation(bucket.type);
    ESP_LOGI(TAG,
             "Cry dev46 allocator bucket %s: requested=%u B used=%u B count=%u",
             bucket.name,
             (unsigned) a.requested_bytes,
             (unsigned) a.used_bytes,
             (unsigned) a.count);
  }

  // dev46 intentionally does NOT call RecordingMicroAllocator::PrintAllocations().
  // dev20 proved that the native report can stall long enough to trip the
  // interrupt watchdog after a failed FinishModelAllocation(). The compact
  // ESPHome-formatted snapshot above contains the diagnostics we need.
}

struct CryDetector::MlState {
  const tflite::Model *model{nullptr};
  // dev3 diagnostic: full resolver to distinguish missing-op failures
  // from tensor-arena / model-contract failures.
  tflite::MicroMutableOpResolver<4> resolver{};
  bool resolver_ready{false};
  tflite::MicroInterpreter *interpreter{nullptr};
  TfLiteTensor *input{nullptr};
  TfLiteTensor *output{nullptr};
  uint8_t *tensor_arena_allocation{nullptr};
  uint8_t *tensor_arena{nullptr};
};

const char *CryDetector::status() const {
  switch (status_.load()) {
    case Status::UNINITIALIZED: return "uninitialized";
    case Status::MODEL_NOT_EMBEDDED: return "model_not_embedded";
    case Status::ALLOC_FAILED: return "alloc_failed";
    case Status::MODEL_INVALID: return "model_invalid";
    case Status::TENSOR_ALLOC_FAILED: return "tensor_alloc_failed";
    case Status::TENSOR_SHAPE_MISMATCH: return "tensor_shape_mismatch";
    case Status::READY: return "ready";
    case Status::INVOKE_FAILED: return "invoke_failed";
    case Status::BAD_SAMPLE_RATE: return "bad_sample_rate";
    default: return "unknown";
  }
}

void CryDetector::set_candidate_threshold(float value) {
  if (!std::isfinite(value)) return;
  candidate_threshold_.store(std::max(0.005f, std::min(0.50f, value)));
}

void CryDetector::reset() {
  reset_requested_.store(true);
}

void CryDetector::init_frontend_() {
  for (size_t n = 0; n < WINDOW_SAMPLES; n++) {
    hann_[n] = 0.5f - 0.5f * cosf((2.0f * PI_F * static_cast<float>(n)) /
                                  static_cast<float>(WINDOW_SAMPLES));
  }

  auto hz_to_mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
  auto mel_to_hz = [](float mel) { return 700.0f * (expf(mel / 1127.0f) - 1.0f); };

  const float lo = hz_to_mel(125.0f);
  const float hi = hz_to_mel(7500.0f);
  for (size_t i = 0; i < MEL_BINS + 2; i++) {
    const float mel = lo + (hi - lo) * static_cast<float>(i) /
                               static_cast<float>(MEL_BINS + 1);
    mel_edges_hz_[i] = mel_to_hz(mel);
  }

  // dev46: precompute the sparse triangular mel bank once. This intentionally
  // calls the existing mel_weight_() so the runtime path uses exactly the same
  // coefficients as dev39 rather than a numerically different approximation.
  mel_sparse_valid_ = true;
  mel_sparse_active_weights_ = 0;
  size_t max_active = 0;
  size_t bins_with_two = 0;
  for (size_t k = 0; k < FFT_MAG_BINS; k++) {
    mel_filter_a_[k] = 0xFF;
    mel_filter_b_[k] = 0xFF;
    mel_weight_a_[k] = 0.0f;
    mel_weight_b_[k] = 0.0f;
    mel_active_count_[k] = 0;

    for (size_t m = 0; m < MEL_BINS; m++) {
      const float weight = mel_weight_(k, m);
      if (!(weight > 0.0f)) continue;

      const uint8_t slot = mel_active_count_[k];
      if (slot == 0) {
        mel_filter_a_[k] = static_cast<uint8_t>(m);
        mel_weight_a_[k] = weight;
      } else if (slot == 1) {
        mel_filter_b_[k] = static_cast<uint8_t>(m);
        mel_weight_b_[k] = weight;
      } else {
        mel_sparse_valid_ = false;
      }

      if (mel_active_count_[k] < 0xFF) mel_active_count_[k]++;
      mel_sparse_active_weights_++;
    }

    max_active = std::max(max_active, static_cast<size_t>(mel_active_count_[k]));
    if (mel_active_count_[k] == 2) bins_with_two++;
  }

  if (!mel_sparse_valid_ || max_active > 2) {
    mel_sparse_valid_ = false;
    ESP_LOGW(TAG,
             "Cry dev46 MEL sparse map INVALID: active_weights=%u max_active=%u; falling back to dev39 dense projection",
             static_cast<unsigned>(mel_sparse_active_weights_),
             static_cast<unsigned>(max_active));
  } else {
    ESP_LOGI(TAG,
             "Cry dev46 MEL sparse map ready: fft_bins=%u mel_bins=%u active_weights=%u bins_with_two=%u max_active=%u runtime_macs/inf=%u dense_weight_tests/inf=%u reduction=%.1fx",
             static_cast<unsigned>(FFT_MAG_BINS),
             static_cast<unsigned>(MEL_BINS),
             static_cast<unsigned>(mel_sparse_active_weights_),
             static_cast<unsigned>(bins_with_two),
             static_cast<unsigned>(max_active),
             static_cast<unsigned>(mel_sparse_active_weights_ * MEL_FRAMES),
             static_cast<unsigned>(FFT_MAG_BINS * MEL_BINS * MEL_FRAMES),
             static_cast<double>(FFT_MAG_BINS * MEL_BINS) /
                 static_cast<double>(std::max<uint16_t>(1, mel_sparse_active_weights_)));
  }

  ESP_LOGI(TAG,
           "Cry dev46 FFT plan ready: size=%u stages=9 bitrev=flash twiddles=256xcomplex flash recurrence=OFF",
           static_cast<unsigned>(FFT_SIZE));
}

float CryDetector::mel_weight_(size_t fft_bin, size_t mel_bin) const {
  const float hz = static_cast<float>(fft_bin) * static_cast<float>(SAMPLE_RATE) /
                   static_cast<float>(FFT_SIZE);
  const float left = mel_edges_hz_[mel_bin];
  const float center = mel_edges_hz_[mel_bin + 1];
  const float right = mel_edges_hz_[mel_bin + 2];

  if (hz <= left || hz >= right) return 0.0f;
  if (hz <= center) return (hz - left) / std::max(1.0e-9f, center - left);
  return (right - hz) / std::max(1.0e-9f, right - center);
}

void CryDetector::fft512_(float *re, float *im) const {
  // dev46: fixed 512-point plan. A direct lookup replaces both the runtime
  // bit-reversal state machine and the recursive complex twiddle update.
  // For stage length L, W_L^k == W_512^(k * 512/L).
  for (size_t i = 0; i < FFT_SIZE; i++) {
    const size_t j = DEV43_BITREV_512[i];
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }

  for (size_t len = 2; len <= FFT_SIZE; len <<= 1) {
    const size_t half = len >> 1;
    const size_t twiddle_step = FFT_SIZE / len;
    for (size_t i = 0; i < FFT_SIZE; i += len) {
      for (size_t k = 0; k < half; k++) {
        const size_t u = i + k;
        const size_t v = u + half;
        const size_t tw = k * twiddle_step;
        const float wr = DEV43_TWIDDLE_RE_512[tw];
        const float wi = DEV43_TWIDDLE_IM_512[tw];
        const float vr = re[v] * wr - im[v] * wi;
        const float vi = re[v] * wi + im[v] * wr;
        const float ur = re[u];
        const float ui = im[u];
        re[u] = ur + vr;
        im[u] = ui + vi;
        re[v] = ur - vr;
        im[v] = ui - vi;
      }
    }
  }
}

bool CryDetector::ensure_buffers_() {
  if (ring_ != nullptr) return true;

  ring_ = static_cast<int16_t *>(
      heap_caps_malloc(PATCH_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  snapshot_ = static_cast<int16_t *>(
      heap_caps_malloc(PATCH_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  fft_re_ = static_cast<float *>(
      heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  fft_im_ = static_cast<float *>(
      heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  hann_ = static_cast<float *>(
      heap_caps_malloc(WINDOW_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (!ring_ || !snapshot_ || !fft_re_ || !fft_im_ || !hann_) {
    status_.store(Status::ALLOC_FAILED);
    return false;
  }

  memset(ring_, 0, PATCH_SAMPLES * sizeof(int16_t));
  memset(snapshot_, 0, PATCH_SAMPLES * sizeof(int16_t));
  init_frontend_();

  mutex_ = xSemaphoreCreateMutex();
  if (mutex_ == nullptr) {
    status_.store(Status::ALLOC_FAILED);
    return false;
  }

  if (xTaskCreatePinnedToCore(task_entry_, "baby_cry_ml", 8192, this, DEV46_ML_PRIORITY_BASE, &task_, 0) != pdPASS) {
    task_ = nullptr;
    status_.store(Status::ALLOC_FAILED);
    return false;
  }
  return true;
}

bool CryDetector::ensure_runtime_() {
  if (model_ready_.load()) return true;

  // CRITICAL dev3 safety rule:
  // initialization is attempted exactly once per boot. dev2 retried after every
  // audio window and leaked a fresh 320 KiB arena on every failure.
  if (runtime_init_attempted_.exchange(true)) return false;

  model_bytes_.store(static_cast<uint32_t>(g_cry_model_data_len));

  const size_t psram_free_before =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  const size_t psram_largest_before =
      heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  ESP_LOGI(TAG,
           "Cry ML dev46 init: production-hardened P5 Invoke, custom INT8 MEAN/FC, 1MiB arena; model=%u B free_psram=%u B",
           (unsigned) g_cry_model_data_len,
           (unsigned) heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (g_cry_model_data_len < 1024) {
    ESP_LOGE(TAG, "Cry model not embedded (bytes=%u)", (unsigned) g_cry_model_data_len);
    status_.store(Status::MODEL_NOT_EMBEDDED);
    return false;
  }

  ml_ = new MlState();
  if (ml_ == nullptr) {
    ESP_LOGE(TAG, "MlState allocation failed");
    status_.store(Status::ALLOC_FAILED);
    return false;
  }

  ml_->model = tflite::GetModel(g_cry_model_data);
  if (ml_->model == nullptr) {
    ESP_LOGE(TAG, "tflite::GetModel() returned nullptr");
    status_.store(Status::MODEL_INVALID);
    return false;
  }

  ESP_LOGI(TAG, "Cry model schema=%lu runtime_schema=%lu",
           (unsigned long) ml_->model->version(),
           (unsigned long) TFLITE_SCHEMA_VERSION);

  if (ml_->model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Cry model schema mismatch");
    status_.store(Status::MODEL_INVALID);
    return false;
  }

  const auto *opcodes = ml_->model->operator_codes();

  const auto *subgraphs = ml_->model->subgraphs();
  const auto *sg0 = (subgraphs != nullptr && subgraphs->size() > 0) ? subgraphs->Get(0) : nullptr;

  if (!ml_->resolver_ready) {
    // Exact operator set discovered from the model FlatBuffer in dev5.
    // dev46 keeps native CONV/DW, replaces MEAN with a controlled INT8
    // global-average kernel, and keeps the dev10 per-channel INT8 FC.
    auto conv_registration = tflite::Register_CONV_2D();
    auto depthwise_registration = tflite::Register_DEPTHWISE_CONV_2D();
    auto fc_registration = tflite::Register_FULLY_CONNECTED();

    dev43_orig_conv_init_ = conv_registration.init;
    dev43_orig_depthwise_init_ = depthwise_registration.init;

    dev43_orig_conv_prepare_ = conv_registration.prepare;
    dev43_orig_depthwise_prepare_ = depthwise_registration.prepare;
    dev43_orig_conv_invoke_ = conv_registration.invoke;
    dev43_orig_depthwise_invoke_ = depthwise_registration.invoke;


    // dev46 buildfix: keep native CONV/DW Init callbacks unwrapped.
    // The original registrations already contain the correct ESP-TFLM init functions.
    // Replace only the final FC with a tiny generic per-channel INT8 kernel.
    // The upstream model remains byte-for-byte unchanged.
    fc_registration.init = nullptr;
    fc_registration.free = nullptr;

    conv_registration.prepare = dev43_conv_prepare_;
    depthwise_registration.prepare = dev43_depthwise_prepare_;
    // dev46: keep native CONV/DW Invoke callbacks unwrapped.
    // Prepare wrappers retain only sequence/error handling; CONV contract audit is disabled.
    conv_registration.invoke = dev43_orig_conv_invoke_;
    depthwise_registration.invoke = dev43_orig_depthwise_invoke_;
    fc_registration.prepare = dev43_fc_prepare_;
    fc_registration.invoke = dev43_fc_invoke_impl_;

    // Espressif's packaged AddMean() accepts no registration argument. Register
    // it normally first, then replace only the stored callbacks through the
    // public FindOp() result.  Parser/version metadata installed by AddMean()
    // stays intact; no header hacks or FlatBuffer modifications are required.
    if (ml_->resolver.AddConv2D(conv_registration) != kTfLiteOk ||
        ml_->resolver.AddDepthwiseConv2D(depthwise_registration) != kTfLiteOk ||
        ml_->resolver.AddMean() != kTfLiteOk ||
        ml_->resolver.AddFullyConnected(fc_registration) != kTfLiteOk) {
      ESP_LOGE(TAG, "Cry dev46 resolver registration failed");
      status_.store(Status::MODEL_INVALID);
      return false;
    }

    const auto *mean_const = ml_->resolver.FindOp(tflite::BuiltinOperator_MEAN);
    if (mean_const == nullptr) {
      ESP_LOGE(TAG, "Cry dev46 resolver: MEAN registration not found after AddMean()");
      status_.store(Status::MODEL_INVALID);
      return false;
    }
    auto *mean_stored = const_cast<TFLMRegistration *>(mean_const);
    mean_stored->init = nullptr;
    mean_stored->free = nullptr;
    mean_stored->prepare = dev43_mean_prepare_;
    mean_stored->invoke = dev43_mean_invoke_impl_;

    ml_->resolver_ready = true;
    ESP_LOGI(TAG, "Cry dev46 resolver ready: CONV/DW native, MEAN/FC custom INT8; Invoke boost=P5");
  }

  // dev46: allocate small canary regions around the exact 1 MiB arena.
  // They are outside the memory handed to TFLM, so checking them after
  // AllocateTensors() returns does not dereference the failed interpreter.
  const size_t arena_allocation_bytes =
      TENSOR_ARENA_BYTES + 2 * DEV43_ARENA_GUARD_BYTES;
  ml_->tensor_arena_allocation = static_cast<uint8_t *>(
      heap_caps_malloc(arena_allocation_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (!ml_->tensor_arena_allocation) {
    ESP_LOGE(TAG,
             "Tensor arena allocation failed: request=%u B (+%u B guards) free_psram=%u B largest=%u B",
             (unsigned) TENSOR_ARENA_BYTES,
             (unsigned) (2 * DEV43_ARENA_GUARD_BYTES),
             (unsigned) heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT),
             (unsigned) heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    status_.store(Status::ALLOC_FAILED);
    return false;
  }

  ml_->tensor_arena = ml_->tensor_arena_allocation + DEV43_ARENA_GUARD_BYTES;
  memset(ml_->tensor_arena_allocation, DEV43_ARENA_GUARD_PREFIX, DEV43_ARENA_GUARD_BYTES);
  memset(ml_->tensor_arena + TENSOR_ARENA_BYTES, DEV43_ARENA_GUARD_SUFFIX,
         DEV43_ARENA_GUARD_BYTES);


  // dev46 FinishModelAllocation boundary probe.  Build the same linear-planner
  // allocator stack explicitly so a Dev34Allocator can override only the
  // private virtual scratch-handle boundary while upstream
  // FinishModelAllocation()/CommitStaticMemoryPlan() remain untouched.
  auto *dev43_arena_allocator = tflite::SingleArenaBufferAllocator::Create(
      ml_->tensor_arena, TENSOR_ARENA_BYTES);
  if (dev43_arena_allocator == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 SingleArenaBufferAllocator::Create failed");
    status_.store(Status::TENSOR_ALLOC_FAILED);
    return false;
  }

  void *dev43_planner_storage = dev43_arena_allocator->AllocatePersistentBuffer(
      sizeof(Dev34TracingLinearPlanner), alignof(Dev34TracingLinearPlanner));
  if (dev43_planner_storage == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 tracing LinearMemoryPlanner storage allocation failed");
    status_.store(Status::TENSOR_ALLOC_FAILED);
    return false;
  }
  auto *dev43_linear_planner =
      new (dev43_planner_storage) Dev34TracingLinearPlanner();

  void *dev43_allocator_storage = dev43_arena_allocator->AllocatePersistentBuffer(
      sizeof(Dev34Allocator), alignof(Dev34Allocator));
  if (dev43_allocator_storage == nullptr) {
    ESP_LOGE(TAG, "Cry dev46 Dev34Allocator storage allocation failed");
    status_.store(Status::TENSOR_ALLOC_FAILED);
    return false;
  }
  auto *dev43_allocator = new (dev43_allocator_storage)
      Dev34Allocator(dev43_arena_allocator, dev43_linear_planner);

  ml_->interpreter = new tflite::MicroInterpreter(
      ml_->model, ml_->resolver, dev43_allocator, nullptr, nullptr);

  if (!ml_->interpreter) {
    ESP_LOGE(TAG, "MicroInterpreter allocation failed");
    heap_caps_free(ml_->tensor_arena_allocation);
    ml_->tensor_arena_allocation = nullptr;
    ml_->tensor_arena = nullptr;
    status_.store(Status::TENSOR_ALLOC_FAILED);
    return false;
  }

  dev43_reset_stage_counters_();
  // dev46 observes only public pre-Finish inputs. The model pointer enables
  // lifetime/const classification during the final FC Prepare; the interpreter
  // pointer deliberately remains null so no allocator accessor is used.
  dev43_active_model_ = ml_->model;
  dev43_active_interpreter_ = nullptr;
  dev43_mark_(0xA000u, 0);
  ESP_LOGI(TAG,
           "Cry dev46 AllocateTensors BEGIN: CommitStaticMemoryPlan boundary probe active; tracing native LinearMemoryPlanner; C1xx/C2xx allocator + D3xx/D4xx planner + E1xx ResetTemp + D5xx post-resizable-head checkpoints; RTC journal armed");

  const size_t dev43_psram_before_allocate =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  const size_t dev43_internal_before_allocate =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  const int64_t dev43_allocate_start_us = esp_timer_get_time();
  const TfLiteStatus dev43_allocate_status = ml_->interpreter->AllocateTensors();
  dev43_mark_(0xF000u, (uint32_t) dev43_allocate_status);
  const uint32_t dev43_allocate_elapsed_us =
      (uint32_t) std::max<int64_t>(0, esp_timer_get_time() - dev43_allocate_start_us);
  const size_t dev43_psram_after_allocate =
      heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  const size_t dev43_internal_after_allocate =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

  size_t dev43_prefix_bad = DEV43_ARENA_GUARD_BYTES;
  size_t dev43_suffix_bad = DEV43_ARENA_GUARD_BYTES;
  for (size_t i = 0; i < DEV43_ARENA_GUARD_BYTES; i++) {
    if (dev43_prefix_bad == DEV43_ARENA_GUARD_BYTES &&
        ml_->tensor_arena_allocation[i] != DEV43_ARENA_GUARD_PREFIX)
      dev43_prefix_bad = i;
    if (dev43_suffix_bad == DEV43_ARENA_GUARD_BYTES &&
        ml_->tensor_arena[TENSOR_ARENA_BYTES + i] != DEV43_ARENA_GUARD_SUFFIX)
      dev43_suffix_bad = i;
  }
  const bool dev43_guards_ok =
      dev43_prefix_bad == DEV43_ARENA_GUARD_BYTES &&
      dev43_suffix_bad == DEV43_ARENA_GUARD_BYTES;
  dev43_mark_(0xF100u, dev43_guards_ok ? 0u : 1u);

  if (!dev43_guards_ok) {
    ESP_LOGE(TAG,
             "Cry dev46 arena canary: FAIL prefix_bad=%d suffix_bad=%d",
             (int) dev43_prefix_bad, (int) dev43_suffix_bad);
  }
  // dev46 isolation point: AllocateTensors() is over.  Replay uses only the
  // copied scalar snapshot and its own 4 KiB planner scratch; it never touches
  // the failed interpreter, TfLiteContext, tensor arena contents or allocator.
  if (dev43_allocate_status != kTfLiteOk) {
    ESP_LOGI(TAG, "Cry dev46 GREEDY_REFERENCE_REPLAY BEGIN: AllocateTensors status=%d",
             (int) dev43_allocate_status);
    dev43_run_native_replay_snapshot_();
    ESP_LOGI(TAG, "Cry dev46 GREEDY_REFERENCE_REPLAY END");
  }

  // dev46 failed-runtime quarantine: after a failed FinishModelAllocation()
  // do not inspect, destroy or free any part of the partially constructed
  // TFLM runtime. The one-shot runtime_init_attempted_ guard prevents retries.
  // Keeping ~1 MiB PSRAM (+1 KiB canary) allocated for the rest of this boot
  // is intentional. dev43 only reads memory outside the arena after failure.
  if (dev43_allocate_status != kTfLiteOk) {
    status_.store(Status::TENSOR_ALLOC_FAILED);
    ESP_LOGE(TAG,
             "Cry dev46 AllocateTensors failed; inspect D4/E1/D5 checkpoint: E1FF=ResetTemp failure; E1A0 without D500=DeallocateResizableBuffer failure; quarantining failed TFLM runtime");
    return false;
  }

  // Success-only diagnostics are safe to retain. No allocator accessor below
  // is ever reached after a failed AllocateTensors() in dev43.
  taskYIELD();

  ESP_LOGI(TAG, "Cry dev46 runtime allocated: arena_used=%u B",
           (unsigned) ml_->interpreter->arena_used_bytes());

  ml_->input = ml_->interpreter->input(0);
  ml_->output = ml_->interpreter->output(0);

  if (!ml_->input || !ml_->output) {
    ESP_LOGE(TAG, "Missing input/output tensor");
    status_.store(Status::TENSOR_SHAPE_MISMATCH);
    return false;
  }

  ESP_LOGI(TAG,
           "Cry tensors: input type=%d bytes=%u dims=%d output type=%d bytes=%u dims=%d",
           (int) ml_->input->type,
           (unsigned) ml_->input->bytes,
           ml_->input->dims ? ml_->input->dims->size : -1,
           (int) ml_->output->type,
           (unsigned) ml_->output->bytes,
           ml_->output->dims ? ml_->output->dims->size : -1);

  if (ml_->input->type != kTfLiteInt8 ||
      ml_->output->type != kTfLiteInt8 ||
      ml_->input->bytes != MEL_FRAMES * MEL_BINS ||
      ml_->output->bytes != 521) {
    ESP_LOGE(TAG,
             "Cry tensor contract mismatch: expected input=%u B output=521 B",
             (unsigned) (MEL_FRAMES * MEL_BINS));
    status_.store(Status::TENSOR_SHAPE_MISMATCH);
    return false;
  }

  model_ready_.store(true);
  status_.store(Status::READY);

  ESP_LOGI(TAG,
           "Cry ML ready: model=%u B arena=%u B free_psram=%u B "
           "input_scale=%.7f zp=%ld output_scale=%.7f zp=%ld",
           (unsigned) g_cry_model_data_len,
           (unsigned) TENSOR_ARENA_BYTES,
           (unsigned) heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT),
           ml_->input->params.scale,
           (long) ml_->input->params.zero_point,
           ml_->output->params.scale,
           (long) ml_->output->params.zero_point);

  return true;
}

void CryDetector::append_samples_(const int16_t *samples, size_t count) {
  if (samples == nullptr || count == 0) return;

  // v1.4.2: an AAC AU normally contributes a small block, but keep the
  // helper correct for any caller. If a block exceeds the ring capacity, only
  // its newest PATCH_SAMPLES samples can affect the latest-patch semantics.
  if (count > PATCH_SAMPLES) {
    samples += count - PATCH_SAMPLES;
    count = PATCH_SAMPLES;
  }

  const size_t first = std::min(count, PATCH_SAMPLES - ring_write_);
  memcpy(ring_ + ring_write_, samples, first * sizeof(int16_t));
  const size_t second = count - first;
  if (second != 0)
    memcpy(ring_, samples + first, second * sizeof(int16_t));

  ring_write_ += count;
  if (ring_write_ >= PATCH_SAMPLES) ring_write_ -= PATCH_SAMPLES;
  ring_filled_ = std::min(PATCH_SAMPLES, ring_filled_ + count);
}

bool CryDetector::snapshot_latest_patch_() {
  if (ring_filled_ < PATCH_SAMPLES) return false;
  const size_t tail = PATCH_SAMPLES - ring_write_;
  memcpy(snapshot_, ring_ + ring_write_, tail * sizeof(int16_t));
  if (ring_write_ != 0)
    memcpy(snapshot_ + tail, ring_, ring_write_ * sizeof(int16_t));
  return true;
}

void CryDetector::feed(const int16_t *samples, size_t sample_count, uint32_t sample_rate_hz) {
  if (samples == nullptr || sample_count == 0) return;
  if (sample_rate_hz != SAMPLE_RATE) {
    status_.store(Status::BAD_SAMPLE_RATE);
    return;
  }
  if (!ensure_buffers_()) return;

  if (reset_requested_.exchange(false)) {
    if (xSemaphoreTake(mutex_, pdMS_TO_TICKS(5)) == pdTRUE) {
      memset(ring_, 0, PATCH_SAMPLES * sizeof(int16_t));
      ring_write_ = ring_filled_ = samples_since_window_ = 0;
      window_pending_.store(false);
      candidate_.store(false);
      cry_score_.store(0.0f);
      xSemaphoreGive(mutex_);
    }
  }

  if (xSemaphoreTake(mutex_, pdMS_TO_TICKS(2)) != pdTRUE) return;
  append_samples_(samples, sample_count);
  samples_since_window_ += sample_count;

  if (ring_filled_ == PATCH_SAMPLES && samples_since_window_ >= HOP_SAMPLES) {
    samples_since_window_ %= HOP_SAMPLES;
    if (!window_pending_.load()) {
      if (snapshot_latest_patch_()) {
        window_pending_.store(true);
        xTaskNotifyGive(task_);
      }
    } else {
      dropped_windows_.fetch_add(1);
    }
  }
  xSemaphoreGive(mutex_);
}

// dev46: per-inference frontend timing snapshot. This remains file-local and
// is written/read only by the Cry ML task, so it does not add synchronization
// or allocations to the hot path.
struct Dev39FrontendTiming {
  uint32_t clear_us{0};
  uint32_t window_us{0};
  uint32_t fft_us{0};
  uint32_t mel_us{0};
  uint32_t logq_us{0};
};
static Dev39FrontendTiming dev43_frontend_timing_{};

bool CryDetector::build_logmel_and_quantize_() {
  if (!ensure_runtime_()) return false;

  int8_t *dst = ml_->input->data.int8;
  const float scale = ml_->input->params.scale;
  const int32_t zero = ml_->input->params.zero_point;
  if (!(scale > 0.0f)) return false;

  Dev39FrontendTiming timing{};

  for (size_t frame = 0; frame < MEL_FRAMES; frame++) {
    const size_t start = frame * FRAME_HOP;

    int64_t stage_us = esp_timer_get_time();
    std::fill(fft_re_, fft_re_ + FFT_SIZE, 0.0f);
    std::fill(fft_im_, fft_im_ + FFT_SIZE, 0.0f);
    timing.clear_us += static_cast<uint32_t>(
        std::max<int64_t>(0, esp_timer_get_time() - stage_us));

    stage_us = esp_timer_get_time();
    for (size_t i = 0; i < WINDOW_SAMPLES; i++) {
      fft_re_[i] = (static_cast<float>(snapshot_[start + i]) / 32768.0f) * hann_[i];
    }
    timing.window_us += static_cast<uint32_t>(
        std::max<int64_t>(0, esp_timer_get_time() - stage_us));

    stage_us = esp_timer_get_time();
    fft512_(fft_re_, fft_im_);
    timing.fft_us += static_cast<uint32_t>(
        std::max<int64_t>(0, esp_timer_get_time() - stage_us));

    stage_us = esp_timer_get_time();
    float mel[MEL_BINS]{};
    if (mel_sparse_valid_) {
      // dev46 hot path: one magnitude plus at most two MACs per FFT bin.
      // The precomputed coefficients were generated by mel_weight_(), so this
      // changes loop structure only, not the filterbank definition.
      for (size_t k = 0; k < FFT_MAG_BINS; k++) {
        const float mag = sqrtf(fft_re_[k] * fft_re_[k] + fft_im_[k] * fft_im_[k]);
        if (mag <= 0.0f) continue;

        const uint8_t count = mel_active_count_[k];
        if (count >= 1) {
          mel[mel_filter_a_[k]] += mag * mel_weight_a_[k];
        }
        if (count >= 2) {
          mel[mel_filter_b_[k]] += mag * mel_weight_b_[k];
        }
      }
    } else {
      // Safety fallback: byte-for-byte equivalent algorithmic path to dev39.
      for (size_t k = 0; k < FFT_MAG_BINS; k++) {
        const float mag = sqrtf(fft_re_[k] * fft_re_[k] + fft_im_[k] * fft_im_[k]);
        if (mag <= 0.0f) continue;
        for (size_t m = 0; m < MEL_BINS; m++) {
          const float w = mel_weight_(k, m);
          if (w > 0.0f) mel[m] += mag * w;
        }
      }
    }
    timing.mel_us += static_cast<uint32_t>(
        std::max<int64_t>(0, esp_timer_get_time() - stage_us));

    stage_us = esp_timer_get_time();
    for (size_t m = 0; m < MEL_BINS; m++) {
      const float logmel = logf(mel[m] + 0.001f);
      int32_t q = static_cast<int32_t>(lroundf(logmel / scale)) + zero;
      q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
      dst[frame * MEL_BINS + m] = static_cast<int8_t>(q);
    }
    timing.logq_us += static_cast<uint32_t>(
        std::max<int64_t>(0, esp_timer_get_time() - stage_us));
  }

  dev43_frontend_timing_ = timing;
  return true;
}

void CryDetector::run_inference_() {
  // v1.4.1-dev4: zero-allocation scope marker for LCD/SPI diagnostics.
  struct InferenceActiveGuard {
    std::atomic<bool> &flag;
    explicit InferenceActiveGuard(std::atomic<bool> &f) : flag(f) { flag.store(true); }
    ~InferenceActiveGuard() { flag.store(false); }
  } inference_guard(inference_active_);

  const int64_t total_start_us = esp_timer_get_time();

  const int64_t preprocess_start_us = total_start_us;
  if (!build_logmel_and_quantize_()) return;
  const uint32_t preprocess_us = static_cast<uint32_t>(
      std::max<int64_t>(0, esp_timer_get_time() - preprocess_start_us));

  // dev46: deterministic Invoke-only priority boost. Keep all frontend and
  // post-processing at the task's normal P1; raise only the TFLM graph execution
  // above baby_rtsp (P4), then restore the exact previous priority immediately.
  const UBaseType_t priority_before = uxTaskPriorityGet(nullptr);
  if (priority_before != DEV46_ML_PRIORITY_INVOKE)
    vTaskPrioritySet(nullptr, DEV46_ML_PRIORITY_INVOKE);

  const int64_t invoke_start_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = ml_->interpreter->Invoke();
  const uint32_t invoke_us = static_cast<uint32_t>(
      std::max<int64_t>(0, esp_timer_get_time() - invoke_start_us));

  if (uxTaskPriorityGet(nullptr) != priority_before)
    vTaskPrioritySet(nullptr, priority_before);

  if (invoke_status != kTfLiteOk) {
    status_.store(Status::INVOKE_FAILED);
    model_ready_.store(false);
    return;
  }

  const int64_t post_start_us = esp_timer_get_time();
  const int8_t *q = ml_->output->data.int8;
  const float scale = ml_->output->params.scale;
  const int32_t zero = ml_->output->params.zero_point;

  float max_logit = -1.0e30f;
  for (size_t i = 0; i < 521; i++) {
    const float v = (static_cast<int32_t>(q[i]) - zero) * scale;
    max_logit = std::max(max_logit, v);
  }

  float denom = 0.0f;
  float p_crying = 0.0f, p_baby = 0.0f, p_speech = 0.0f;
  float p_yell = 0.0f, p_scream = 0.0f, p_babble = 0.0f;

  for (size_t i = 0; i < 521; i++) {
    const float v = (static_cast<int32_t>(q[i]) - zero) * scale;
    const float e = expf(v - max_logit);
    denom += e;
    if (i == IDX_CRYING) p_crying = e;
    else if (i == IDX_BABY_CRY) p_baby = e;
    else if (i == IDX_SPEECH) p_speech = e;
    else if (i == IDX_YELL) p_yell = e;
    else if (i == IDX_SCREAMING) p_scream = e;
    else if (i == IDX_BABBLING) p_babble = e;
  }
  if (denom <= 0.0f) return;

  p_crying /= denom;
  p_baby /= denom;
  p_speech /= denom;
  p_yell /= denom;
  p_scream /= denom;
  p_babble /= denom;

  const float cry = p_crying + p_baby;
  crying_score_.store(p_crying);
  baby_cry_score_.store(p_baby);
  speech_score_.store(p_speech);
  yell_scream_score_.store(p_yell + p_scream);
  babbling_score_.store(p_babble);
  cry_score_.store(cry);
  candidate_.store(cry >= candidate_threshold_.load());
  const uint32_t inference_number = inference_count_.fetch_add(1) + 1;

  const uint32_t post_us = static_cast<uint32_t>(
      std::max<int64_t>(0, esp_timer_get_time() - post_start_us));
  const uint32_t total_us = static_cast<uint32_t>(
      std::max<int64_t>(0, esp_timer_get_time() - total_start_us));
  inference_ms_.store((total_us + 500) / 1000);
  status_.store(Status::READY);

  // dev46: aggregate the same five successful inferences as dev38, but split
  // the frontend into its real algorithmic stages. No per-frame logging is
  // emitted, so the logger cannot dominate the measurement.
  static uint64_t perf_clear_us = 0;
  static uint64_t perf_window_us = 0;
  static uint64_t perf_fft_us = 0;
  static uint64_t perf_mel_us = 0;
  static uint64_t perf_logq_us = 0;
  static uint64_t perf_pre_us = 0;
  static uint64_t perf_invoke_us = 0;
  static uint64_t perf_post_us = 0;
  static uint64_t perf_total_us = 0;
  static uint32_t perf_samples = 0;
  static uint32_t perf_invoke_min_us = UINT32_MAX;
  static uint32_t perf_invoke_max_us = 0;

  perf_clear_us += dev43_frontend_timing_.clear_us;
  perf_window_us += dev43_frontend_timing_.window_us;
  perf_fft_us += dev43_frontend_timing_.fft_us;
  perf_mel_us += dev43_frontend_timing_.mel_us;
  perf_logq_us += dev43_frontend_timing_.logq_us;
  perf_pre_us += preprocess_us;
  perf_invoke_us += invoke_us;
  perf_post_us += post_us;
  perf_total_us += total_us;
  perf_samples++;
  perf_invoke_min_us = std::min(perf_invoke_min_us, invoke_us);
  perf_invoke_max_us = std::max(perf_invoke_max_us, invoke_us);

  if (perf_samples >= 5) {
    const uint64_t staged_us = perf_clear_us + perf_window_us + perf_fft_us +
                               perf_mel_us + perf_logq_us;
    const uint64_t other_us = perf_pre_us > staged_us ? perf_pre_us - staged_us : 0;
    const uint32_t dense_weight_tests_per_inference =
        static_cast<uint32_t>(MEL_FRAMES * FFT_MAG_BINS * MEL_BINS);
    const uint32_t sparse_macs_per_inference =
        static_cast<uint32_t>(MEL_FRAMES) *
        static_cast<uint32_t>(mel_sparse_active_weights_);

    ESP_LOGI(TAG,
             "Cry dev46 PREP: n=%u clear=%ums avg window=%ums avg fft=%ums avg mel=%ums avg logq=%ums avg other=%ums avg frontend=%ums avg frames=%u fft_calls/inf=%u sparse=%s sparse_macs/inf=%u dense_weight_tests/inf=%u",
             (unsigned) perf_samples,
             (unsigned) ((perf_clear_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_window_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_fft_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_mel_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_logq_us / perf_samples + 500) / 1000),
             (unsigned) ((other_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_pre_us / perf_samples + 500) / 1000),
             (unsigned) MEL_FRAMES, (unsigned) MEL_FRAMES,
             mel_sparse_valid_ ? "YES" : "NO",
             (unsigned) sparse_macs_per_inference,
             (unsigned) dense_weight_tests_per_inference);

    ESP_LOGI(TAG,
             "Cry dev46 PERF[P5]: n=%u inference_count=%u preprocess=%ums avg invoke=%ums avg post=%ums avg total=%ums avg invoke_min=%ums invoke_max=%ums dropped=%u",
             (unsigned) perf_samples, (unsigned) inference_number,
             (unsigned) ((perf_pre_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_invoke_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_post_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_total_us / perf_samples + 500) / 1000),
             (unsigned) ((perf_invoke_min_us + 500) / 1000),
             (unsigned) ((perf_invoke_max_us + 500) / 1000),
             (unsigned) dropped_windows_.load());

    perf_clear_us = perf_window_us = perf_fft_us = perf_mel_us = perf_logq_us = 0;
    perf_pre_us = perf_invoke_us = perf_post_us = perf_total_us = 0;
    perf_samples = 0;
    perf_invoke_min_us = UINT32_MAX;
    perf_invoke_max_us = 0;
  }

}

void CryDetector::task_entry_(void *arg) {
  static_cast<CryDetector *>(arg)->task_loop_();
}

void CryDetector::task_loop_() {
  ensure_runtime_();
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (!window_pending_.exchange(false)) continue;
    run_inference_();
    taskYIELD();
  }
}

}  // namespace esphome::baby_monitor
