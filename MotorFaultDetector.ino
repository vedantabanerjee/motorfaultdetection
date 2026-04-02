/***********************************************************************
 * Mechanical Fault Detection in Motor - Vedanta Banerjee
 * ─────────────────────────────────────────────────────────────────────
 * Hardware:
 *   • Seeed XIAO ESP32C3
 *   • LSM6DSO 6-DoF IMU (I2C, shared SDA/SCL with OLED)
 *   • SSD1306 1.3" OLED 128×64 (I2C via LSM6DSO module's pass-through)
 *
 * Libraries (install via Arduino Library Manager):
 *   • SparkFun 6DoF IMU Breakout - LSM6DSO  (by SparkFun)
 *   • U8g2                                  (by oliver)
 *   • EdgeNeuron + EdgeMath                 (by Consentium IoT)
 *
 * Inference cycle:
 *   1. Collect 2080 samples at 416 Hz (~5 seconds) — uninterrupted
 *   2. Run model on 4 non-overlapping 512-sample windows
 *   3. Majority vote → display result for 4 seconds
 *   4. Repeat
 *
 * Files needed in sketch folder:
 *   • MotorFaultDetector.ino
 *   • motor_fault_model.h
 ***********************************************************************/

#include <Wire.h>
#include <U8g2lib.h>
#include <SparkFunLSM6DSO.h>
#include <EdgeNeuron.h>
#include <EdgeMath.h>
#include "motor_fault_model.h"

// ════════════════════════════════════════════════════════════════════
//  CONFIGURATION
// ════════════════════════════════════════════════════════════════════

// Tensor arena
constexpr size_t kTensorArenaSize = 75 * 1024;

// Sampling
constexpr int    SAMPLE_HZ       = 416;
constexpr ulong  SAMPLE_US       = 1000000UL / SAMPLE_HZ;   // 2404 µs

// Collection
constexpr int    COLLECT_SEC     = 5;
constexpr int    TOTAL_SAMPLES   = SAMPLE_HZ * COLLECT_SEC;  // 2080

// Model I/O
constexpr int    SEQUENCE_LEN    = 512;    // samples per inference window
constexpr int    NUM_CHANNELS    = 4;      // ax, ay, az, av
constexpr int    kInputSize      = SEQUENCE_LEN * NUM_CHANNELS;  // 2048
constexpr int    kOutputSize     = 4;
constexpr int    N_WINDOWS       = TOTAL_SAMPLES / SEQUENCE_LEN; // 4

// LSM6DSO I2C address
constexpr uint8_t IMU_ADDR = 0x6B;

// ════════════════════════════════════════════════════════════════════
//  GLOBALS
// ════════════════════════════════════════════════════════════════════

// TFLite working memory
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Raw vibration data: 2080 samples × 4 channels × 4 bytes ≈ 33 KB
static float raw_data[TOTAL_SAMPLES][NUM_CHANNELS];

// Flattened, normalised input for the model and its output
static float input_data[kInputSize];
static float output_data[kOutputSize];

// ════════════════════════════════════════════════════════════════════
//  OBJECTS
// ════════════════════════════════════════════════════════════════════

// SSD1306 128×64 I2C — full frame-buffer mode (1 KB GRAM in MCU)
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

LSM6DSO myIMU;
EdgeMath edgemath;

// ════════════════════════════════════════════════════════════════════
//  CLASS METADATA
// ════════════════════════════════════════════════════════════════════


const char* CLASS_LABELS[] = {
    "Motor OFF",   // 0
    "Motor ON",    // 1
    "No Fan",      // 2  ← fault
    "Bad Fan"      // 3  ← fault
};

// Mark which classes are fault states (shown with inverted alert bar)
const bool IS_FAULT[] = { false, false, true, true };

// ════════════════════════════════════════════════════════════════════
//  OLED HELPERS
// ════════════════════════════════════════════════════════════════════

/** General-purpose 3-line text screen */
void oledText(const char* l1, const char* l2 = nullptr, const char* l3 = nullptr) {
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tf);
    if (l1) u8g2.drawStr(2, 14, l1);
    if (l2) u8g2.drawStr(2, 32, l2);
    if (l3) u8g2.drawStr(2, 50, l3);
    u8g2.sendBuffer();
}

/**
 * Progress screen shown during data collection.
 * Draws a progress bar and elapsed time.
 *
 * @param elapsed_ms  Milliseconds elapsed since collection started
 * @param total_ms    Total collection duration in milliseconds
 */
void oledProgress(int elapsed_ms, int total_ms) {
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tf);

    u8g2.drawStr(2, 13, "Collecting data...");

    // Progress bar (126 px wide, inside a 1-px frame)
    u8g2.drawFrame(1, 18, 126, 12);
    int filled = (int)((long)elapsed_ms * 122 / total_ms);
    if (filled < 0) filled = 0;
    if (filled > 122) filled = 122;
    if (filled > 0) u8g2.drawBox(2, 19, filled, 10);

    // Elapsed / total seconds
    char buf[24];
    snprintf(buf, sizeof(buf), "%d sec / %d sec", elapsed_ms / 1000, total_ms / 1000);
    u8g2.drawStr(2, 46, buf);

    u8g2.drawStr(2, 58, "Vibration analysis");
    u8g2.sendBuffer();
}

/**
 * Shows which inference window is currently running.
 *
 * @param win    Current window index (1-based)
 * @param total  Total number of windows
 */
void oledInferring(int win, int total) {
    char buf[32];
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(2, 14, "Running ML model...");
    snprintf(buf, sizeof(buf), "Window %d / %d", win, total);
    u8g2.drawStr(2, 32, buf);
    u8g2.drawStr(2, 50, "Please wait...");
    u8g2.sendBuffer();
}

/**
 * Displays the final classification result.
 * Fault states get an inverted alert header.
 *
 * @param class_id        Predicted class index (0–3)
 * @param confidence_pct  Majority-vote confidence (0–100)
 * @param votes           Number of windows that voted for this class
 * @param total_wins      Total windows inferred
 */
void oledResult(int class_id, int confidence_pct, int votes, int total_wins) {
    u8g2.clearBuffer();

    if (IS_FAULT[class_id]) {
        // Inverted header bar for fault states
        u8g2.drawBox(0, 0, 128, 15);
        u8g2.setDrawColor(0);
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(10, 11, "!! FAULT DETECTED !!");
        u8g2.setDrawColor(1);
    } else {
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(2, 11, "Status: Normal");
    }

    u8g2.drawHLine(0, 16, 128);

    // Class name (slightly larger font for prominence)
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(2, 33, CLASS_LABELS[class_id]);

    // Confidence line
    u8g2.setFont(u8g2_font_6x10_tf);
    char conf_buf[28];
    snprintf(conf_buf, sizeof(conf_buf), "Conf: %d%%  (%d/%d wins)", confidence_pct, votes, total_wins);
    u8g2.drawStr(2, 50, conf_buf);

    // Refresh countdown hint
    u8g2.drawStr(2, 62, "Refreshing in 4s...");

    u8g2.sendBuffer();
}

// ════════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    delay(500);

    // XIAO ESP32C3 default I2C pins: SDA = D4 (GPIO6), SCL = D5 (GPIO7)
    Wire.begin();
    Wire.setClock(400000); // 400 kHz fast-mode (both LSM6DSO and SSD1306 support it)

    // ── OLED ──────────────────────────────────────────────────────
    u8g2.begin();
    oledText("VibroSense", "v2.0", "Device Initializing...");
    delay(7000);
    Serial.println("[OK] OLED initialised");

    // ── LSM6DSO ───────────────────────────────────────────────────
    if (!myIMU.begin(IMU_ADDR, Wire)) {
    if (!myIMU.begin(IMU_ADDR == 0x6B ? 0x6A : 0x6B, Wire)) {
        Serial.println("[ERR] LSM6DSO not found on 0x6A or 0x6B!");
        oledText("LSM6DSO FAILED", "Check I2C wiring", "Halting.");
        while (true) delay(1000);
    }
}

    // ±2g range
    myIMU.setAccelRange(2);

    // 416 Hz is the nearest supported ODR to the training rate of 445 Hz.
    myIMU.setAccelDataRate(416);

    Serial.println("[OK] LSM6DSO — ±2g @ 416 Hz");

    // ── TFLite Model ──────────────────────────────────────────────
    Serial.print("[..] Initialising TFLite model... ");
    if (!initializeModel(motor_fault_model, tensor_arena, kTensorArenaSize)) {
        Serial.println("FAILED!");
        Serial.println("      Increase kTensorArenaSize by 5 KB and re-flash.");
        Serial.printf ("      Current size: %zu KB\n", kTensorArenaSize / 1024);
        oledText("Model FAILED!", "Raise arena size", "See Serial log.");
        while (true) delay(1000);
    }
    Serial.printf("OK  (arena: %zu KB)\n", kTensorArenaSize / 1024);

    // ── Memory report ─────────────────────────────────────────────
    Serial.printf("[i] Free heap: %lu bytes\n", (unsigned long)ESP.getFreeHeap());
    Serial.printf("[i] N_WINDOWS per cycle: %d\n", N_WINDOWS);

    oledText("Ready!", "Starting cycle...");
    delay(2000);
    Serial.println("[OK] All systems ready. Starting inference loop.\n");
}

// ════════════════════════════════════════════════════════════════════
//  MAIN LOOP
// ════════════════════════════════════════════════════════════════════

void loop() {

    // ── Phase 1: Collect TOTAL_SAMPLES at SAMPLE_HZ ──────────────
    Serial.printf("=== Phase 1: Collecting %d samples @ %d Hz ===\n",
                  TOTAL_SAMPLES, SAMPLE_HZ);

    unsigned long collect_start = millis();

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long t0 = micros();

        // Read all three axes
        float ax = myIMU.readFloatAccelX();
        float ay = myIMU.readFloatAccelY();
        float az = myIMU.readFloatAccelZ();

        // Compute acceleration vector magnitude (4th feature, same as training)
        float av = sqrtf(ax*ax + ay*ay + az*az);

        raw_data[i][0] = ax;
        raw_data[i][1] = ay;
        raw_data[i][2] = az;
        raw_data[i][3] = av;

        // Update OLED progress ~4× per second to avoid slowing the loop
        // (every 104 samples ≈ 250 ms at 416 Hz)
        if (i % 104 == 0) {
            oledProgress((int)(millis() - collect_start),
                         COLLECT_SEC * 1000);
        }

        // Busy-wait to maintain SAMPLE_HZ timing
        // The IMU's hardware ODR governs the actual data rate; this pacing
        // ensures we don't poll faster than new data is ready.
        while (micros() - t0 < SAMPLE_US);
    }

    unsigned long actual_ms = millis() - collect_start;
    Serial.printf("Collection done: %lu ms  (target: %d ms)\n\n",
                  actual_ms, COLLECT_SEC * 1000);

    // ── Phase 2: Inference on N_WINDOWS non-overlapping windows ──
    Serial.printf("=== Phase 2: Inference on %d windows ===\n", N_WINDOWS);

    int vote_counts[kOutputSize] = {0, 0, 0, 0};
    int windows_ok = 0;

    for (int w = 0; w < N_WINDOWS; w++) {
        oledInferring(w + 1, N_WINDOWS);

        int offset = w * SEQUENCE_LEN; // first sample index of this window
        Serial.printf("[Window %d/%d] samples %d-%d\n",
                      w + 1, N_WINDOWS, offset, offset + SEQUENCE_LEN - 1);

        // ── Normalise and flatten into input_data ─────────────────
        // Model input layout: (time=512, channel=4, depth=1), row-major.
        // input_data[t*4 + ch] = (raw - mean[ch]) / scale[ch]
        //
        // scaler_mean[] and scaler_scale[] are defined in motor_fault_model.h
        // with 4 entries each: [ax, ay, az, av]
        for (int t = 0; t < SEQUENCE_LEN; t++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                input_data[t * NUM_CHANNELS + ch] =
                    (raw_data[offset + t][ch] - scaler_mean[ch]) / scaler_scale[ch];
            }
        }

        // ── Feed input tensor ─────────────────────────────────────
        for (int i = 0; i < kInputSize; i++) {
            setModelInput(input_data[i], i);
        }

        // ── Run inference ─────────────────────────────────────────
        unsigned long t_inf = millis();
        if (!runModelInference()) {
            Serial.printf("  [ERR] Inference failed — skipping window %d\n", w + 1);
            continue;
        }
        Serial.printf("  Inference time: %lu ms\n", millis() - t_inf);

        // ── Read outputs ──────────────────────────────────────────
        for (int i = 0; i < kOutputSize; i++) {
            output_data[i] = getModelOutput(i);
            Serial.printf("  %-14s: %.4f\n", CLASS_LABELS[i], output_data[i]);
        }

        int predicted = edgemath.argmax(output_data, kOutputSize);
        vote_counts[predicted]++;
        windows_ok++;
        Serial.printf("  → Vote: %s\n\n", CLASS_LABELS[predicted]);
    }

    // ── Phase 3: Majority vote ────────────────────────────────────
    Serial.println("=== Phase 3: Majority vote ===");

    int final_class = 0;
    int max_votes   = vote_counts[0];
    for (int i = 1; i < kOutputSize; i++) {
        if (vote_counts[i] > max_votes) {
            max_votes   = vote_counts[i];
            final_class = i;
        }
    }

    int confidence_pct = (windows_ok > 0) ? (max_votes * 100 / windows_ok) : 0;

    Serial.println("Vote breakdown:");
    for (int i = 0; i < kOutputSize; i++) {
        Serial.printf("  %-14s: %d vote(s)%s\n",
                      CLASS_LABELS[i], vote_counts[i],
                      (i == final_class) ? "  ← WINNER" : "");
    }
    Serial.printf("RESULT: %s | Confidence: %d%% (%d/%d windows)\n\n",
                  CLASS_LABELS[final_class], confidence_pct, max_votes, windows_ok);

    // ── Phase 4: Display result for 4 seconds ────────────────────
    oledResult(final_class, confidence_pct, max_votes, windows_ok);
    delay(4000);
}
