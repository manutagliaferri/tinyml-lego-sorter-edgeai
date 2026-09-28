#include "camera.h"
#include "himax.h"
#include "SDRAM.h"
#include "lego_model.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMAGE_MODE CAMERA_GRAYSCALE
#define FRAME_RATE 30
#define CAMERA_WIDTH 320
#define CAMERA_HEIGHT 240
#define CAMERA_PIXEL_COUNT (CAMERA_WIDTH * CAMERA_HEIGHT)
#define SERIAL_BITRATE (CAMERA_PIXEL_COUNT * 8 * FRAME_RATE)
#define MIN_BLOB_SIZE 100
#define DETECTION_THRESHOLD 50
#define STABILIZATION_FRAMES 10
#define FRAME_ALPHA 0.5f
#define MODEL_SIZE 128
#define MODEL_PIXEL_COUNT (MODEL_SIZE * MODEL_SIZE)
#define BUTTON_PIN 10
#define CLASS_COUNT 15
#define TENSOR_ARENA_SIZE (2 * 1024 * 1024)
#define SDRAM_OFFSET (4 * 1024 * 1024)

struct BoundingBox {
    int min_x, min_y, max_x, max_y;
    bool valid;
};

struct PingPong {
    uint8_t* a;
    uint8_t* b;
    bool use_a;
    
    PingPong(uint8_t* _a, uint8_t* _b) : a(_a), b(_b), use_a(false) {}
    
    uint8_t* read() const { return use_a ? a : b; }
    uint8_t* write() const { return use_a ? b : a; }
    void swap() { use_a = !use_a; }
};

uint8_t background[CAMERA_PIXEL_COUNT];
uint8_t frame_ema[CAMERA_PIXEL_COUNT];
uint8_t temp_buffer_1[CAMERA_PIXEL_COUNT];
uint8_t temp_buffer_2[CAMERA_PIXEL_COUNT];

unsigned int frame_count = 0;

uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

const int CLASSES[CLASS_COUNT] = {
    2357, 2420, 3001, 3002, 3003, 
    3004, 3005, 3010, 3020, 3021, 
    3022, 3023, 3024, 3622, 3623
};

HM01B0 himax;
Camera cam(himax);
FrameBuffer fb(CAMERA_WIDTH, CAMERA_HEIGHT, 1);
PingPong pong(temp_buffer_1, temp_buffer_2);

void setup() {
    Serial.begin(SERIAL_BITRATE);
    while (!Serial);

    Serial.println("Starting setup");

    if (!SDRAM.begin(SDRAM_START_ADDRESS + SDRAM_OFFSET)) {
        Serial.println("ERROR: SDRAM failed!");
        while (true);
    }
    Serial.println("SDRAM initialized");

    setup_mpu_for_sdram();

    tensor_arena = (uint8_t*)SDRAM_START_ADDRESS;

    Serial.print("Tensor arena at: 0x");
    Serial.println((uint32_t)tensor_arena, HEX);

    cam.begin(CAMERA_R320x240, IMAGE_MODE, FRAME_RATE);

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    const tflite::Model* model = tflite::GetModel(lego_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        while(true);
    }

    Serial.println("Setting up resolver");

    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddDepthwiseConv2D();
    resolver.AddTranspose();
    resolver.AddSum();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddAdd();
    resolver.AddRsqrt();
    resolver.AddGatherNd();
    resolver.AddLeakyRelu();
    resolver.AddConv2D();
    resolver.AddFullyConnected();

    Serial.println("Setting up interpreter");
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    Serial.println("Allocating tensors");

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("CRITICAL ERROR: AllocateTensors() failed!");
        size_t used = interpreter->arena_used_bytes();
        Serial.print("Arena Used: ");
        Serial.print(used);
        Serial.print(" bytes out of ");
        Serial.println(TENSOR_ARENA_SIZE);
        while(true);
    }
    
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
    
    Serial.println("Setup Complete.");
}

void loop() {
    frame_count++;

    if (cam.grabFrame(fb, 3000) != 0) {
        return;
    }
    uint8_t* frame = fb.getBuffer();

    int buttonState = digitalRead(BUTTON_PIN);
    if (buttonState == LOW) {
        frame_count = 0;
    }

    if (frame_count < STABILIZATION_FRAMES) {
        if (frame_count == 0) {
            memcpy(background, frame, CAMERA_PIXEL_COUNT);
            memcpy(frame_ema, frame, CAMERA_PIXEL_COUNT);
        } else {
            ema_update(background, frame, FRAME_ALPHA);
        }
        return;
    }

    ema_update(frame_ema, frame, FRAME_ALPHA);

    background_subtraction(pong.write(), frame_ema);
    pong.swap();

    BoundingBox box = find_bounding_box(pong.read());
    if (!box.valid) {
        return;
    }
    
    fit_square(box);
    int square_size = box.max_x - box.min_x + 1;

    extract_square(pong.write(), pong.read(), box, square_size);
    pong.swap();
    
    uint8_t model_image[MODEL_PIXEL_COUNT];

    resample_nearest(model_image, pong.read(), square_size);

    histogram_equalization(model_image, MODEL_PIXEL_COUNT);

    memcpy(model_input->data.uint8, model_image, MODEL_PIXEL_COUNT);

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status == kTfLiteOk) {
        int part_id = get_top_prediction(model_output, CLASS_COUNT, CLASSES);
        drawNumber(model_image, MODEL_SIZE, part_id, 5, 5, 2);
        // print_top5_predictions(model_output, CLASS_COUNT, CLASSES);
    }

    Serial.write(0xFF);
    Serial.write(0xD8);
    Serial.write(0xFF);
    Serial.write(0xD9);
    Serial.write(model_image, MODEL_PIXEL_COUNT);
}

void ema_update(uint8_t* ema, const uint8_t* in, float alpha) {
    const float inv = 1.0f - alpha;
    for (int i = 0; i < CAMERA_PIXEL_COUNT; i++) {
        ema[i] = ema[i] * inv + in[i] * alpha;
    }
}

void background_subtraction(uint8_t* out, const uint8_t* in) {
    for (int i = 0; i < CAMERA_PIXEL_COUNT; i++) {
        int diff = abs(in[i] - background[i]);
        out[i] = (diff > DETECTION_THRESHOLD) ? in[i] : 0;
    }
}

void erode3x3(uint8_t* out, const uint8_t* in) {
    for (int y = 0; y < CAMERA_HEIGHT; y++) {
        for (int x = 0; x < CAMERA_WIDTH; x++) {
            int idx = y * CAMERA_WIDTH + x;
            if ((x > 0 && !in[idx - 1]) ||
                (x < CAMERA_WIDTH - 1 && !in[idx + 1]) ||
                (y > 0 && !in[idx - CAMERA_WIDTH]) ||
                (y < CAMERA_HEIGHT - 1 && !in[idx + CAMERA_WIDTH])) {
                out[idx] = 0;
            } else {
                out[idx] = in[idx];
            }
        }
    }
}

void dilate3x3(uint8_t* out, const uint8_t* in) {
    for (int y = 0; y < CAMERA_HEIGHT; y++) {
        for (int x = 0; x < CAMERA_WIDTH; x++) {
            int idx = y * CAMERA_WIDTH + x;
            if ((x > 0 && in[idx - 1]) ||
                (x < CAMERA_WIDTH - 1 && in[idx + 1]) ||
                (y > 0 && in[idx - CAMERA_WIDTH]) ||
                (y < CAMERA_HEIGHT - 1 && in[idx + CAMERA_WIDTH])) {
                out[idx] = 255;
            } else {
                out[idx] = in[idx];
            }
        }
    }
}

BoundingBox find_bounding_box(uint8_t* frame) {
    BoundingBox box = {CAMERA_WIDTH, CAMERA_HEIGHT, 0, 0, false};
    int blob_pixels = 0;
    for (int y = 0; y < CAMERA_HEIGHT; y++) {
        for (int x = 0; x < CAMERA_WIDTH; x++) {
            if (!frame[y * CAMERA_WIDTH + x]) {
                continue;
            }
            if (x < box.min_x) {
                box.min_x = x;
            }
            if (x > box.max_x) {
                box.max_x = x;
            }
            if (y < box.min_y) {
                box.min_y = y;
            }
            if (y > box.max_y) {
                box.max_y = y;
            }
            blob_pixels++;
        }
    }
    box.valid = (blob_pixels >= MIN_BLOB_SIZE);
    return box;
}

void draw_box(uint8_t* frame, BoundingBox box) {
    if (!box.valid) {
        return;
    }
    
    for (int x = box.min_x; x <= box.max_x; x++) {
        frame[box.min_y * CAMERA_WIDTH + x] = 255;
        frame[box.max_y * CAMERA_WIDTH + x] = 255;
    }
    
    for (int y = box.min_y; y <= box.max_y; y++) {
        frame[y * CAMERA_WIDTH + box.min_x] = 255;
        frame[y * CAMERA_WIDTH + box.max_x] = 255;
    }
}

void fit_square(BoundingBox &box) {
    int w = box.max_x - box.min_x + 1;
    int h = box.max_y - box.min_y + 1;
    if (w == h) return;
    
    int center_x = (box.min_x + box.max_x) / 2;
    int center_y = (box.min_y + box.max_y) / 2;
    int size = max(w, h);
    bool is_odd = (size & 1);
    
    int radius = size / 2;
    radius = min(radius, center_x);
    radius = min(radius, center_y);
    radius = min(radius, CAMERA_WIDTH - 1 - center_x);
    radius = min(radius, CAMERA_HEIGHT - 1 - center_y);
    
    box.min_x = center_x - radius;
    box.max_x = center_x + radius + is_odd;
    box.min_y = center_y - radius;
    box.max_y = center_y + radius + is_odd;
}

void extract_square(uint8_t* out, const uint8_t* in, const BoundingBox &box, int box_size) {
    for (int y = 0; y < box_size; y++) {
        int yy = box.min_y + y;
        for (int x = 0; x < box_size; x++) {
            int xx = box.min_x + x;
            out[y * box_size + x] = in[yy * CAMERA_WIDTH + xx];
        }
    }
}

void resample_nearest(uint8_t* out, const uint8_t* in, int box_size) {
    for (int y = 0; y < MODEL_SIZE; y++) {
        int yy = (box_size * y) / MODEL_SIZE;
        for (int x = 0; x < MODEL_SIZE; x++) {
            int xx = (box_size * x) / MODEL_SIZE;
            out[y * MODEL_SIZE + x] = in[yy * box_size + xx];
        }
    }
}

// 4x8 font for digits 0-9
const uint8_t font_digits[10][4] = {
  {0x7E, 0x81, 0x81, 0x7E}, // 0
  {0x00, 0x82, 0xFF, 0x80}, // 1
  {0xC2, 0xA1, 0x91, 0x8E}, // 2
  {0x42, 0x81, 0x89, 0x76}, // 3
  {0x18, 0x14, 0x12, 0xFF}, // 4
  {0x4F, 0x89, 0x89, 0x71}, // 5
  {0x7C, 0x89, 0x89, 0x72}, // 6
  {0x01, 0x71, 0x09, 0x07}, // 7
  {0x76, 0x89, 0x89, 0x76}, // 8
  {0x4E, 0x91, 0x91, 0x7E}  // 9
};

// Helper: Draws a single digit
void drawDigit(uint8_t* buffer, int width, int number, int x, int y, int scale) {
    if (number < 0 || number > 9) return; 

    for (int i = 0; i < 4; i++) { 
        uint8_t col = font_digits[number][i];
        for (int j = 0; j < 8; j++) { 
            if (col & (1 << j)) {
                for (int sx = 0; sx < scale; sx++) {
                    for (int sy = 0; sy < scale; sy++) {
                        int px = x + (i * scale) + sx;
                        int py = y + (j * scale) + sy;
                        if (px >= 0 && px < width && py >= 0 && py < 128) {
                            buffer[py * width + px] = 255; // White pixel
                        }
                    }
                }
            }
        }
    }
}

// Main Function: Handles up to 5 digits using pure math
void drawNumber(uint8_t* buffer, int width, int number, int x, int y, int scale) {
    // 1. Safety Caps
    if (number < 0) number = 0;
    if (number > 99999) number = 99999; 

    // 2. Setup Layout
    int digit_width = 4 * scale;
    int spacing = 2 * scale;
    int current_x = x;
    
    // 3. Divisors for 5 digits: 10000, 1000, 100, 10, 1
    int divisor = 10000;
    
    // Flag to skip leading zeros (e.g., don't draw "00123", just "123")
    bool leading_zeros = true;

    // 4. Special Case: If number is exactly 0, draw it and return
    if (number == 0) {
        drawDigit(buffer, width, 0, current_x, y, scale);
        return;
    }

    // 5. Loop through powers of 10
    while (divisor > 0) {
        int digit = number / divisor;
        
        // Logic: If we found a non-zero digit, OR we are already printing numbers
        if (digit != 0 || !leading_zeros) {
            leading_zeros = false; // We found the start of the number
            drawDigit(buffer, width, digit, current_x, y, scale);
            current_x += digit_width + spacing;
        }
        
        // Prepare for next digit
        number %= divisor; // Remove the digit we just processed
        divisor /= 10;     // Move to next power of 10
    }
}

// Histogram Equalization
void histogram_equalization(uint8_t* img, int size) {
    int histogram[256] = {0};
    
    // 1. Compute Histogram
    for (int i = 0; i < size; i++) {
        histogram[img[i]]++;
    }
    
    // 2. Find cdf_min (first non-zero histogram bin)
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            cdf_min = histogram[i];
            break;
        }
    }
    
    // 3. Compute CDF and create Mapping
    int cdf[256] = {0};
    int sum = 0;
    int range = size - cdf_min;
    if (range <= 0) range = 1; // Prevent division by zero
    for (int i = 0; i < 256; i++) {
        sum += histogram[i];
        // Standard Equalization Formula: (cdf(v) - cdf_min) / (total - cdf_min) * (L-1)
        cdf[i] = ((sum - cdf_min) * 255) / range;
        
        // Clamp values just in case
        if (cdf[i] < 0) {
            cdf[i] = 0;
        }
        if (cdf[i] > 255) {
            cdf[i] = 255;
        }
    }
    
    // 4. Apply Mapping
    for (int i = 0; i < size; i++) {
        img[i] = (uint8_t)cdf[img[i]];
    }
}

int get_top_prediction(TfLiteTensor* model_output, int class_count, const int* classes) {
    uint8_t max_val = 0;
    int predicted_class = -1;
    
    for (int i = 0; i < class_count; i++) {
        if (model_output->data.uint8[i] > max_val) {
            max_val = model_output->data.uint8[i];
            predicted_class = i;
        }
    }
    
    int part_id = 0;
    if (predicted_class >= 0 && predicted_class < class_count) {
        part_id = classes[predicted_class];
    }
    
    return part_id;
}

void print_top5_predictions(TfLiteTensor* model_output, int class_count, const int* classes) {
    // Create array of indices and scores
    struct Prediction {
        int class_idx;
        uint8_t score;
    };
    Prediction predictions[class_count];
    
    // Fill predictions array
    for (int i = 0; i < class_count; i++) {
        predictions[i].class_idx = i;
        predictions[i].score = model_output->data.uint8[i];
    }
    
    // Simple selection sort to find top 5
    for (int i = 0; i < 5 && i < class_count; i++) {
        // Find max in remaining elements
        int max_idx = i;
        for (int j = i + 1; j < class_count; j++) {
            if (predictions[j].score > predictions[max_idx].score) {
                max_idx = j;
            }
        }
        // Swap
        Prediction temp = predictions[i];
        predictions[i] = predictions[max_idx];
        predictions[max_idx] = temp;
    }
    
    // Print top 5
    Serial.println("Top 5 predictions:");
    for (int i = 0; i < 5 && i < class_count; i++) {
        int class_idx = predictions[i].class_idx;
        int part_id = classes[class_idx];
        uint8_t score = predictions[i].score;
        
        Serial.print(i + 1);
        Serial.print(". Part ID: ");
        Serial.print(part_id);
        Serial.print(" (class ");
        Serial.print(class_idx);
        Serial.print(") - Score: ");
        Serial.println(score);
    }
    Serial.println();
}

void setup_mpu_for_sdram() {
    // Disable MPU
    HAL_MPU_Disable();
    
    MPU_Region_InitTypeDef MPU_InitStruct = {0};
    
    // Configure SDRAM region as non-cacheable
    MPU_InitStruct.Enable = MPU_REGION_ENABLE;
    MPU_InitStruct.Number = MPU_REGION_NUMBER0;
    MPU_InitStruct.BaseAddress = SDRAM_START_ADDRESS;
    MPU_InitStruct.Size = MPU_REGION_SIZE_8MB;  // Adjust based on your SDRAM size
    MPU_InitStruct.SubRegionDisable = 0x0;
    MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
    MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
    MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
    MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
    MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;  // CRITICAL: Disable cache
    MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    
    HAL_MPU_ConfigRegion(&MPU_InitStruct);
    
    // Enable MPU with default background region
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
    
    Serial.println("MPU configured for SDRAM");
}