#include "coco_detect.hpp"
#include "esp_log.h"
#include "camera_capture.hpp"
#include "esp_camera.h"
#include "sd_handling.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "BleClient.h"
#include <string>
#include <vector>
#include <time.h>
#include "cJSON.h"


const char *TAG = "yolo_main";

static size_t log_psram(const char *label)
{
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "%s - free PSRAM: %u bytes", label, free_psram);
    return free_psram;
}

/**
 * @brief Creates a JSON payload string from detection results.
 *
 * @param detect_results A vector containing the results from the model.
 * @param confidence_threshold The confidence threshold to filter results.
 * @return A std::string containing the formatted JSON payload.
 */
std::string create_json_payload(const std::list<dl::detect::result_t>& detect_results, float confidence_threshold) {
    cJSON *root = cJSON_CreateObject();
    if (root == NULL) {
        ESP_LOGE(TAG, "Failed to create cJSON root object.");
        return "";
    }

    // // 1. Add static and timestamp information
    // cJSON_AddStringToObject(root, "device_id", "cam-01");
    // cJSON_AddStringToObject(root, "location", "placeholder-location");

    // 2. Create the predictions array
    cJSON *predictions = cJSON_CreateArray();
    if (predictions == NULL) {
        ESP_LOGE(TAG, "Failed to create cJSON predictions array.");
        cJSON_Delete(root);
        return "";
    }
    cJSON_AddItemToObject(root, "predictions", predictions);

    int total_detected = 0;
    for (const auto& res : detect_results) {
        if (res.score >= confidence_threshold) {
            total_detected++;
            // cJSON *pred_obj = cJSON_CreateObject();
            // // cJSON_AddNumberToObject(pred_obj, "category", res.category);
            // cJSON_AddNumberToObject(pred_obj, "confidence", res.score);

            // cJSON *bbox = cJSON_CreateArray();
            // cJSON_AddItemToArray(bbox, cJSON_CreateNumber(res.box[0]));
            // cJSON_AddItemToArray(bbox, cJSON_CreateNumber(res.box[1]));
            // cJSON_AddItemToArray(bbox, cJSON_CreateNumber(res.box[2]));
            // cJSON_AddItemToArray(bbox, cJSON_CreateNumber(res.box[3]));
            // cJSON_AddItemToObject(pred_obj, "bbox", bbox);

            // cJSON_AddItemToArray(predictions, pred_obj);
        }
    }

    // 3. Add the total count
    cJSON_AddNumberToObject(root, "total_detected", total_detected);

    // 4. Convert the cJSON object to a string
    char *json_string_ptr = cJSON_PrintUnformatted(root);
    std::string json_payload(json_string_ptr);

    // 5. Clean up
    cJSON_Delete(root);
    free(json_string_ptr);

    return json_payload;
}


// create BLE client instance
BleClient ble_client;

extern "C" void app_main(void)
{
    const float confidence_threshold = 0.1;

    /*
    -----------------------------
    BLE Client connect to server
    -----------------------------
    */

    ble_client.connect_to_server();
    ESP_LOGI(TAG, "Waiting for BLE connection...");
    vTaskDelay(pdMS_TO_TICKS(3000)); // Wait 3 seconds for connection to establish

    // Wait for the BLE client to connect
    while (!ble_client.is_connected()) {
        ESP_LOGW(TAG, "Not connected. Waiting to reconnect...");
        vTaskDelay(pdMS_TO_TICKS(5000)); // Wait 5 seconds before checking again
    }

    ESP_LOGI(TAG, "BLE client connected successfully");

    // Initialize SD card
    esp_err_t sd_ret = init_sd_card();
    if (sd_ret != ESP_OK) {
        ESP_LOGE(TAG, "SD card initilization failed");
    }

    setupCamera();

    while (true) {
        log_psram("Start of loop");

        COCODetect *detect = new COCODetect();

        ESP_LOGI(TAG, "Taking picture...");
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            recalibrateCamera();
            delete detect;
            continue;
        }

        dl::image::jpeg_img_t jpeg_img = {.data = fb->buf, .data_len = fb->len};
        auto img = sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

        if (!img.data) {
             ESP_LOGE(TAG, "Failed to decode JPEG");
             esp_camera_fb_return(fb); 
             delete detect;
             continue;
        }

        ESP_LOGI(TAG, "Running detection on captured image...");
        auto &detect_results = detect->run(img);

        int saddle_count = 0;
        int bike_count = 0;

        if (detect_results.size() > 0) {
            ESP_LOGI(TAG, "Number of detected objects: %d", detect_results.size());
            for (const auto &res : detect_results) {
                if (res.score >= confidence_threshold) {
                    ESP_LOGI(TAG,
                             "[category: %d, score: %.2f, box: (%d, %d, %d, %d)]",
                             res.category,
                             res.score,
                             res.box[0], res.box[1], res.box[2], res.box[3]);

                    if (res.category == 0) bike_count++;
                    else if (res.category == 1) saddle_count++;
                }
            }
        } else {
            ESP_LOGI(TAG, "No objects detected.");
        }

        ESP_LOGI(TAG, "-> Found %d bikes and %d saddles in this frame.", bike_count, saddle_count);
        
        // Save image to SD
        if (is_sd_card_mounted()) {
            ESP_LOGI(TAG, "Saving image and detection results to SD card...");
            
            char image_filename[64];
            char results_filename[64];
            int counter = get_image_counter();

            snprintf(image_filename, sizeof(image_filename), "detection_%04d.jpg", counter); 
            snprintf(results_filename, sizeof(results_filename), "detection_%04d.txt", counter);
            
            esp_err_t img_ret = save_jpeg(fb, image_filename);
            if (img_ret == ESP_OK) {
                ESP_LOGI(TAG, "Image saved successfully");
            } else {
                ESP_LOGE(TAG, "Failed to save image");
            }
            
            if (bike_count > 0) {
                esp_err_t results_ret = save_detection_results(detect_results, confidence_threshold, results_filename);
                if (results_ret == ESP_OK) {
                    ESP_LOGI(TAG, "Detection results saved successfully");
                } else {
                    ESP_LOGE(TAG, "Failed to save detection results");
                }
            } else {
                ESP_LOGI(TAG, "No bikes detected, skipping results save");
            }
            
            increment_image_counter();
        } else if (is_sd_card_mounted()) {
            ESP_LOGI(TAG, "No objects detected, skipping SD card save");
        }

        /*
        ----------------------------------------------
        BLE client sending model predictions to server
        ----------------------------------------------
        */

        // Check if the BLE client is fully connected and ready
        if (ble_client.is_connected()) {
            // Create the JSON payload from the detection results
            std::string payload = create_json_payload(detect_results, confidence_threshold);

            if (!payload.empty()) {
                ESP_LOGI(TAG, "Sending payload via BLE");
                ble_client.send_data(payload);
            }
        } else {
            ESP_LOGW(TAG, "BLE not connected. Skipping data send.");
        }

        // Only return the frame buffer after all operations are done to avoid issues with accessing the frame buffer after it has been returned
        esp_camera_fb_return(fb);

        // Now free the other resources
        heap_caps_free(img.data);
        delete detect;

        log_psram("End of loop");
        ESP_LOGI(TAG, "----------------------------------\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }

}