#include <Arduino.h>
#include <Wire.h>
#include <U8x8lib.h>

#include "data.h"
#include "cole_estimator.h"

#define BUZZER_PIN A3
#define BEEP_FREQ  2700
#define BEEP_SHORT 150
#define BEEP_LONG  (2 * BEEP_SHORT)

U8X8_SSD1306_128X64_NONAME_HW_I2C u8x8(U8X8_PIN_NONE);

void beep(unsigned long duration_ms)
{
    tone(BUZZER_PIN, BEEP_FREQ);
    delay(duration_ms);
    noTone(BUZZER_PIN);
}

void showLine(uint8_t row, const String& text)
{
    u8x8.clearLine(row);
    u8x8.drawString(0, row, text.c_str());
}

void setup()
{
    Serial.begin(115200);
    unsigned long t_wait = millis();
    while (!Serial && millis() - t_wait < 2000) { }

    pinMode(BUZZER_PIN, OUTPUT);

    u8x8.begin();
    u8x8.setFlipMode(1);
    u8x8.setFont(u8x8_font_chroma48medium8_r);
    u8x8.clearDisplay();

    showLine(0, "Cole estimation");
    showLine(2, "N = " + String(N_DATA));
    showLine(4, "Running...");

    Serial.println();
    Serial.println("Estimation started...");

    beep(BEEP_SHORT);

    unsigned long t0 = micros();
    ColeResult res = cole_estimate(frequency, R_data, N_DATA);
    unsigned long t1 = micros();

    float exe_ms = (t1 - t0) / 1000.0f;

    beep(BEEP_LONG);

    u8x8.clearDisplay();
    showLine(0, "Cole estimation");
    showLine(2, "R0  =" + String(res.R0, 4));
    showLine(3, "Rinf=" + String(res.Rinf, 4));
    showLine(4, "tau =" + String(res.tau * 1e6, 4) + "us");
    showLine(5, "alfa=" + String(res.alpha, 3));
    showLine(7, "t   =" + String(exe_ms, 1) + "ms");

    Serial.println("========================================================");
    Serial.println("        COLE MODEL - ESTIMATED VALUES USING ONLY Re{Z}");
    Serial.println("========================================================");
    Serial.print("R0_est       = "); Serial.print(res.R0, 6);        Serial.println(" Ohm");
    Serial.print("Rinf_est     = "); Serial.print(res.Rinf, 6);      Serial.println(" Ohm");
    Serial.print("tau_est      = "); Serial.print(res.tau * 1e6, 6); Serial.println(" us");
    Serial.print("alpha_est    = "); Serial.println(res.alpha, 3);
    Serial.print("SSE          = "); Serial.println(res.SSE, 6);
    Serial.print("RMSE         = "); Serial.print(sqrt(res.SSE / N_DATA), 6); Serial.println(" Ohm");
    Serial.print("exe time     = "); Serial.print(exe_ms, 1);       Serial.println(" ms");
}

void loop()
{
}
