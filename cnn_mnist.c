/**
 * @file cnn_mnist.c
 * @brief Inferência de Letras (EMNIST A-Z) - Versão Compatível Pico W
 */

#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"

// Headers do projeto
#include "tflm_wrapper.h"
#include "mnist_sample.h"
#include "mnist_cnn_int8_model.h"

// ==========================================
// Funções Auxiliares
// ==========================================

static int8_t quantize_f32_to_i8(float x, float scale, int zero_point)
{
    float tmp = x / scale + (float)zero_point;
    int32_t q = (int32_t)roundf(tmp);
    if (q < -128)
        q = -128;
    if (q > 127)
        q = 127;
    return (int8_t)q;
}

static float dequantize_i8_to_f32(int8_t q, float scale, int zero_point)
{
    return ((float)q - (float)zero_point) * scale;
}

void print_ascii_art(const uint8_t *image_data)
{
    printf("\n--- O QUE O PICO ESTA VENDO ---\n");
    for (int y = 0; y < 28; y++)
    {
        for (int x = 0; x < 28; x++)
        {
            uint8_t pixel = image_data[y * 28 + x];
            if (pixel > 100)
                printf("##");
            else
                printf("..");
        }
        printf("\n");
    }
    printf("-------------------------------\n");
}

// ==========================================
// MAIN
// ==========================================
int main()
{
    stdio_init_all();

    // Loop infinito aguardando conexão serial para não perdermos o print
    // Pode remover o while(!stdio_usb_connected()); se quiser bootar direto
    sleep_ms(3000);

    printf("\n\n=== INICIO DO SISTEMA DE LEITURA DE LETRAS ===\n");

    if (tflm_init() != 0)
    {
        printf("ERRO: Falha ao inicializar TFLM!\n");
        while (1)
            tight_loop_contents();
    }
    printf("TFLM Inicializado.\n");

    sleep_ms(3000);
    
    int in_bytes = 0, out_bytes = 0;
    int8_t *in_ptr = tflm_input_ptr(&in_bytes);
    int8_t *out_ptr = tflm_output_ptr(&out_bytes);

    float in_scale = tflm_input_scale();
    int in_zp = tflm_input_zero_point();
    float out_scale = tflm_output_scale();
    int out_zp = tflm_output_zero_point();

    // Debug Visual
    print_ascii_art(mnist_sample_28x28);
    printf("Label Esperado: %d -> Letra '%c'\n",
           mnist_sample_label, mnist_sample_label + 'A');

    // Prepara Input
    for (int i = 0; i < 28 * 28; i++)
    {
        float pixel_norm = (float)mnist_sample_28x28[i] / 255.0f;
        in_ptr[i] = quantize_f32_to_i8(pixel_norm, in_scale, in_zp);
    }

    // Roda Inferência
    printf("Pensando...\n");
    if (tflm_invoke() != 0)
    {
        printf("ERRO na inferencia.\n");
        while (1)
            tight_loop_contents();
    }

    // Interpreta Resultados
    int8_t max_val = -128;
    int max_idx = 0;

    printf("\n--- Probabilidades ---\n");
    for (int i = 0; i < 26; i++)
    {
        float prob = dequantize_i8_to_f32(out_ptr[i], out_scale, out_zp);

        if (prob > 0.05)
        {
            printf("Letra '%c': %d (%.1f%%)\n", i + 'A', out_ptr[i], prob * 100.0f);
        }

        if (out_ptr[i] > max_val)
        {
            max_val = out_ptr[i];
            max_idx = i;
        }
    }

    float confianca = dequantize_i8_to_f32(max_val, out_scale, out_zp);
    char letra_final = max_idx + 'A';

    printf("\n=========================================\n");
    printf(" PREVISAO DA IA: %c\n", letra_final);
    printf(" CONFIANCA:      %.1f%%\n", confianca * 100.0f);
    printf("=========================================\n");

    // Fim - Pisca LED Removido para compatibilidade Pico W
    // Apenas imprime "FIM" repetidamente
    while (1)
    {
        printf("Aguardando...\n");
        sleep_ms(2000);
    }

    return 0;
}