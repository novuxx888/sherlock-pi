#include "main.h"


// Initialization

// Work Init
void buttonLight(void);


// Utility Init
void SystemClock_Config(void); // set up by STM IDE
void GPIO_LED_Init(void);
void GPIO_PA0_Init(void);

int main(void)
{
    HAL_Init(); // set up by STM IDE
    SystemClock_Config();  // You can keep CubeMX’s version here
    GPIO_LED_Init();           // Our custom GPIO setup

    // LED set up
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET); // Turn OFF

    while (1)
    {
        buttonLight();
    }
}
// Working Functions
void buttonLight(void)
{
	// User Button PC13
	GPIO_PinState buttonState = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13);
	if (buttonState == GPIO_PIN_RESET){
		HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5); // Toggle LED
        HAL_Delay(200); // Wait 200 ms to debounce
	}
}


// Utility
// GPIO Setup
void GPIO_LED_Init(void)
{
  //  LED and User Button
    __HAL_RCC_GPIOA_CLK_ENABLE();  // Enable GPIOA clock
    __HAL_RCC_GPIOC_CLK_ENABLE();  // Enable GPIOC clock

    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Configure PA5 as output (LED)
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;  // Push-pull output
    GPIO_InitStruct.Pull = GPIO_NOPULL;          // No internal pull-up or pull-down
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure PC13 as input (Button)
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLDOWN;        // Use pull-down (depends on your board)
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}
