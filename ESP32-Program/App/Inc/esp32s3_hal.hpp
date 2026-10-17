#pragma once

#include <cstdint>

#if defined(ESP32S3_TARGET)

extern "C" {
int ets_printf(const char *fmt, ...);
void ets_delay_us(uint32_t us);
int uart_tx_one_char(uint8_t ch);
int uart_rx_one_char(uint8_t *ch);
void uart_tx_wait_idle(uint8_t uart_no);
void uart_div_modify(uint8_t uart_no, uint32_t div_latch);
uint32_t ets_get_cpu_frequency(void);
uint32_t gpio_input_get(void);
void gpio_output_set(uint32_t set_mask, uint32_t clear_mask,
                     uint32_t enable_mask, uint32_t disable_mask);
void gpio_pad_input_enable(uint32_t gpio_num);
void gpio_pad_pullup(uint32_t gpio_num);
void gpio_pad_select_gpio(uint8_t gpio_num);
}

namespace pbl {
namespace hal {

inline volatile uint32_t &reg32(uint32_t addr) {
  return *reinterpret_cast<volatile uint32_t *>(addr);
}

inline void disable_boot_watchdogs() {
  // Disable Timer Group 0 MWDT0 (0x6001F000)
  reg32(0x6001F064U) = 0x50D83AA1U;
  reg32(0x6001F048U) = 0U;

  // Disable Timer Group 1 MWDT1 (0x60020000)
  reg32(0x60020064U) = 0x50D83AA1U;
  reg32(0x60020048U) = 0U;

  // Disable RTC Watchdog RWDT (0x60008000)
  reg32(0x600080B0U) = 0x50D83AA1U;
  reg32(0x60008098U) = 0U;

  // Enable Super Watchdog SWD auto-feed
  reg32(0x600080B8U) = 0x8F1D312AU;
  reg32(0x600080B4U) |= (1U << 31);
}

inline uint32_t read_ccount() {
  uint32_t ccount = 0;
  __asm__ volatile("rsr.ccount %0" : "=a"(ccount));
  return ccount;
}

class SystemClock {
public:
  static void update() {
    uint32_t now = read_ccount();
    uint32_t delta = now - last_ccount_;
    last_ccount_ = now;
    total_cycles_ += delta;
  }

  static uint32_t micros() {
    update();
    uint32_t mhz = ets_get_cpu_frequency();
    if (mhz == 0) {
      mhz = 40;
    }
    return static_cast<uint32_t>(total_cycles_ / mhz);
  }

  static uint32_t millis() { return micros() / 1000U; }

  static void delay_ms(uint32_t ms) {
    for (uint32_t i = 0; i < ms; ++i) {
      ets_delay_us(1000U);
      update();
    }
  }

private:
  static inline uint32_t last_ccount_ = 0;
  static inline uint64_t total_cycles_ = 0;
};

// Bit-banged SPI Mode 0 probe for W5500 Version Register (0x0039, Control 0x00)
inline uint8_t probe_w5500_version_spi(uint8_t cs_pin, uint8_t mosi_pin,
                                       uint8_t sclk_pin, uint8_t miso_pin) {
  gpio_pad_select_gpio(cs_pin);
  gpio_pad_select_gpio(mosi_pin);
  gpio_pad_select_gpio(sclk_pin);
  gpio_pad_select_gpio(miso_pin);

  const uint32_t cs_mask = (1UL << cs_pin);
  const uint32_t mosi_mask = (1UL << mosi_pin);
  const uint32_t sclk_mask = (1UL << sclk_pin);
  const uint32_t out_enable = cs_mask | mosi_mask | sclk_mask;

  // CS HIGH, SCLK LOW, MOSI LOW
  gpio_output_set(cs_mask, mosi_mask | sclk_mask, out_enable, 0);
  gpio_pad_input_enable(miso_pin);
  gpio_pad_pullup(miso_pin);
  ets_delay_us(10);

  // Assert CS LOW
  gpio_output_set(0, cs_mask, out_enable, 0);
  ets_delay_us(2);

  // W5500 frame: Addr=0x0039 (16-bit), Control=0x00 (8-bit, Common Reg Read),
  // Data=0x00 (8-bit clock-out)
  const uint32_t tx_frame = 0x00390000UL;
  uint8_t rx_byte = 0;

  for (int bit = 31; bit >= 0; --bit) {
    if ((tx_frame >> bit) & 1U) {
      gpio_output_set(mosi_mask, 0, out_enable, 0);
    } else {
      gpio_output_set(0, mosi_mask, out_enable, 0);
    }
    ets_delay_us(1);
    // SCLK HIGH (sample MISO)
    gpio_output_set(sclk_mask, 0, out_enable, 0);
    ets_delay_us(1);
    if (bit < 8) {
      uint32_t in_val = (gpio_input_get() >> miso_pin) & 1U;
      rx_byte = static_cast<uint8_t>((rx_byte << 1) | in_val);
    }
    // SCLK LOW
    gpio_output_set(0, sclk_mask, out_enable, 0);
  }

  // Deassert CS HIGH
  ets_delay_us(2);
  gpio_output_set(cs_mask, mosi_mask | sclk_mask, out_enable, 0);
  return rx_byte;
}

} // namespace hal
} // namespace pbl

#endif // defined(ESP32S3_TARGET)
