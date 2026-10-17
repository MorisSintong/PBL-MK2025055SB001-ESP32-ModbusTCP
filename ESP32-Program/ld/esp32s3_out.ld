/* ESP32-S3 Native Bare-Metal / ROM-Backed Linker Script */

ENTRY(call_start_cpu0)

/* ESP32-S3 Mask ROM Function Symbols (ESP-IDF v5.5 ROM Table) */
PROVIDE(ets_printf            = 0x400005d0);
PROVIDE(ets_delay_us          = 0x40000600);
PROVIDE(uart_tx_one_char      = 0x40000648);
PROVIDE(uart_tx_one_char2     = 0x40000654);
PROVIDE(uart_rx_one_char      = 0x40000660);
PROVIDE(uart_tx_flush         = 0x40000690);
PROVIDE(uart_tx_wait_idle     = 0x4000069c);
PROVIDE(uart_div_modify       = 0x400006a8);
PROVIDE(ets_write_char_uart   = 0x400006b4);
PROVIDE(software_reset        = 0x400006d8);
PROVIDE(ets_get_cpu_frequency = 0x40001a40);
PROVIDE(gpio_input_get        = 0x40001a88);
PROVIDE(gpio_output_set       = 0x40001ac4);
PROVIDE(gpio_pad_input_enable = 0x40001ae8);
PROVIDE(gpio_pad_pullup       = 0x40001b00);
PROVIDE(gpio_pad_select_gpio  = 0x40001b0c);

MEMORY
{
    /* SRAM1 Instruction Bus (0x40380000 .. 0x403A0000 -> physical 0x3FC90000 .. 0x3FCB0000) */
    IRAM (rwx) : ORIGIN = 0x40380000, LENGTH = 128K

    /* SRAM1 Data Bus (0x3FCB0000 .. 0x3FCD8000, non-overlapping with IRAM and ROM stack) */
    DRAM (rw)  : ORIGIN = 0x3FCB0000, LENGTH = 160K
}

SECTIONS
{
    .iram.text : ALIGN(4)
    {
        _iram_start = ABSOLUTE(.);
        KEEP(*(.entry.literal))
        KEEP(*(.entry.text))
        *(.literal .literal.* .text .text.* .stub .gnu.linkonce.literal.* .gnu.linkonce.t.*)
        . = ALIGN(4);
        _iram_end = ABSOLUTE(.);
    } > IRAM

    .dram.data : ALIGN(4)
    {
        _data_start = ABSOLUTE(.);
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.data .data.* .gnu.linkonce.d.*)
        *(.sdata .sdata.* .gnu.linkonce.s.*)
        *(.gcc_except_table .gcc_except_table.*)
        *(.eh_frame .eh_frame.*)

        . = ALIGN(4);
        __init_array_start = ABSOLUTE(.);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        __init_array_end = ABSOLUTE(.);

        . = ALIGN(4);
        _data_end = ABSOLUTE(.);
    } > DRAM

    .dram.bss (NOLOAD) : ALIGN(8)
    {
        _bss_start = ABSOLUTE(.);
        *(.dynsbss)
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        *(.scommon)
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(8);
        _bss_end = ABSOLUTE(.);
        _end = ABSOLUTE(.);
        PROVIDE(end = ABSOLUTE(.));
    } > DRAM
}
