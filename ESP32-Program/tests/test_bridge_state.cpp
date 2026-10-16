#include "bridge_state.hpp"
#include "modbus_bridge.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>

#define TEST_ASSERT(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << " Assertion failed: " #expr << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void test_initial_state() {
  pbl::BridgeState bridge;
  pbl::PlcStatus status = bridge.get_plc_status();

  TEST_ASSERT(!status.running);
  TEST_ASSERT(!status.stopped);
  TEST_ASSERT(!status.emergency);
  TEST_ASSERT(status.last_latency_us == 0);
  TEST_ASSERT(!bridge.get_user_detection());

  std::cout << "  [PASS] test_initial_state" << std::endl;
}

void test_command_triggers_and_per_channel_latency() {
  pbl::BridgeState bridge;

  // 1. Channel 1: Start
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Start, 1000));
  TEST_ASSERT(!bridge.consume_stop_request(1100));
  TEST_ASSERT(!bridge.consume_emergency_request(1100));
  TEST_ASSERT(!bridge.consume_reset_request(1100));
  TEST_ASSERT(bridge.consume_start_request(1250));
  TEST_ASSERT(bridge.get_last_latency_us() == 250);
  TEST_ASSERT(!bridge.consume_start_request(1250));

  // 2. Channel 2: Stop
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Stop, 2000));
  TEST_ASSERT(bridge.consume_stop_request(2350));
  TEST_ASSERT(bridge.get_last_latency_us() == 350);
  TEST_ASSERT(!bridge.consume_stop_request(2350));

  // 3. Channel 3: Emergency
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Emergency, 5000));
  TEST_ASSERT(bridge.consume_emergency_request(5420));
  TEST_ASSERT(bridge.get_last_latency_us() == 420);
  TEST_ASSERT(!bridge.consume_emergency_request(5420));

  // 4. Channel 4: Reset
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Reset, 8000));
  TEST_ASSERT(bridge.consume_reset_request(8150));
  TEST_ASSERT(bridge.get_last_latency_us() == 150);
  TEST_ASSERT(!bridge.consume_reset_request(8150));

  // 5. Unknown command must not arm any channel or overwrite timestamps
  TEST_ASSERT(!bridge.trigger_command(pbl::CommandType::Unknown, 9999));
  TEST_ASSERT(!bridge.consume_start_request(10000));
  TEST_ASSERT(!bridge.consume_stop_request(10000));
  TEST_ASSERT(!bridge.consume_emergency_request(10000));
  TEST_ASSERT(!bridge.consume_reset_request(10000));
  TEST_ASSERT(bridge.get_last_latency_us() == 150);

  std::cout << "  [PASS] test_command_triggers_and_per_channel_latency"
            << std::endl;
}

void test_overlapping_channels_do_not_clobber_timestamps() {
  pbl::BridgeState bridge;

  // Trigger Start at t=1000, then Stop at t=1800, then Emergency at t=2100
  // BEFORE Core 1 consumes any of them
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Start, 1000));
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Stop, 1800));
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Emergency, 2100));

  // Consuming Start at t=2500 must compute 2500 - 1000 = 1500 us
  TEST_ASSERT(bridge.consume_start_request(2500));
  TEST_ASSERT(bridge.get_last_latency_us() == 1500);

  // Consuming Stop at t=2500 must compute 2500 - 1800 = 700 us
  TEST_ASSERT(bridge.consume_stop_request(2500));
  TEST_ASSERT(bridge.get_last_latency_us() == 700);

  // Consuming Emergency at t=2500 must compute 2500 - 2100 = 400 us
  TEST_ASSERT(bridge.consume_emergency_request(2500));
  TEST_ASSERT(bridge.get_last_latency_us() == 400);

  std::cout << "  [PASS] test_overlapping_channels_do_not_clobber_timestamps"
            << std::endl;
}

void test_micros_32bit_wraparound_and_underflow_protection() {
  pbl::BridgeState bridge;

  // 1. 32-bit micros() wrap-around: 0xFFFFFFF0 -> 0x00000030 (elapsed = 0x40 =
  // 64 us)
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Start, 0xFFFFFFF0u));
  TEST_ASSERT(bridge.consume_start_request(0x00000030u));
  TEST_ASSERT(bridge.get_last_latency_us() == 0x40u);

  // 2. Zero timestamp on trigger (t = 0) does not overwrite valid latency
  pbl::BridgeState bridge_zero;
  TEST_ASSERT(bridge_zero.trigger_command(pbl::CommandType::Start, 0));
  TEST_ASSERT(bridge_zero.consume_start_request(100));
  TEST_ASSERT(bridge_zero.get_last_latency_us() == 0);

  // 3. Backward clock anomaly (t_consume = 4000 < t_trigger = 5000) is
  // rejected
  TEST_ASSERT(bridge_zero.trigger_command(pbl::CommandType::Stop, 5000));
  TEST_ASSERT(bridge_zero.consume_stop_request(4000));
  TEST_ASSERT(bridge_zero.get_last_latency_us() == 0);

  // 4. Exact same timestamp (0 us latency)
  TEST_ASSERT(bridge.trigger_command(pbl::CommandType::Emergency, 10000));
  TEST_ASSERT(bridge.consume_emergency_request(10000));
  TEST_ASSERT(bridge.get_last_latency_us() == 0);

  std::cout << "  [PASS] test_micros_32bit_wraparound_and_underflow_protection"
            << std::endl;
}

void test_momentary_pulse_timers_and_millis_wraparound() {
  pbl::BridgeState bridge;

  // Inactive timers must not clear
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn1(100, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn2(100, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn3(100, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn4(100, 500));

  // Start all 4 across 32-bit millis() wrap-around boundary: 0xFFFFFF00 (256 ms
  // before wrap)
  const uint32_t t_start = 0xFFFFFF00u;
  bridge.start_pulse_btn1(t_start);
  bridge.start_pulse_btn2(t_start);
  bridge.start_pulse_btn3(t_start);
  bridge.start_pulse_btn4(t_start);

  // At 0x000000F3 (elapsed = 499 ms < 500 ms): still active
  const uint32_t t_before = t_start + 499u; // 0x000000F3u
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn1(t_before, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn2(t_before, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn3(t_before, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn4(t_before, 500));

  // At 0x00000150 (elapsed = 0x250 = 592 ms >= 500 ms): all expire and clear
  const uint32_t t_after = 0x00000150u;
  TEST_ASSERT(bridge.check_and_clear_pulse_btn1(t_after, 500));
  TEST_ASSERT(bridge.check_and_clear_pulse_btn2(t_after, 500));
  TEST_ASSERT(bridge.check_and_clear_pulse_btn3(t_after, 500));
  TEST_ASSERT(bridge.check_and_clear_pulse_btn4(t_after, 500));

  // Subsequent checks return false
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn1(t_after + 1, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn2(t_after + 1, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn3(t_after + 1, 500));
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn4(t_after + 1, 500));

  // Pulse re-trigger extension test
  bridge.start_pulse_btn1(100);
  bridge.start_pulse_btn1(400);
  TEST_ASSERT(!bridge.check_and_clear_pulse_btn1(600, 500));
  TEST_ASSERT(bridge.check_and_clear_pulse_btn1(900, 500));

  std::cout << "  [PASS] test_momentary_pulse_timers_and_millis_wraparound"
            << std::endl;
}

void test_user_detection_packed_atomic_latch() {
  pbl::BridgeState bridge;
  bool current = false;

  TEST_ASSERT(!bridge.consume_user_detection_changed(current));

  // Transition false -> true sets dirty bit and value bit atomically
  bridge.set_user_detection(true);
  TEST_ASSERT(bridge.get_user_detection() == true);
  TEST_ASSERT(bridge.consume_user_detection_changed(current));
  TEST_ASSERT(current == true);
  TEST_ASSERT(!bridge.consume_user_detection_changed(current));

  // Redundant true does not set dirty bit
  bridge.set_user_detection(true);
  TEST_ASSERT(!bridge.consume_user_detection_changed(current));

  // Rapid toggle true -> false -> true before consumption
  bridge.set_user_detection(false);
  bridge.set_user_detection(true);
  TEST_ASSERT(bridge.consume_user_detection_changed(current));
  TEST_ASSERT(current == true);
  TEST_ASSERT(!bridge.consume_user_detection_changed(current));

  // Transition true -> false
  bridge.set_user_detection(false);
  TEST_ASSERT(bridge.consume_user_detection_changed(current));
  TEST_ASSERT(current == false);
  TEST_ASSERT(!bridge.consume_user_detection_changed(current));

  std::cout << "  [PASS] test_user_detection_packed_atomic_latch" << std::endl;
}

void test_plc_status_and_json_serialization() {
  pbl::BridgeState bridge;

  std::string json0 = bridge.generate_status_json();
  TEST_ASSERT(json0 ==
              "{\"running\":false,\"stopped\":false,\"emergency\":false,"
              "\"last_internal_latency_us\":0}");

  bridge.update_plc_status(true, false, true);
  bridge.trigger_command(pbl::CommandType::Start, 1000);
  bridge.consume_start_request(1543);

  std::string json1 = bridge.generate_status_json();
  TEST_ASSERT(json1 == "{\"running\":true,\"stopped\":false,\"emergency\":true,"
                       "\"last_internal_latency_us\":543}");

  std::cout << "  [PASS] test_plc_status_and_json_serialization" << std::endl;
}

void test_multithreaded_core0_core1_concurrency() {
  pbl::BridgeState bridge;
  constexpr int ITERATIONS = 10000;
  std::atomic<bool> producer_done{false};
  std::atomic<int> start_consumed{0};
  std::atomic<int> stop_consumed{0};
  std::atomic<bool> last_consumed_detect{false};

  // Core 1 Consumer & PLC Status Writer Thread
  std::thread core1([&]() {
    uint32_t sim_us = 10000;
    uint32_t step = 0;
    while (!producer_done.load(std::memory_order_acquire)) {
      sim_us += 10;
      if (bridge.consume_start_request(sim_us)) {
        start_consumed.fetch_add(1, std::memory_order_relaxed);
      }
      if (bridge.consume_stop_request(sim_us)) {
        stop_consumed.fetch_add(1, std::memory_order_relaxed);
      }
      bool det = false;
      if (bridge.consume_user_detection_changed(det)) {
        last_consumed_detect.store(det, std::memory_order_relaxed);
      }
      // Write valid mutually-exclusive PLC states: (run=true, stop=false) or
      // (run=false, stop=true)
      bool run_state = (step % 2 == 0);
      bridge.update_plc_status(run_state, !run_state, false);
      step++;
    }

    // Drain any final pending state after producer finishes
    sim_us += 10;
    if (bridge.consume_start_request(sim_us)) {
      start_consumed.fetch_add(1, std::memory_order_relaxed);
    }
    if (bridge.consume_stop_request(sim_us)) {
      stop_consumed.fetch_add(1, std::memory_order_relaxed);
    }
    bool det = false;
    if (bridge.consume_user_detection_changed(det)) {
      last_consumed_detect.store(det, std::memory_order_relaxed);
    }
  });

  // Core 0 Producer & Status Reader Thread
  std::thread core0([&]() {
    for (int i = 1; i <= ITERATIONS; ++i) {
      uint32_t ts = static_cast<uint32_t>(i * 10);
      bridge.trigger_command(pbl::CommandType::Start, ts);
      bridge.trigger_command(pbl::CommandType::Stop, ts + 5);

      bool target_det = (i % 2 == 1); // Ends with i=10000 -> false, then true
      bridge.set_user_detection(target_det);

      // Verify PLC status never tears (running and stopped are always
      // complementary)
      pbl::PlcStatus st = bridge.get_plc_status();
      if (st.running || st.stopped) {
        TEST_ASSERT(st.running != st.stopped);
        TEST_ASSERT(!st.emergency);
      }
    }
    // Final deterministic user detection state = true
    bridge.set_user_detection(false);
    bridge.set_user_detection(true);
    producer_done.store(true, std::memory_order_release);
  });

  core0.join();
  core1.join();

  TEST_ASSERT(start_consumed.load() > 0);
  TEST_ASSERT(stop_consumed.load() > 0);
  // Final user detection state must be converged to true with zero lost wakeup
  TEST_ASSERT(bridge.get_user_detection() == true);
  TEST_ASSERT(last_consumed_detect.load() == true);

  std::cout << "  [PASS] test_multithreaded_core0_core1_concurrency"
            << std::endl;
}

void test_modbus_bridge_host_integration() {
  pbl::BridgeState state;
  pbl::ModbusBridge bridge;

  bridge.init();
  TEST_ASSERT(bridge.get_connection_count() == 0);

  state.trigger_command(pbl::CommandType::Start, 1000);
  bridge.update(state, 100, 1200);
  TEST_ASSERT(state.get_last_latency_us() == 200);

  bridge.update(state, 300, 1300);
  bridge.update(state, 650, 1650);

  std::cout << "  [PASS] test_modbus_bridge_host_integration" << std::endl;
}

int main() {
  std::cout << "--- Running BridgeState Unit Tests ---" << std::endl;
  test_initial_state();
  test_command_triggers_and_per_channel_latency();
  test_overlapping_channels_do_not_clobber_timestamps();
  test_micros_32bit_wraparound_and_underflow_protection();
  test_momentary_pulse_timers_and_millis_wraparound();
  test_user_detection_packed_atomic_latch();
  test_plc_status_and_json_serialization();
  test_multithreaded_core0_core1_concurrency();
  test_modbus_bridge_host_integration();
  std::cout << "All BridgeState tests PASSED!" << std::endl;
  return 0;
}
