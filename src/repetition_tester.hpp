#pragma once
#include <cstdint>

enum TestMode : uint32_t
{
    UNINITIALIZED,
    TESTING,
    COMPLETED,
    ERRORR,
};

enum RepetitionValueType
{
    TESTCOUNT,

    CPUTIMER,
    MEMPAGEFAULTS,
    BYTECOUNT,
    OPCOUNT,

    COUNT,
};

struct RepetitionValue
{
    uint64_t e[RepetitionValueType::COUNT];
};

struct RepetitionTestResults
{
    RepetitionValue total_;
    RepetitionValue min_;
    RepetitionValue max_;
};

struct RepetitionTester
{
    uint64_t target_processed_byte_count_;
    uint64_t cpu_timer_freq_;
    uint64_t try_for_time_;
    uint64_t tests_started_at_;

    TestMode mode_;
    bool print_new_minimums_;
    uint32_t open_block_count_;
    uint32_t close_block_count_;

    RepetitionValue accumulated_on_this_test_;
    RepetitionTestResults results_;
};

double SecondsFromCpuTime(double cpu_time, uint64_t cpu_timer_freq);
void PrintValue(char const *label, RepetitionValue value, uint64_t cpu_timer_freq);
void PrintResults(const RepetitionTestResults &results, uint64_t cpu_timer_freq, uint64_t byte_count);
void Error(RepetitionTester &tester, char const *error_message);
void NewTestWave(RepetitionTester &tester, uint64_t target_processed_byte_count, uint64_t cpu_timer_freq, uint32_t seconds_to_try = 10);
void BeginTime(RepetitionTester &tester);
void EndTime(RepetitionTester &tester);
void CountBytes(RepetitionTester &tester, uint64_t byte_count);
void CountOps(RepetitionTester &tester, uint64_t op_count);
bool IsTesting(RepetitionTester &tester);
