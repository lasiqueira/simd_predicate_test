#include <cstdio>
#include "repetition_tester.hpp"
#include "platform_metrics.hpp"

#define ArrayCount(array) (sizeof(array) / sizeof((array)[0]))

double SecondsFromCpuTime(double cpu_time, uint64_t cpu_timer_freq)
{
    double result = 0.0;
    if (cpu_timer_freq)
    {
        result = (cpu_time / (double)cpu_timer_freq);
    }

    return result;
}

void PrintValue(char const *label, RepetitionValue value, uint64_t cpu_timer_freq)
{
    uint64_t test_count = value.e[RepetitionValueType::TESTCOUNT];
    double divisor = test_count ? (double)test_count : 1;

    double e[RepetitionValueType::COUNT];
    for (uint32_t e_index = 0; e_index < ArrayCount(e); ++e_index)
    {
        e[e_index] = (double)value.e[e_index] / divisor;
    }

    printf("%s: %.0f", label, e[RepetitionValueType::CPUTIMER]);
    if (cpu_timer_freq)
    {
        double seconds = SecondsFromCpuTime(e[RepetitionValueType::CPUTIMER], cpu_timer_freq);
        printf(" (%fms)", 1000.0f * seconds);

        if (e[RepetitionValueType::OPCOUNT] > 0)
        {
            double giga_ops = e[RepetitionValueType::OPCOUNT] / (1000000000.0 * seconds);
            printf(" %fGOp/s", giga_ops);
        }

        if (e[RepetitionValueType::BYTECOUNT] > 0)
        {
            double gigabyte = (1024.0f * 1024.0f * 1024.0f);
            double bandwidth = e[RepetitionValueType::BYTECOUNT] / (gigabyte * seconds);
            printf(" %fgb/s", bandwidth);
        }
    }

    if (e[RepetitionValueType::MEMPAGEFAULTS] > 0)
    {
        printf(" PF: %0.4f (%0.4fk/fault)", e[RepetitionValueType::MEMPAGEFAULTS], e[RepetitionValueType::BYTECOUNT] / (e[RepetitionValueType::MEMPAGEFAULTS] * 1024.0));
    }
}

void PrintResults(const RepetitionTestResults &results, uint64_t cpu_timer_freq, uint64_t byte_count)
{
    PrintValue("Min", results.min_, cpu_timer_freq);
    printf("\n");
    PrintValue("Max", results.max_, cpu_timer_freq);
    printf("\n");
    PrintValue("Avg", results.total_, cpu_timer_freq);
    printf("\n");
}

void Error(RepetitionTester &tester, char const *error_message)
{
    tester.mode_ = TestMode::ERRORR;
    fprintf(stderr, "ERROR: %s\n", error_message);
}

void NewTestWave(RepetitionTester &tester, uint64_t target_processed_byte_count, uint64_t cpu_timer_freq, uint32_t seconds_to_try)
{
    if (tester.mode_ == TestMode::UNINITIALIZED)
    {
        tester.mode_ = TestMode::TESTING;
        tester.target_processed_byte_count_ = target_processed_byte_count;
        tester.cpu_timer_freq_ = cpu_timer_freq;
        tester.print_new_minimums_ = true;
        tester.results_.min_.e[RepetitionValueType::CPUTIMER] = (uint64_t)-1; // sentinel max
    }
    else if (tester.mode_ == TestMode::COMPLETED)
    {
        tester.mode_ = TestMode::TESTING;

        if (tester.target_processed_byte_count_ != target_processed_byte_count)
        {
            Error(tester, "target_processed_byte_count_ changed");
        }

        if (tester.cpu_timer_freq_ != cpu_timer_freq)
        {
            Error(tester, "CPU frequency changed");
        }
    }

    tester.try_for_time_ = seconds_to_try * cpu_timer_freq;
    tester.tests_started_at_ = ReadCPUTimer();
}

void BeginTime(RepetitionTester &tester)
{
    ++tester.open_block_count_;
    auto &accum = tester.accumulated_on_this_test_;
    accum.e[RepetitionValueType::MEMPAGEFAULTS] -= ReadOSPageFaultCount();
    accum.e[RepetitionValueType::CPUTIMER] -= ReadCPUTimer();
}

void EndTime(RepetitionTester &tester)
{
    auto &accum = tester.accumulated_on_this_test_;
    accum.e[RepetitionValueType::MEMPAGEFAULTS] += ReadOSPageFaultCount();
    accum.e[RepetitionValueType::CPUTIMER] += ReadCPUTimer();

    ++tester.close_block_count_;
}

void CountBytes(RepetitionTester &tester, uint64_t byte_count)
{
    auto &accum = tester.accumulated_on_this_test_;
    accum.e[RepetitionValueType::BYTECOUNT] += byte_count;
}

void CountOps(RepetitionTester &tester, uint64_t op_count)
{
    auto &accum = tester.accumulated_on_this_test_;
    accum.e[RepetitionValueType::OPCOUNT] += op_count;
}

bool IsTesting(RepetitionTester &tester)
{
    if (tester.mode_ == TestMode::TESTING)
    {
        auto &accum = tester.accumulated_on_this_test_;
        uint64_t current_time = ReadCPUTimer();

        if (tester.open_block_count_) // tests with no timing blocks took some other path, don't count them
        {
            if (tester.open_block_count_ != tester.close_block_count_)
            {
                Error(tester, "Unbalanced BeginTime/EndTime");
            }

            if (accum.e[RepetitionValueType::BYTECOUNT] != tester.target_processed_byte_count_)
            {
                Error(tester, "Processed byte count mismatch");
            }

            if (tester.mode_ == TestMode::TESTING)
            {
                RepetitionTestResults &results = tester.results_;

                accum.e[RepetitionValueType::TESTCOUNT] = 1;

                for (uint32_t e_index = 0; e_index < ArrayCount(accum.e); ++e_index)
                {
                    results.total_.e[e_index] += accum.e[e_index];
                }

                if (results.max_.e[RepetitionValueType::CPUTIMER] < accum.e[RepetitionValueType::CPUTIMER])
                {
                    results.max_ = accum;
                }

                if (results.min_.e[RepetitionValueType::CPUTIMER] > accum.e[RepetitionValueType::CPUTIMER])
                {
                    results.min_ = accum;

                    // whenever we get a new minimum time, reset the clock to the full trial time
                    tester.tests_started_at_ = current_time;

                    if (tester.print_new_minimums_)
                    {
                        PrintValue("Min", results.min_, tester.cpu_timer_freq_);
                        printf("                                   \r");
                    }
                }

                tester.open_block_count_ = 0;
                tester.close_block_count_ = 0;
                tester.accumulated_on_this_test_ = {};
            }
        }

        if ((current_time - tester.tests_started_at_) > tester.try_for_time_)
        {
            tester.mode_ = TestMode::COMPLETED;

            printf("                                                          \r");
            PrintResults(tester.results_, tester.cpu_timer_freq_, tester.target_processed_byte_count_);
        }
    }

    bool result = (tester.mode_ == TestMode::TESTING);
    return result;
}
