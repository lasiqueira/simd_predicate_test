#include "platform_metrics.hpp"
#pragma comment(lib, "psapi.lib")

OsPlatform g_platform;

uint64_t GetOSTimerFreq()
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return freq.QuadPart;
}

uint64_t ReadOSTimer()
{
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return value.QuadPart;
}

uint64_t ReadOSPageFaultCount()
{
    PROCESS_MEMORY_COUNTERS_EX memory_counters = {};
    memory_counters.cb = sizeof(memory_counters);
    GetProcessMemoryInfo(g_platform.process_handle_, (PROCESS_MEMORY_COUNTERS *)&memory_counters, sizeof(memory_counters));

    uint64_t result = memory_counters.PageFaultCount;
    return result;
}

void InitializeOSMetrics()
{
    if (!g_platform.initialized_)
    {
        g_platform.initialized_ = true;
        g_platform.process_handle_ = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, GetCurrentProcessId());
        g_platform.cpu_timer_freq_ = GetOSTimerFreq();
    }
}

uint64_t ReadCPUTimer()
{
    return __rdtsc();
}

uint64_t GetCPUFreqEstimate(void)
{
    uint64_t milliseconds_to_wait = PERF_TIME_TO_WAIT;

    uint64_t os_freq = GetOSTimerFreq();

    uint64_t cpu_start = ReadCPUTimer();
    uint64_t os_start = ReadOSTimer();
    uint64_t os_end = 0;
    uint64_t os_elapsed = 0;
    uint64_t os_wait_time = os_freq * milliseconds_to_wait / 1000;
    while (os_elapsed < os_wait_time)
    {
        os_end = ReadOSTimer();
        os_elapsed = os_end - os_start;
    }

    uint64_t cpu_end = ReadCPUTimer();
    uint64_t cpu_elapsed = cpu_end - cpu_start;
    uint64_t cpu_freq = 0;
    if (os_elapsed)
    {
        cpu_freq = os_freq * cpu_elapsed / os_elapsed;
    }

    return cpu_freq;
}
