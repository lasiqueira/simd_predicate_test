#pragma once
#include <cstdint>
#include <intrin.h>
#include <windows.h>
#include <psapi.h>

struct OsPlatform
{
    bool initialized_;
    HANDLE process_handle_;
    uint64_t cpu_timer_freq_;
};

#define PERF_TIME_TO_WAIT 100
extern OsPlatform g_platform;

uint64_t GetOSTimerFreq();
uint64_t ReadOSTimer();
uint64_t ReadCPUTimer();
uint64_t GetCPUFreqEstimate();
void InitializeOSMetrics();
uint64_t ReadOSPageFaultCount();
