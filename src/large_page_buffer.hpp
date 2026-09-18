#pragma once
#include <windows.h>
#include <cstdint>
#pragma comment(lib, "advapi32.lib")

// Requires the "Lock pages in memory" user right to already be assigned to the account
// (Local Security Policy / secpol.msc); this only activates the privilege in the token,
// it cannot grant the right itself.
inline bool EnableLockMemoryPrivilege()
{
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
        return false;

    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    bool ok = LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)
        && AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr)
        && GetLastError() == ERROR_SUCCESS;

    CloseHandle(token);
    return ok;
}

// Commits the whole buffer up front so no page faults happen on first touch inside a timed loop.
// Uses large pages (fully resident, no per-page demand-zero faults) when the process holds
// SeLockMemoryPrivilege, otherwise falls back to a normal VirtualAlloc commit.
struct LargePageBuffer
{
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool usingLargePages_ = false;

    explicit LargePageBuffer(size_t requestedSize)
    {
        size_t largePageSize = GetLargePageMinimum();
        if (largePageSize != 0 && EnableLockMemoryPrivilege())
        {
            size_t rounded = (requestedSize + largePageSize - 1) & ~(largePageSize - 1);
            data_ = (uint8_t*)VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (data_)
            {
                size_ = rounded;
                usingLargePages_ = true;
            }
        }

        if (!data_)
        {
            data_ = (uint8_t*)VirtualAlloc(nullptr, requestedSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            size_ = requestedSize;
        }
    }

    ~LargePageBuffer()
    {
        if (data_)
            VirtualFree(data_, 0, MEM_RELEASE);
    }

    LargePageBuffer(const LargePageBuffer&) = delete;
    LargePageBuffer& operator=(const LargePageBuffer&) = delete;

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool usingLargePages() const { return usingLargePages_; }
};
