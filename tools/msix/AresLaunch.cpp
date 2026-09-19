// The MSIX Application entry point.
//
// A package needs one Executable, and the Extensions (the COM server and the context-menu verb)
// hang off that Application element. This is that executable: it starts ARES the same way the
// desktop launcher does, so the package entry is a working app rather than a dead placeholder.
//
//   Ares.exe                 -> ARES.vbs app
//   Ares.exe --open <path>   -> ARES.vbs app --open <path>
//
// Windowless via wscript.exe, matching every other entry point into the launcher.

#include <windows.h>
#include <shellapi.h>
#include <strsafe.h>

#include "AresRoot.h"   // generated: the checkout this package was built from

#pragma comment(lib, "shell32.lib")

int APIENTRY wWinMain(HINSTANCE, HINSTANCE, LPWSTR cmdLine, int)
{
    wchar_t args[4096];
    HRESULT hr = (cmdLine && *cmdLine)
        ? StringCchPrintfW(args, ARRAYSIZE(args), L"\"%s\\ARES.vbs\" app %s", ARES_ROOT, cmdLine)
        : StringCchPrintfW(args, ARRAYSIZE(args), L"\"%s\\ARES.vbs\" app", ARES_ROOT);
    if (FAILED(hr)) return 1;

    SHELLEXECUTEINFOW ei = { sizeof(ei) };
    ei.fMask = SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI;
    ei.lpVerb = L"open";
    ei.lpFile = L"wscript.exe";
    ei.lpParameters = args;
    ei.nShow = SW_HIDE;
    return ShellExecuteExW(&ei) ? 0 : 1;
}
