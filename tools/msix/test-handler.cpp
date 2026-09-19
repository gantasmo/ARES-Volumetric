// Unit test for the Explorer command handler, with no packaging and no registration involved.
//
// LoadLibrary + DllGetClassObject reaches the class directly, so this exercises exactly the code
// Explorer will run — GetTitle, GetState, GetCanonicalName and the folder/file distinction —
// without needing the MSIX installed or the certificate trusted. Invoke is deliberately NOT
// called: it launches the app.
//
//   cl /nologo /std:c++17 /EHsc /DUNICODE /D_UNICODE test-handler.cpp /link /OUT:test-handler.exe
//   test-handler.exe <a folder> <a file>

#include <windows.h>
#include <shobjidl_core.h>
#include <shlwapi.h>
#include <stdio.h>

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")   // SHCreateItemFromParsingName, SHCreateShellItemArrayFromShellItem

static const CLSID CLSID_AresConvertCommand =
    { 0x7107efe2, 0x8605, 0x4f5d, { 0x87, 0x57, 0x74, 0x72, 0x17, 0x1d, 0x33, 0x27 } };

typedef HRESULT (STDAPICALLTYPE *PFN_DllGetClassObject)(REFCLSID, REFIID, void**);

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static IShellItemArray* ArrayFor(PCWSTR path)
{
    IShellItem* item = nullptr;
    if (FAILED(SHCreateItemFromParsingName(path, nullptr, IID_PPV_ARGS(&item)))) return nullptr;
    IShellItemArray* arr = nullptr;
    SHCreateShellItemArrayFromShellItem(item, IID_PPV_ARGS(&arr));
    item->Release();
    return arr;
}

static void probe(IClassFactory* factory, PCWSTR path, const char* label, PCWSTR expectTitle)
{
    IExplorerCommand* cmd = nullptr;
    if (FAILED(factory->CreateInstance(nullptr, IID_PPV_ARGS(&cmd)))) { check(false, "CreateInstance"); return; }

    IShellItemArray* arr = ArrayFor(path);
    if (!arr) { printf("SKIP  %s (path not found)\n", label); cmd->Release(); return; }

    PWSTR title = nullptr;
    HRESULT hr = cmd->GetTitle(arr, &title);
    wprintf(L"      %hs -> title \"%s\"\n", label, SUCCEEDED(hr) && title ? title : L"(none)");
    check(SUCCEEDED(hr) && title && wcscmp(title, expectTitle) == 0, label);
    if (title) CoTaskMemFree(title);

    EXPCMDSTATE state = ECS_HIDDEN;
    hr = cmd->GetState(arr, FALSE, &state);
    check(SUCCEEDED(hr) && state == ECS_ENABLED, "GetState == ECS_ENABLED");

    GUID canon = {};
    hr = cmd->GetCanonicalName(&canon);
    check(SUCCEEDED(hr) && IsEqualGUID(canon, CLSID_AresConvertCommand), "GetCanonicalName == CLSID");

    arr->Release();
    cmd->Release();
}

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3) { printf("usage: test-handler <folder> <file>\n"); return 2; }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    HMODULE dll = LoadLibraryW(L"AresShell.dll");
    check(dll != nullptr, "LoadLibrary AresShell.dll");
    if (!dll) return 1;

    auto getClassObject = (PFN_DllGetClassObject)GetProcAddress(dll, "DllGetClassObject");
    check(getClassObject != nullptr, "GetProcAddress DllGetClassObject");
    if (!getClassObject) return 1;

    IClassFactory* factory = nullptr;
    HRESULT hr = getClassObject(CLSID_AresConvertCommand, IID_PPV_ARGS(&factory));
    check(SUCCEEDED(hr) && factory, "DllGetClassObject -> IClassFactory");
    if (!factory) return 1;

    // An unknown CLSID must be refused, not silently served.
    CLSID other = CLSID_AresConvertCommand; other.Data1 ^= 1;
    void* dummy = nullptr;
    check(getClassObject(other, IID_IClassFactory, &dummy) == CLASS_E_CLASSNOTAVAILABLE,
          "unknown CLSID -> CLASS_E_CLASSNOTAVAILABLE");

    probe(factory, argv[1], "folder", L"Convert folder to .ares");
    probe(factory, argv[2], "file",   L"Convert to .ares");

    factory->Release();
    CoUninitialize();
    printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
