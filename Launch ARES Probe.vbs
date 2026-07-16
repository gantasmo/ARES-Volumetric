' ARES 1-click launcher - double-click me. No terminal window will appear.
' Checks/installs dependencies, builds, starts the COOP/COEP dev server,
' and opens the Phase 0 capability probe in your default browser.
' Troubleshooting: run tools\launch-debug.cmd instead (visible console),
' or read tools\launch.log.
Option Explicit
Dim fso, sh, dir, ps1
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\tools\launch.ps1"
If Not fso.FileExists(ps1) Then
    MsgBox "Cannot find " & ps1, vbCritical, "ARES Launcher"
    WScript.Quit 1
End If
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, False
