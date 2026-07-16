' ARES Phase 0 bench - double-click me. No terminal window will appear.
' Builds (if needed), runs the intra-representation benchmark (~1-2 minutes,
' silent), then opens the report page in your default browser.
' Troubleshooting: read tools\bench.log.
Option Explicit
Dim fso, sh, dir, ps1
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\tools\bench.ps1"
If Not fso.FileExists(ps1) Then
    MsgBox "Cannot find " & ps1, vbCritical, "ARES Bench"
    WScript.Quit 1
End If
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, False
