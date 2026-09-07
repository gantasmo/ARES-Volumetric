' ARES — double-click me. No terminal window appears.
'
' The one launcher: installs Node.js if missing, installs dependencies and builds when stale,
' generates a demo clip if the checkout has none, starts the COOP/COEP dev server (or reuses a
' running one), and opens the app in your default browser.
'
' Optional argument picks what to open — app (default), probe, bench, or sam:
'     ARES.vbs probe        cscript "ARES.vbs" bench
' Troubleshooting: run tools\launch-console.cmd for a visible console, or read tools\launch.log.
Option Explicit
Dim fso, sh, dir, ps1, cmd, i
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\tools\launch.ps1"
If Not fso.FileExists(ps1) Then
    MsgBox "Cannot find " & ps1, vbCritical, "ARES"
    WScript.Quit 1
End If
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """"
For i = 0 To WScript.Arguments.Count - 1
    cmd = cmd & " """ & WScript.Arguments(i) & """"
Next
sh.Run cmd, 0, False
