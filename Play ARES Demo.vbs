' ARES P1 volumetric demo - double-click me. No terminal window will appear.
' Builds (if needed), generates the demo clip, starts the COOP/COEP dev server,
' and opens the WebGPU volumetric player in your default browser.
' Troubleshooting: read tools\demo.log.
Option Explicit
Dim fso, sh, dir, ps1
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\tools\demo.ps1"
If Not fso.FileExists(ps1) Then
    MsgBox "Cannot find " & ps1, vbCritical, "ARES Demo"
    WScript.Quit 1
End If
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, False
