' ARES SAM service 1-click launcher - double-click me. No terminal window will appear.
' Starts the local SAM ViT-H segmentation API (http://127.0.0.1:7263) using the WanGP
' python env; the dev server proxies it at /sam/*. First start loads weights (~10-20 s).
' Troubleshooting: read tools\sam-service\sam-service.log.
Option Explicit
Dim fso, sh, dir, ps1
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\tools\sam-service\run-sam-service.ps1"
If Not fso.FileExists(ps1) Then
    MsgBox "Cannot find " & ps1, vbCritical, "ARES SAM Service"
    WScript.Quit 1
End If
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, False
