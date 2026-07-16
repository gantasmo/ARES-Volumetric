@echo off
rem Visible-console fallback for "Launch ARES Probe.vbs" - same launch flow,
rem but you can watch the output. Log also written to launch.log.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
echo.
echo Exit code: %ERRORLEVEL%  (log: %~dp0launch.log)
pause
