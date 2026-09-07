@echo off
rem Visible-console fallback for ARES.vbs — same launcher, output on screen as well as in
rem tools\launch.log. Takes the same arguments: app (default), probe, bench, sam.
rem   tools\launch-console.cmd probe
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
echo.
echo Exit code: %ERRORLEVEL%  (log: %~dp0launch.log)
pause
