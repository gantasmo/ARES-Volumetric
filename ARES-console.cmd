@echo off
rem Visible-console fallback for ARES.vbs — same launcher, output on screen as well as in
rem tools\launch.log. Takes the same arguments: app (default), probe, bench, sam.
rem   ARES-console.cmd probe
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch.ps1" %*
echo.
echo Exit code: %ERRORLEVEL%  (log: %~dp0tools\launch.log)
pause
