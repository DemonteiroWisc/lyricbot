@echo off
REM This batch file runs the LyricBot Python script.

REM Change to the directory where the script is located.
cd "C:\Users\danie\Desktop\LyricBot"

REM Run the Python script.
REM Make sure 'python' is in your system's PATH, or provide the full path to python.exe
python lyricbot_vivid.py
set PIPELINE_EXIT=%ERRORLEVEL%

:: Exit 2 = first Sunday run (recap printed) — keep window open to review
if %PIPELINE_EXIT% EQU 2 (
    echo.
    echo Press any key to close...
    pause >nul
    goto :eof
)

:: Exit 1 = failure — keep window open so the error stays visible
if %PIPELINE_EXIT% EQU 1 (
    echo.
    pause
)