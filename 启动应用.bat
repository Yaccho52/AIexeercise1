@echo off
chcp 936 >nul 2>&1
cd /d "%~dp0"
title 时间序列预测对比工具

set "PY="
py --version >nul 2>&1
if not errorlevel 1 set "PY=py"
if defined PY goto :gotpy
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :gotpy

echo.
echo   [错误] 没有找到 Python。
echo   请到 https://www.python.org/downloads/ 下载安装，
echo   安装时务必勾选 "Add python.exe to PATH"，装完重新双击本文件。
echo.
pause
exit /b 1

:gotpy
REM ---- 情况一：本机 Python 已经装好全部依赖，直接用 ----
%PY% -c "import streamlit, plotly, pandas, openpyxl" >nul 2>&1
if not errorlevel 1 goto :run_system

REM ---- 情况二：本文件夹里的 .venv 已经就绪，用虚拟环境 ----
if exist ".venv\Scripts\python.exe" goto :run_venv

REM ---- 情况三：首次运行，创建虚拟环境并安装依赖 ----
echo.
echo   首次运行，正在准备运行环境（需要联网，约 1-3 分钟）...
echo.
%PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto :venv_fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
goto :run_venv

:run_system
echo.
echo   正在启动，浏览器会自动打开 http://localhost:8501
echo   用完后关闭本窗口即可停止程序。
echo.
%PY% run_app.py
goto :done

:run_venv
echo.
echo   正在启动，浏览器会自动打开 http://localhost:8501
echo   用完后关闭本窗口即可停止程序。
echo.
".venv\Scripts\python.exe" run_app.py
goto :done

:venv_fail
echo.
echo   [错误] 创建虚拟环境失败。请确认 Python 安装完整后重试。
echo.
pause
exit /b 1

:pip_fail
echo.
echo   [错误] 依赖安装失败，请检查网络后重新双击本文件。
echo.
pause
exit /b 1

:done
echo.
echo   程序已停止。按任意键关闭窗口。
pause >nul
