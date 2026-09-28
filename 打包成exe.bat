@echo off
chcp 936 >nul 2>&1
cd /d "%~dp0"
title 打包成 EXE

set "PY="
py --version >nul 2>&1
if not errorlevel 1 set "PY=py"
if defined PY goto :ok
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :ok
echo.
echo   [错误] 没有找到 Python，请先安装（安装时勾选 Add python.exe to PATH）。
echo.
pause
exit /b 1

:ok
echo ============================================================
echo   第 1 步 / 共 2 步：安装打包与运行所需的库（首次较慢）
echo ============================================================
%PY% -m pip install --upgrade pyinstaller streamlit plotly pandas openpyxl
if errorlevel 1 (
    echo.
    echo   [错误] 依赖安装失败，请检查网络后重新双击本文件。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   第 2 步 / 共 2 步：开始打包
echo   体积约 300-500 MB，可能需要 5-15 分钟，期间请勿关闭窗口
echo ============================================================
%PY% -m PyInstaller --noconfirm --clean --onefile ^
  --name "时间序列预测对比" ^
  --collect-all streamlit ^
  --collect-all plotly ^
  --copy-metadata streamlit ^
  --add-data "app.py;." ^
  --add-data "ts_core.py;." ^
  run_app.py

echo.
if exist "dist\时间序列预测对比.exe" (
    echo   打包完成！exe 就在 dist 文件夹里，可以拷到桌面双击运行。
    echo   提示：首次启动要解压内置文件，可能需要等 10-30 秒。
) else (
    echo   打包失败，请把上面的报错信息复制给 Claude。
)
echo.
pause
