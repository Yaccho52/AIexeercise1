# -*- coding: utf-8 -*-
"""
run_app.py —— 统一启动入口（Streamlit 界面版）
==========================================================================
双击「启动应用.bat」走它，打包成 exe 之后内部也走它。

【打包时的重要提醒】
  这个文件必须作为 PyInstaller 的"入口脚本"，也就是命令最后那一个参数。
  不要把 app.py 当入口：那样 exe 会把 app.py 当普通 Python 脚本执行，
  只会打印 "missing ScriptRunContext" 警告，网页根本起不来。
      正确：  ... --add-data "app.py;." run_app.py
      错误：  ... run_app.py --add-data "app.py;." app.py
==========================================================================
"""

import os
import sys
from pathlib import Path


def _resource(name):
    """打包后资源在 sys._MEIPASS 里；未打包时就在本文件旁边。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def _skip_welcome_prompt():
    """跳过 Streamlit 首次运行的邮箱询问（写一次，失败也无妨）。"""
    try:
        cred = Path.home() / ".streamlit" / "credentials.toml"
        if not cred.exists():
            cred.parent.mkdir(parents=True, exist_ok=True)
            cred.write_text('[general]\nemail = ""\n', encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    _skip_welcome_prompt()

    script = _resource("app.py")

    # 关键配置同时用环境变量钉死：即使命令行参数没被解析到，也能生效
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    print(f"[run_app] 正在启动 Streamlit 服务，界面脚本：{script}", flush=True)

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", script,
        "--global.developmentMode=false",
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
