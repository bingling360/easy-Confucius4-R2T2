@echo off
cd /d %~dp0
echo 正在启动字幕工具网页界面,浏览器将自动打开 http://127.0.0.1:7860
echo 使用过程中请勿关闭本窗口;用完后直接关闭窗口即可退出。
r2t2-env\Scripts\python.exe r2t2_gui.py
pause
