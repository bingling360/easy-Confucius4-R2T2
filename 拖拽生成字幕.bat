@echo off
cd /d %~dp0
if "%~1"=="" (
    echo 把音频或视频文件拖到本文件图标上,即可在原位置生成同名 .srt 字幕文件。
    echo 支持多个文件一起拖入,支持 mp4 mkv mov mp3 wav m4a flac 等格式。
    pause
    exit /b
)
r2t2-env\Scripts\python.exe make_srt.py %*
echo.
echo 全部完成,窗口可以关闭。
pause
