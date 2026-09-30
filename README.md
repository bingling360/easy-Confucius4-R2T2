# easy-Confucius4-R2T2

基于网易有道开源 [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 流式语音识别模型的本地字幕生成工具:一键把音频/视频转成带时间轴的 `.srt` 字幕和 `.txt` 纯文本,全部本地推理,数据不出本机。

## 使用方式

1. **拖拽**:把音视频文件拖到 `拖拽生成字幕.bat` 图标上,原地生成字幕。
2. **网页界面**:双击 `启动字幕工具.bat`,浏览器自动打开操作页面。
3. **命令行**:`python make_srt.py 视频.mp4 --language Chinese --max-chars 20`

详细用法、参数说明和常见问题见 [使用说明.md](使用说明.md)。

## 环境准备

- Python 3.12 + CUDA 版 PyTorch(约需 4.6GB 显存)
- 模型:Confucius4-R2T2 与 Qwen3-ForcedAligner-0.6B,下载后放到 `models/` 目录
- 官方仓库源码可 clone 到 `Confucius4-R2T2/` 目录(参考用,运行不需要)

(以上大体积内容均已从 git 仓库排除,详见 `.gitignore`)
