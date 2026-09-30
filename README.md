# easy-Confucius4-R2T2

基于网易有道开源 [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 流式语音识别模型的本地字幕生成工具:一键把音频/视频转成带时间轴的 `.srt` 字幕和 `.txt` 纯文本,全部本地推理,数据不出本机。

## 使用方式

1. **拖拽**:把音视频文件拖到 `拖拽生成字幕.bat` 图标上,原地生成字幕。
2. **网页界面**:双击 `启动字幕工具.bat`,浏览器自动打开操作页面。
3. **命令行**:`python make_srt.py 视频.mp4 --language Chinese --max-chars 20`

详细用法、参数说明和常见问题见 [使用说明.md](使用说明.md)。

## 环境准备

- Python 3.12,安装依赖(含 CUDA 12.8 版 PyTorch,约需 4.6GB 显存):

  ```bat
  pip install -r requirements.txt
  ```

- **ffmpeg**(系统依赖):从视频抽取音轨时通过命令行调用,需安装并加入 PATH。
  Windows 可从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载 essentials 构建包,
  或直接 `winget install ffmpeg`
- 模型(必需,共约 5.6GB):下载后放到 `models/` 目录,目录结构为

  ```
  models/
  ├─ Confucius4-R2T2/           识别模型(3.9GB)
  └─ Qwen3-ForcedAligner-0.6B/  时间戳对齐模型(1.8GB)
  ```

  两个模型都在 Hugging Face 和 ModelScope 上发布,任选其一:

  | 模型 | Hugging Face | ModelScope |
  |---|---|---|
  | Confucius4-R2T2 | [netease-youdao/Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) | [netease-youdao/Confucius4-R2T2](https://modelscope.cn/models/netease-youdao/Confucius4-R2T2) |
  | Qwen3-ForcedAligner-0.6B | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | [Qwen/Qwen3-ForcedAligner-0.6B](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B) |

  用 modelscope 命令行下载(国内推荐):

  ```bat
  pip install modelscope
  modelscope download --model netease-youdao/Confucius4-R2T2 --local_dir models/Confucius4-R2T2
  modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir models/Qwen3-ForcedAligner-0.6B
  ```

  或用 huggingface 命令行(国内慢可先 `set HF_ENDPOINT=https://hf-mirror.com`):

  ```bat
  pip install -U "huggingface_hub[cli]"
  hf download netease-youdao/Confucius4-R2T2 --local-dir models/Confucius4-R2T2
  hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir models/Qwen3-ForcedAligner-0.6B
  ```

- 官方仓库源码可 clone 到 `Confucius4-R2T2/` 目录(参考用,运行不需要)

(以上大体积内容均已从 git 仓库排除,详见 `.gitignore`)
