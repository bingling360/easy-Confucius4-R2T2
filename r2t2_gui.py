# -*- coding: utf-8 -*-
"""
字幕工具网页界面(本地运行)
启动: python r2t2_gui.py  或双击 启动字幕工具.bat
浏览器会自动打开 http://127.0.0.1:7860
"""

import os
import queue
import threading

import gradio as gr

from make_srt import transcribe_to_srt, MODEL_DIR, ALIGNER_DIR

LANGS = ["自动检测", "Chinese", "English", "Cantonese", "Japanese", "Korean",
         "French", "German", "Spanish", "Russian", "Portuguese", "Italian"]


def worker(q, files, language, seg_sec, batch, max_chars):
    try:
        for i, f in enumerate(files, 1):
            q.put(f"===== 文件 {i}/{len(files)}:{os.path.basename(f)} =====")
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(os.path.dirname(os.path.abspath(f)), base + ".srt")
            transcribe_to_srt(
                f, out,
                language=None if language == "自动检测" else language,
                seg_sec=seg_sec, batch=batch, max_chars=max_chars,
                log=lambda m: q.put(m),
            )
            q.put(("FILE", out))
        q.put(("DONE", None))
    except Exception as e:
        q.put(("DONE", f"[错误] {e}"))


def run(files, language, seg_sec, batch, max_chars):
    if not files:
        yield "请先上传音频或视频文件", None
        return
    q = queue.Queue()
    t = threading.Thread(
        target=worker,
        args=(q, files, language, seg_sec, int(batch), int(max_chars)),
        daemon=True,
    )
    t.start()
    logs, downloads = [], []
    while True:
        item = q.get()
        if isinstance(item, tuple):
            if item[0] == "FILE":
                downloads.append(item[1])
                yield "\n".join(logs), downloads
            else:  # DONE
                if item[1]:
                    logs.append(item[1])
                yield "\n".join(logs), downloads
                return
        else:
            logs.append(item)
            yield "\n".join(logs), downloads


with gr.Blocks(title="R2T2 字幕工具") as demo:
    gr.Markdown(
        "## 语音 / 视频 → SRT 字幕\n"
        "基于网易有道 **Confucius4-R2T2** 流式语音识别模型(本地 GPU 推理,数据不出电脑)。"
        "上传文件后点开始,生成的 `.srt` 和 `.txt` 与源文件放在同一目录。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            files_in = gr.File(
                label="上传音频/视频(可多选:mp4 mkv mov mp3 wav m4a flac...)",
                file_count="multiple", type="filepath",
            )
            lang_in = gr.Dropdown(LANGS, value="自动检测", label="语言")
            seg_in = gr.Slider(10, 60, value=30, step=5, label="切分目标时长(秒)")
            batch_in = gr.Slider(1, 8, value=4, step=1, label="识别批大小(显存不足请调小)")
            chars_in = gr.Slider(10, 40, value=22, step=1, label="每条字幕最大字数")
            btn = gr.Button("开始生成字幕", variant="primary")
        with gr.Column(scale=2):
            log_out = gr.Textbox(label="处理日志", lines=20, interactive=False)
            file_out = gr.File(label="生成的字幕文件", file_count="multiple")

    btn.click(
        run,
        inputs=[files_in, lang_in, seg_in, batch_in, chars_in],
        outputs=[log_out, file_out],
    )


if __name__ == "__main__":
    assert os.path.isdir(MODEL_DIR) and os.path.isdir(ALIGNER_DIR), "模型目录缺失"
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
