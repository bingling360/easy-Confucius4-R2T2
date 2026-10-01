# -*- coding: utf-8 -*-
"""
语音/视频 → SRT 字幕工具(基于网易有道 Confucius4-R2T2 / Qwen3-ASR)

用法:
    python make_srt.py 视频或音频文件 [更多文件...] [-o 输出.srt] [--language Chinese]

流程:
    1. ffmpeg 抽取 16k 单声道音频
    2. 按静音切分成 ~30 秒的段落
    3. ASR 模型(fp16)逐批识别文本
    4. 释放 ASR 显存,加载 0.6B 强制对齐器,得到每个字/词的时间戳
    5. 按字幕节奏(字数/时长/标点)合并成字幕条,写出 SRT + 纯文本
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from collections import namedtuple

import numpy as np
import soundfile as sf
import torch

Item = namedtuple("Item", ["text", "start_time", "end_time"])

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models", "Confucius4-R2T2")
ALIGNER_DIR = os.path.join(BASE, "models", "Qwen3-ForcedAligner-0.6B")
SR = 16000

CJK_END_PUNCT = "。！？；，、,.!?;:…"
STRONG_PUNCT = "。！？!?;"
MID_PUNCT = "，、,:：,;"
PUNCT_SET = set(CJK_END_PUNCT + "：:…\"'“”‘’（）()[]【】《》<>-—=~·|")

# SRT 行尾要去掉的标点(中文字幕规范:行尾不放标点;句中标点保留)
END_PUNCT_STRIP = "，。、,.。;；:：!！?？…\"'“”‘’》」』)）]】"

# 网页界面下载文件要经过 URL,半角 # 会被浏览器当锚点截断链接、% 干扰百分号转义,
# 会导致下载失败;转成显示相近的全角字符
_URL_UNSAFE = str.maketrans({"#": "＃", "%": "％"})


def safe_stem(name: str) -> str:
    """文件名主干中影响 URL 下载的字符转全角(如抖音标题里的 #话题标签)"""
    return name.translate(_URL_UNSAFE)


def merge_punct_into_items(text, items):
    """把 ASR 原文里的标点按位置贴回对齐条目;对不上时返回 None"""
    text_body = [c for c in text if not c.isspace() and c not in PUNCT_SET]
    item_chars = [c for it in items for c in it.text if not c.isspace()]
    if len(text_body) != len(item_chars):
        return None
    if [c.casefold() for c in text_body] != [c.casefold() for c in item_chars]:
        return None
    # 每个正文位置属于哪条 item(英文单词一条 item 含多个字符)
    item_of_pos = []
    for i, it in enumerate(items):
        item_of_pos.extend([i] * sum(1 for c in it.text if not c.isspace()))
    out = [""] * len(items)
    pos = 0
    for c in text:
        if c.isspace():
            continue
        if c in PUNCT_SET:
            if pos > 0:
                out[item_of_pos[pos - 1]] += c
        else:
            out[item_of_pos[pos]] += c
            pos += 1
    return [Item(out[i], items[i].start_time, items[i].end_time) for i in range(len(items))]


# ---------------------------------------------------------------- audio I/O

def extract_audio_16k(src: str, dst_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
        "-acodec", "pcm_s16le", dst_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore")[-500:]
        raise RuntimeError(f"ffmpeg 处理失败: {err}")


def load_audio_16k(src: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "a.wav")
        extract_audio_16k(src, wav_path)
        y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        return y.mean(axis=1)


# ------------------------------------------------------- silence segmentation

def split_segments(y: np.ndarray, target_sec=30.0, max_sec=60.0, min_sec=6.0):
    """按静音位置切段,返回 [(start_sample, end_sample), ...]"""
    hop = int(0.02 * SR)
    n_frames = len(y) // hop
    if n_frames == 0:
        return [(0, len(y))]
    frames = y[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1) + 1e-12)

    speech_level = np.percentile(rms, 90)
    thr = max(0.01, 0.10 * speech_level)
    silent = rms < thr

    # 收集 >=0.3s 的静音段,切点取静音段中点
    cuts = []
    i = 0
    while i < n_frames:
        if silent[i]:
            j = i
            while j < n_frames and silent[j]:
                j += 1
            if (j - i) * hop >= 0.3 * SR:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1

    target_f = int(target_sec * SR / hop)
    max_f = int(max_sec * SR / hop)
    min_f = int(min_sec * SR / hop)

    segs = []
    start = 0
    for cf in cuts:
        if cf - start >= target_f:
            segs.append((start, cf))
            start = cf
    if n_frames - start > min_f:
        segs.append((start, n_frames))
    elif segs:
        segs[-1] = (segs[-1][0], n_frames)
    else:
        segs.append((0, n_frames))

    # 没有可用切点的超长段(如背景音乐连续)按 max_sec 硬切
    out = []
    for a, b in segs:
        while b - a > max_f:
            out.append((a, a + max_f))
            a += max_f
        out.append((a, b))
    return [(a * hop, min(b * hop, len(y))) for a, b in out if b * hop - a * hop > SR]


# ----------------------------------------------------------------- models

def load_asr_model(batch=4):
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel  # 导入即注册 qwen3_asr 架构

    free_b, total_b = torch.cuda.mem_get_info()
    need_b = int(4.6e9)
    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True,
                  max_inference_batch_size=batch, max_new_tokens=512)
    if free_b > need_b:
        print(f"  显存充足({free_b/2**30:.1f} GiB 空闲),模型整体载入 GPU")
        m = Qwen3ASRModel.from_pretrained(MODEL_DIR, **kwargs)
        m.model.to("cuda")
    else:
        gpu_gib = max(3.0, (free_b - 1.1 * 2**30) / 2**30)
        print(f"  空闲显存 {free_b/2**30:.1f} GiB 不足,自动分层:GPU {gpu_gib:.1f} GiB + 内存卸载")
        kwargs.update(device_map="auto", max_memory={0: f"{gpu_gib:.1f}GiB", "cpu": "12GiB"})
        m = Qwen3ASRModel.from_pretrained(MODEL_DIR, **kwargs)
    return m


def free_asr_model(m):
    del m.model
    del m
    gc.collect()
    torch.cuda.empty_cache()


def load_aligner():
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    a = Qwen3ForcedAligner.from_pretrained(ALIGNER_DIR, torch_dtype=torch.float16)
    a.model.to("cuda")
    return a


# ----------------------------------------------------------------- srt

def join_items(texts):
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " " + t
        else:
            out += t
    return out


def build_cues(items, max_chars=22, max_sec=5.0, keep_end_punct=False):
    """把逐字/词时间戳按字幕节奏合并成条;优先在标点处断句,超限回退到最近标点"""
    cues, cur, clen = [], [], 0
    last_punct = None  # (cur内下标, 该下标处的累计字数)

    def clen_upto(sub):
        return sum(len(x.text.replace(" ", "")) for x in sub)

    for it in items:
        cur.append(it)
        clen += len(it.text.replace(" ", ""))
        tail = it.text.strip()[-1:] if it.text.strip() else ""
        if tail in STRONG_PUNCT:
            last_punct = (len(cur) - 1, clen)
        elif tail in MID_PUNCT:
            last_punct = (len(cur) - 1, clen)

        dur = it.end_time - cur[0].start_time
        if clen >= max_chars or dur >= max_sec:
            if last_punct is not None and clen - last_punct[1] <= 14 and last_punct[1] >= 6:
                cut = last_punct[0] + 1
                cues.append(cur[:cut])
                cur = cur[cut:]
                last_punct = None
            else:
                cues.append(cur)
                cur = []
                last_punct = None
            clen = clen_upto(cur)
    if cur:
        cues.append(cur)

    evs = []
    for c in cues:
        t0 = max(0.0, c[0].start_time - 0.05)
        t1 = c[-1].end_time + 0.30
        txt = join_items([x.text for x in c])
        if not keep_end_punct:
            txt = txt.rstrip(END_PUNCT_STRIP)
        if txt:
            evs.append([t0, t1, txt])

    for k, ev in enumerate(evs):
        if k + 1 < len(evs):
            ev[1] = min(ev[1], evs[k + 1][0] - 0.05)
        if ev[1] - ev[0] < 0.9:
            ev[1] = ev[0] + 0.9
    return evs


def fmt_ts(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_srt(evs, path):
    lines = []
    for i, (t0, t1, txt) in enumerate(evs, 1):
        lines.append(f"{i}\n{fmt_ts(t0)} --> {fmt_ts(t1)}\n{txt}\n")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------- pipeline

def transcribe_to_srt(src: str, out_srt: str, language=None, seg_sec=30.0,
                      batch=4, max_chars=22, keep_end_punct=False, log=print):
    from qwen_asr.inference.utils import SAMPLE_RATE as _SR

    t_start = time.time()
    audio = load_audio_16k(src)
    dur = len(audio) / _SR
    log(f"音频时长 {dur/60:.1f} 分钟")

    segs = split_segments(audio, target_sec=seg_sec)
    log(f"切分出 {len(segs)} 段")

    log("加载 ASR 模型(fp16)...")
    asr = load_asr_model(batch=batch)

    GROUP = batch * 5
    results = []  # (text, language)
    try:
        for gi in range(0, len(segs), GROUP):
            group = segs[gi: gi + GROUP]
            wavs = [audio[a:b] for a, b in group]
            rs = asr.transcribe(audio=[(w, _SR) for w in wavs],
                                language=[language] * len(wavs) if language else [None] * len(wavs),
                                return_time_stamps=False)
            for r in rs:
                results.append((r.text.strip(), r.language))
            done_sec = group[-1][1] / _SR
            log(f"  识别进度 {done_sec/60:.1f}/{dur/60:.1f} 分钟")
    finally:
        free_asr_model(asr)

    log("加载强制对齐器(0.6B)...")
    aligner = load_aligner()
    items_all = []
    try:
        for (a, b), (txt, lang) in zip(segs, results):
            if not txt:
                continue
            offset = a / _SR
            r = aligner.align(audio=[(audio[a:b], _SR)],
                              text=[txt],
                              language=[lang or "Chinese"])[0]
            seg_items = [Item(it.text, it.start_time + offset, it.end_time + offset)
                         for it in r.items]
            # 用 ASR 原文把标点贴回来(失败则退回无标点)
            merged = merge_punct_into_items(txt, seg_items)
            items_all.extend(merged or seg_items)
    finally:
        del aligner
        gc.collect()
        torch.cuda.empty_cache()

    cues = build_cues(
        [Item(t, s, e) for t, s, e in items_all],
        max_chars=max_chars,
        keep_end_punct=keep_end_punct,
    )
    write_srt(cues, out_srt)

    txt_path = os.path.splitext(out_srt)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(join_items([t for t, _ in results]))

    log(f"完成:{out_srt}(共 {len(cues)} 条字幕,耗时 {time.time()-t_start:.0f} 秒)")
    return out_srt


def main():
    ap = argparse.ArgumentParser(description="语音/视频转 SRT 字幕")
    ap.add_argument("inputs", nargs="+", help="音频或视频文件(mp4/mp3/wav/m4a/mkv...)")
    ap.add_argument("-o", "--output", default=None, help="输出 SRT 路径(多文件时忽略)")
    ap.add_argument("--language", default=None,
                    help="强制语言,如 Chinese / English(默认自动检测)")
    ap.add_argument("--seg-sec", type=float, default=30.0, help="切分目标时长(秒)")
    ap.add_argument("--batch", type=int, default=4, help="识别批大小(显存不足可改 1/2)")
    ap.add_argument("--max-chars", type=int, default=22, help="每条字幕最大字数")
    ap.add_argument("--keep-end-punct", action="store_true",
                    help="保留每条字幕末尾的标点(默认按字幕规范去掉行尾标点)")
    args = ap.parse_args()

    for src in args.inputs:
        src = os.path.abspath(src)
        print(f"\n===== 处理:{os.path.basename(src)} =====")
        out = args.output or os.path.splitext(src)[0] + ".srt"
        try:
            transcribe_to_srt(src, out, language=args.language,
                              seg_sec=args.seg_sec, batch=args.batch,
                              max_chars=args.max_chars,
                              keep_end_punct=args.keep_end_punct)
        except Exception as e:
            print(f"[错误] {e}")
            if len(args.inputs) == 1:
                raise


if __name__ == "__main__":
    main()
