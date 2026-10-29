"""Telegram 实验遥控 Bot：每次运行拉取新消息、确认、执行命令、回传结果。
环境变量：TG_TOKEN（必需），TG_ALLOWED_CHAT（可选，逗号分隔的 chat id 白名单），WORK（工作目录）。"""
import json, os, subprocess, sys, time, glob, shlex, urllib.request, urllib.parse
TOKEN = os.environ["TG_TOKEN"]; API = f"https://api.telegram.org/bot{TOKEN}/"
ALLOWED = {c.strip() for c in os.environ.get("TG_ALLOWED_CHAT", "").split(",") if c.strip()}
WORK = os.environ.get("WORK", os.getcwd()); HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["solar_system", "trappist1", "kepler11", "kepler33", "hd219134"]
HELP = ("可用命令：\n/stellar [星系] — 行星空隙预测+轨道图（solar_system, trappist1, kepler11, kepler33, hd219134）\n"
        "/ael [话题] [轮数] [每模式次数] — 三模式对照实验，例：/ael 记忆的本质 3 1\n"
        "/test — 运行两个项目的测试\n/id — 显示你的 chat id\n/help — 帮助\n"
        "提示：Bot 每 5–15 分钟检查一次消息，AEL 实验约需 10–30 分钟。")

def call(method, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(API + method, data=data, timeout=60) as r: return json.load(r)

def send(chat, text):
    for i in range(0, max(len(text), 1), 4000):
        call("sendMessage", chat_id=chat, text=text[i:i + 4000] or "(空)")

def send_file(chat, path, kind="document", caption=""):
    boundary = "----labbot%d" % time.time_ns(); field = "photo" if kind == "photo" else "document"
    with open(path, "rb") as f: content = f.read()
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat}\r\n",
             f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption[:1000]}\r\n"]
    body = "".join(parts).encode() + (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{os.path.basename(path)}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API + ("sendPhoto" if kind == "photo" else "sendDocument"), data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=120).read()

def sh(cmd, cwd=None, timeout=5400):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)

def tail(s, n=1): return "\n".join(s.strip().splitlines()[-n:])

import re
PLANETS = {"Mercury": "水星", "Venus": "金星", "Earth": "地球", "Mars": "火星", "Jupiter": "木星",
           "Saturn": "土星", "Uranus": "天王星", "Neptune": "海王星"}
ROW = re.compile(r"^\s*(\d+)\s+(.+?)\s{2,}([\d.e+]+) AU\s+([\d.e+]+) (yr|d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
def pname(n): return PLANETS.get(n.strip(), n.strip())

def stellar_zh(out):
    lines = []
    m = re.search(r"a_n = ([\d.]+) x ([\d.]+)\^n", out); r2 = re.search(r"R\^2: ([\d.]+)", out)
    if m: lines.append(f"Titius–Bode 拟合：a = {m.group(1)} × {m.group(2)}ⁿ" + (f"，R² = {r2.group(1)}" if r2 else ""))
    f = re.search(r"\((\d+) gap\(s\) filtered", out)
    if f: lines.append(f"已过滤 {f.group(1)} 个不可靠空隙")
    if re.search(r"resonant chain", out): lines.append("检测到紧密共振链：插入隐藏行星会破坏共振，因此空隙分数已降权")
    rows = [ROW.match(l) for l in out.splitlines()]; rows = [r for r in rows if r]
    if rows:
        lines.append("\n预测空隙（按综合分排序）：")
        for r in rows:
            names = r.group(2); e = re.search(r"\(outer edge \+(\d+)\)", names)
            where = f"{pname(names[:e.start()])} 外侧第 {e.group(1)} 个外推位置" if e else "–".join(pname(x) for x in re.split(r"\s{2,}", names)) + " 之间"
            unit = "年" if r.group(5) == "yr" else "天"
            lines.append(f"{r.group(1)}. {where}：{r.group(3)} AU，周期 {r.group(4)} {unit}｜TB {r.group(6)} · 稳定性 {r.group(7)} · 综合 {r.group(8)}")
    if rows: lines.append(f"\n最可能位置：#1，综合分 {rows[0].group(8)}（仅代表“值得找”，不代表一定有行星）")
    if not lines: lines.append(out[-3000:])
    return "\n".join(lines)

def pytest_zh(s):
    s = tail(s)
    for a, b in [("passed", "通过"), ("failed", "失败"), ("skipped", "跳过"), ("warnings", "警告"), ("warning", "警告"), ("errors", "错误"), ("error", "错误"), (", ", "，"), (" in ", "，耗时 ")]:
        s = s.replace(a, b)
    return s

METRICS = {"Metric": "指标", "single_agent": "单模型", "self_dialogue": "自对话", "dual_basic": "双Agent",
           "overall_score": "综合分", "utterances": "发言数", "signals_per_utterance": "每句信号", "raw_signals_per_utterance": "每句原始信号",
           "template_echo_ratio": "模板回声率", "signals_per_100_tokens": "每百token信号", "novel_concepts": "新概念",
           "unique_concepts": "独特概念", "concept_uptake": "概念采纳", "cross_domain_analogies": "跨域类比",
           "critical_challenges": "批判质疑", "reasoning_steps": "推理步骤", "metacognitive_count": "元认知",
           "repetition_mean": "重复度", "self_repetition_mean": "自我重复", "repetition_combined_mean": "综合重复度",
           "topic_drift_mean": "话题漂移", "judge_score": "评审分"}
def ael_zh(md):
    keep = ["overall_score", "repetition_combined_mean", "utterances", "signals_per_utterance", "template_echo_ratio",
            "novel_concepts", "concept_uptake", "critical_challenges", "topic_drift_mean"]
    head, rows = None, []
    for l in md.splitlines():
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if cells and cells[0] == "Metric": head = cells
        elif head and cells and cells[0] in keep and len(cells) == len(head): rows.append(cells)
    if not head: return md[:3500]
    modes = [METRICS.get(h, h) for h in head[1:]]
    out = ["模式：" + " / ".join(modes)]
    for c in rows: out.append(f"{METRICS[c[0]]}：" + " / ".join(v.replace(' ± 0', '') for v in c[1:]))
    out.append("\n说明：综合分越高越好，重复度越低越好；每模式只跑 1 次时不做显著性检验，结果仅供参考。")
    return "\n".join(out)

def do_stellar(chat, args):
    system = args[0] if args else "solar_system"
    if system not in SYSTEMS: return send(chat, f"未知星系 {system}。可选：{', '.join(SYSTEMS)}")
    png = f"{WORK}/stellar_{system}.png"
    code, out = sh(f"python -m stellar_predictor.cli analyze --system {system} --plot {png}", cwd=f"{WORK}/STELLAR-PREDICTOR")
    send(chat, f"🪐 STELLAR · {system}\n\n" + stellar_zh(out))
    if os.path.exists(png): send_file(chat, png, "photo", f"{system} 轨道图 + Titius–Bode 拟合")

def do_test(chat, args):
    _, a = sh("python -m pytest -q", cwd=f"{WORK}/Agent-Emergence-Lab")
    _, s = sh("python -m pytest -q", cwd=f"{WORK}/STELLAR-PREDICTOR")
    send(chat, f"🧪 测试结果\nAEL：{pytest_zh(a)}\nSTELLAR：{pytest_zh(s)}")

def do_ael(chat, args):
    nums = [a for a in args if a.isdigit()]; words = [a for a in args if not a.isdigit()]
    topic = " ".join(words) or "记忆的本质"; turns = int(nums[0]) if nums else 3; runs = int(nums[1]) if len(nums) > 1 else 1
    turns, runs = min(turns, 10), min(runs, 5)
    send(chat, f"🧠 开始 AEL 实验：话题「{topic}」，{turns} 轮，每模式 {runs} 次。完成后发结果。")
    d = f"{WORK}/Agent-Emergence-Lab"; sh("rm -rf logs/experiments logs/conversations logs/memory", cwd=d)
    code, out = sh(f"python -m src.experiment --config configs/low_resource_4g.yaml --topic {shlex.quote(topic)} "
                   f"--turns {turns} --runs {runs} --seed 42", cwd=d)
    rep = f"{d}/logs/experiments/latest.md"
    if not os.path.exists(rep): return send(chat, "❌ 实验失败，日志末尾：\n" + out[-3000:])
    md = open(rep, encoding="utf-8").read()
    send(chat, f"✅ AEL 结果（话题「{topic}」）\n" + ael_zh(md))
    send_file(chat, rep, caption="完整报告")
    ts = sorted(glob.glob(f"{d}/logs/conversations/*.md"))
    if ts:
        bundle = f"{WORK}/ael_transcripts.md"
        with open(bundle, "w", encoding="utf-8") as f:
            for t in ts: f.write(open(t, encoding="utf-8").read() + "\n\n---\n\n")
        send_file(chat, bundle, caption="三种模式的对话记录")

CMDS = {"/stellar": do_stellar, "/ael": do_ael, "/test": do_test}

def main():
    ups = call("getUpdates", timeout=0).get("result", [])
    if not ups: print("no updates"); return
    call("getUpdates", offset=ups[-1]["update_id"] + 1, timeout=0)  # 确认，避免重复执行
    for u in ups:
        m = u.get("message") or {}; text = (m.get("text") or "").strip(); chat = str(m.get("chat", {}).get("id", ""))
        if not text or not chat: continue
        parts = text.split(); cmd = parts[0].split("@")[0].lower(); args = parts[1:]
        print("cmd", chat, cmd, args)
        if cmd in ("/start", "/id"):
            send(chat, f"你的 chat id：{chat}\n" + ("" if chat in ALLOWED else "（尚未加入白名单，暂不能执行实验）\n") + "\n" + HELP); continue
        if chat not in ALLOWED:
            send(chat, "⛔ 未授权。"); continue
        if cmd == "/help": send(chat, HELP); continue
        fn = CMDS.get(cmd)
        if not fn: send(chat, "看不懂这个命令。\n\n" + HELP); continue
        try: fn(chat, args)
        except Exception as e: send(chat, f"❌ 出错：{e}")

if __name__ == "__main__": main()
