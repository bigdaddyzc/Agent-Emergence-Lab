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

def do_stellar(chat, args):
    system = args[0] if args else "solar_system"
    if system not in SYSTEMS: return send(chat, f"未知星系 {system}。可选：{', '.join(SYSTEMS)}")
    png = f"{WORK}/stellar_{system}.png"
    code, out = sh(f"python -m stellar_predictor.cli analyze --system {system} --plot {png}", cwd=f"{WORK}/STELLAR-PREDICTOR")
    i = out.find("Predicted Gaps"); send(chat, f"🪐 STELLAR · {system}\n\n" + (out[i:] if i >= 0 else out[-3500:]))
    if os.path.exists(png): send_file(chat, png, "photo", f"{system} 轨道图 + Titius–Bode 拟合")

def do_test(chat, args):
    _, a = sh("python -m pytest -q", cwd=f"{WORK}/Agent-Emergence-Lab")
    _, s = sh("python -m pytest -q", cwd=f"{WORK}/STELLAR-PREDICTOR")
    send(chat, f"🧪 测试结果\nAEL：{tail(a)}\nSTELLAR：{tail(s)}")

def do_ael(chat, args):
    nums = [a for a in args if a.isdigit()]; words = [a for a in args if not a.isdigit()]
    topic = " ".join(words) or "记忆的本质"; turns = int(nums[0]) if nums else 3; runs = int(nums[1]) if len(nums) > 1 else 1
    turns, runs = min(turns, 10), min(runs, 5)
    send(chat, f"🧠 开始 AEL 实验：话题「{topic}」，{turns} 轮，每模式 {runs} 次。完成后发结果。")
    d = f"{WORK}/Agent-Emergence-Lab"; sh("rm -rf logs/experiments logs/conversations logs/memory", cwd=d)
    code, out = sh(f"python -m src.experiment --config configs/low_resource_4g.yaml --topic {shlex.quote(topic)} "
                   f"--turns {turns} --runs {runs} --seed 42", cwd=d)
    rep = f"{d}/logs/experiments/latest.md"
    if not os.path.exists(rep): return send(chat, "❌ 实验失败：\n" + out[-3000:])
    md = open(rep, encoding="utf-8").read(); table = md.split("## ")[0]
    send(chat, "✅ AEL 结果\n" + table[:3800])
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
