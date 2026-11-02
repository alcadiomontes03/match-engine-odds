"""Probe 2 (2026-11-01): understat DNS from a GitHub runner. Writes data/xg/probe.txt."""
import subprocess, json, urllib.request
from pathlib import Path
out = []
def sh(c):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=25)
        return (r.stdout + r.stderr).strip()[:600]
    except Exception as e:
        return f"ERR {e}"
out.append("resolv.conf: " + sh("cat /etc/resolv.conf | grep -v '^#'"))
for srv in ["", "@1.1.1.1", "@8.8.8.8"]:
    out.append(f"dig understat.com {srv}: " + sh(f"dig +short understat.com {srv}"))
    out.append(f"dig www.understat.com {srv}: " + sh(f"dig +short www.understat.com {srv}"))
try:
    d = json.load(urllib.request.urlopen("https://dns.google/resolve?name=understat.com&type=A", timeout=15))
    out.append("dns.google DoH: " + json.dumps(d.get("Answer") or d)[:400])
except Exception as e:
    out.append(f"DoH ERR {e}")
ip = sh("dig +short understat.com @1.1.1.1 | tail -1")
if ip and ip[0].isdigit():
    out.append(f"curl via {ip}: " + sh(f"curl -s -m 20 --resolve understat.com:443:{ip} -A 'Mozilla/5.0' -o /tmp/u.html -w '%{{http_code}} %{{size_download}}' https://understat.com/league/EPL/2025; grep -c datesData /tmp/u.html"))
out.append("curl www: " + sh("curl -s -m 20 -A 'Mozilla/5.0' -o /tmp/w.html -w '%{http_code} %{size_download}' https://www.understat.com/league/EPL/2025; grep -c datesData /tmp/w.html"))
Path("data/xg").mkdir(parents=True, exist_ok=True)
Path("data/xg/probe.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))
