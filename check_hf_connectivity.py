import socket
import sys
import urllib.error
import urllib.request
TIMEOUT = 15
URLS = [
    "https://huggingface.co/",
    "https://hf-mirror.com/",
    "https://hf-mirror.com/Rostlab/ProstT5/resolve/main/config.json",
]
def check_dns(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = sorted({x[4][0] for x in infos})
        print(f"DNS OK  {host} -> {ips[:5]}{' ...' if len(ips) > 5 else ''}")
    except OSError as e:
        print(f"DNS FAIL {host} -> {e}")
def check_url(url: str) -> None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            print(f"HTTP OK  {url} -> status {r.status}")
    except urllib.error.HTTPError as e:
        print(f"HTTP     {url} -> status {e.code} ({e.reason})")
    except Exception as e:
        print(f"HTTP FAIL {url} -> {type(e).__name__}: {e}")
def main() -> int:
    print("== DNS ==")
    for h in ("huggingface.co", "hf-mirror.com", "cdn-lfs.huggingface.co"):
        check_dns(h)
    print()
    print(f"== HTTPS (timeout {TIMEOUT}s) ==")
    for u in URLS:
        check_url(u)
    print()
    print("Interpretation: DNS OK + HTTP 200/301/302 on URLs => connectivity is fine.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
