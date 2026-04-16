"""Check status of all remote servers."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = [
    ("A", "connect.westb.seetacloud.com", 42894),
    ("B", "connect.westb.seetacloud.com", 25821),
    ("C-R14", "connect.westc.seetacloud.com", 16005),
]

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=15)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, host, port in SERVERS:
    print(f"\n{'='*60}")
    print(f"Server {name}: {host}:{port}")
    print("=" * 60)
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password="r1zlTZQUb+E4", timeout=10)

        print(f"  Uptime: {run(c, 'uptime')}")
        print(f"  Cores: {run(c, 'nproc')}")

        procs = run(c, "ps aux | grep python | grep -v grep | wc -l")
        print(f"  Python processes: {procs}")

        rounds = run(c, "ps aux | grep -E 'run_round|run_exp' | grep python | grep -v grep")
        if rounds:
            print("  Running scripts:")
            for line in rounds.split("\n"):
                parts = line.split()
                if len(parts) > 10:
                    print(f"    PID={parts[1]} CPU={parts[2]}% MEM={parts[3]}% CMD={' '.join(parts[10:])}")
        else:
            print("  No round/exp scripts running!")

        for rdir in ["round6_results", "round7_results", "round8_results", "round9_results",
                      "round10_results", "round11_results", "round12_results", "round13_results",
                      "round14_results", "results/round14_results"]:
            path = f"/root/gold-quant-trading/{rdir}"
            exists = run(c, f"test -d {path} && echo YES || echo NO")
            if exists == "YES":
                count = run(c, f"ls {path}/*.txt 2>/dev/null | wc -l")
                latest = run(c, f"ls -lt {path}/*.txt 2>/dev/null | head -3")
                print(f"  {rdir}/: {count} txt files")
                if latest:
                    for l in latest.split("\n")[:3]:
                        print(f"    {l}")

                master = run(c, f"cat {path}/00_master_log.txt 2>/dev/null | tail -10")
                if master:
                    print(f"  Master log (last 10 lines):")
                    for l in master.split("\n"):
                        print(f"    {l}")

                log = run(c, f"tail -20 {path}/r14_log.txt 2>/dev/null")
                if log:
                    print(f"  R14 log (last 20 lines):")
                    for l in log.split("\n"):
                        print(f"    {l}")

        screens = run(c, "screen -ls 2>/dev/null")
        if screens and "No Sockets" not in screens:
            print(f"  Screen sessions:")
            for line in screens.split("\n"):
                line = line.strip()
                if line and ("Attached" in line or "Detached" in line):
                    print(f"    {line}")

        c.close()
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}")

print("\n" + "=" * 60)
print("ALL SERVERS CHECKED")
print("=" * 60)
