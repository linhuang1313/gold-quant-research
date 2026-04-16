"""Check status of the two active servers."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVERS = [
    ("C", "connect.westc.seetacloud.com", 16005),
    ("D", "connect.westd.seetacloud.com", 35258),
]

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=20)
    return stdout.read().decode('utf-8', errors='replace').strip()

for name, host, port in SERVERS:
    print(f"\n{'='*70}")
    print(f"Server {name}: {host}:{port}")
    print("=" * 70)
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password="r1zlTZQUb+E4", timeout=15)

        print(f"  Uptime: {run(c, 'uptime')}")
        print(f"  Cores:  {run(c, 'nproc')}")
        print(f"  Disk:   {run(c, 'df -h / | tail -1')}")
        print(f"  Memory: {run(c, 'free -h | head -2')}")

        procs = run(c, "ps aux | grep python | grep -v grep | wc -l")
        print(f"\n  Python processes: {procs}")

        rounds = run(c, "ps aux | grep -E 'run_round|run_exp' | grep python | grep -v grep")
        if rounds:
            print("  Running scripts:")
            for line in rounds.split("\n"):
                parts = line.split()
                if len(parts) > 10:
                    print(f"    PID={parts[1]} CPU={parts[2]}% MEM={parts[3]}% CMD={' '.join(parts[10:])}")
        else:
            print("  No round/exp scripts running!")

        screens = run(c, "screen -ls 2>/dev/null")
        if screens and "No Sockets" not in screens:
            print(f"\n  Screen sessions:")
            for line in screens.split("\n"):
                line = line.strip()
                if line and ("Attached" in line or "Detached" in line):
                    print(f"    {line}")

        print(f"\n  --- Result directories ---")
        for rdir in ["round6_results", "round7_results", "round8_results", "round9_results",
                      "round10_results", "round11_results", "round12_results", "round13_results",
                      "round14_results", "results/round14_results"]:
            path = f"/root/gold-quant-trading/{rdir}"
            exists = run(c, f"test -d {path} && echo YES || echo NO")
            if exists == "YES":
                count = run(c, f"ls {path}/*.txt 2>/dev/null | wc -l")
                latest = run(c, f"ls -lt {path}/*.txt 2>/dev/null | head -1")
                master = run(c, f"tail -15 {path}/00_master_log.txt 2>/dev/null")
                print(f"\n  [{rdir}] {count} txt files")
                if latest:
                    print(f"    Latest: {latest}")
                if master:
                    print(f"    Master log:")
                    for l in master.split("\n"):
                        print(f"      {l}")

        # R14 specific
        r14_log = f"/root/gold-quant-trading/round14_results/r14_log.txt"
        log_exists = run(c, f"test -f {r14_log} && echo YES || echo NO")
        if log_exists == "YES":
            print(f"\n  --- R14 Phase Progress ---")
            phases = run(c, f"grep -E '(>>>|<<<|FAILED)' {r14_log} 2>/dev/null")
            if phases:
                for line in phases.split("\n"):
                    print(f"    {line}")
            print(f"\n  R14 log lines: {run(c, f'wc -l {r14_log}')}")
            print(f"  R14 log tail (non-progress):")
            tail = run(c, f"grep -v '^\\ *[0-9]*%' {r14_log} | grep -v '^$' | tail -5")
            if tail:
                for l in tail.split("\n"):
                    print(f"    {l}")

        c.close()
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}")

print(f"\n{'='*70}")
print("CHECK COMPLETE")
print("=" * 70)
