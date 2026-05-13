#!/usr/bin/env python3
"""Diagnose D2 phase progress on remote."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westd.seetacloud.com', port=41109,
          username='root', password='3sCdENtzYfse', timeout=60)

cmds = [
    ('Process', 'ps -ef | grep python3 | grep -v grep'),
    ('CPU top', 'top -bn1 -o %CPU | head -15'),
    ('Stdout line count', 'wc -l /root/gold-quant-research/results/r200_stdout.txt'),
    ('Backtest count', 'grep -c "^Backtest:" /root/gold-quant-research/results/r200_stdout.txt'),
    ('Last "Backtest" lines with date range', 'grep "^Backtest:" /root/gold-quant-research/results/r200_stdout.txt | tail -5'),
    ('D2 start line', 'grep -n "Phase D2:" /root/gold-quant-research/results/r200_stdout.txt'),
    ('Current file size MB', 'du -m /root/gold-quant-research/results/r200_stdout.txt'),
    ('Tmux status', 'tmux ls'),
]

for label, cmd in cmds:
    _, o, e = c.exec_command(cmd)
    out = o.read().decode('utf-8', errors='replace').strip()
    print(f'=== {label} ===')
    print(out)
    print()

c.close()
