"""
Gold Quant Monitor — combined entry point.

Modes:
    live     — Monitor live MT4 trades, check stop criteria
    refresh  — Update data + re-validate strategies for decay
    all      — Both live monitor and data refresh

Usage:
    python -m monitor.run_monitor --mode live
    python -m monitor.run_monitor --mode refresh
    python -m monitor.run_monitor --mode all --interval 60
    python -m monitor.run_monitor --mode live --loop --interval 15
"""
import argparse
import time
import datetime as dt


def run_live(loop: bool = False, interval: int = 15):
    from monitor.live_monitor import run_once, run_loop
    if loop:
        run_loop(interval)
    else:
        run_once()


def run_refresh():
    from monitor.update_data import run as update_run
    from monitor.auto_revalidate import run as revalidate_run

    print("\n" + "=" * 60)
    print("  Phase 1: Data Update")
    print("=" * 60)
    updated = update_run()

    print("\n" + "=" * 60)
    print("  Phase 2: Re-validation")
    print("=" * 60)
    results = revalidate_run()
    return results


def run_all(loop: bool = False, interval: int = 60):
    """Run live check + refresh in a combined loop."""
    if not loop:
        run_live(loop=False)
        print()
        run_refresh()
        return

    print(f"Starting combined monitor (interval={interval}min)")
    print(f"Press Ctrl+C to stop.\n")
    while True:
        try:
            now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"\n{'#' * 60}")
            print(f"  Combined Monitor Cycle — {now}")
            print(f"{'#' * 60}\n")

            run_live(loop=False)
            print()
            run_refresh()

            print(f"\nNext cycle in {interval} minutes...")
            time.sleep(interval * 60)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Gold Quant Research — Monitoring Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m monitor.run_monitor --mode live               # one-shot live check
  python -m monitor.run_monitor --mode live --loop         # continuous live monitor
  python -m monitor.run_monitor --mode refresh             # update data + revalidate
  python -m monitor.run_monitor --mode all                 # one-shot: live + refresh
  python -m monitor.run_monitor --mode all --loop --interval 60  # full loop hourly
        """,
    )
    parser.add_argument(
        "--mode", choices=["live", "refresh", "all"], default="live",
        help="Monitor mode (default: live)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run in continuous loop",
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Minutes between checks (default: 15 for live, 60 for all)",
    )
    args = parser.parse_args()

    if args.mode == "live":
        run_live(loop=args.loop, interval=args.interval)
    elif args.mode == "refresh":
        run_refresh()
    elif args.mode == "all":
        interval = args.interval if args.interval != 15 else 60
        run_all(loop=args.loop, interval=interval)


if __name__ == "__main__":
    main()
