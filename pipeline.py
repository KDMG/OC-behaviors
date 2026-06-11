from __future__ import annotations

import argparse
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def count_process_executions(ocel_path: str, leading_type: str) -> int:
    p = Path(ocel_path)
    if p.suffix.lower() not in (".sqlite", ".db"):
        return 0
    try:
        con = sqlite3.connect(str(p))
        row = con.execute(
            "SELECT COUNT(*) FROM object WHERE ocel_type = ?", (leading_type,)
        ).fetchone()
        con.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0

def detect_types_to_remove(ocel_path: str, tau: float = 0.9) -> list[str]:
    from mlpa.my_ocel_importer import apply as load_ocel_sqlite
    from mlpa.MLPAMiner import mlpaDiscovery

    print(f"\n[mlpa] Running mlpaDiscovery on {ocel_path}")
    ocel = load_ocel_sqlite(ocel_path)
    _, _, process_view_with_events = mlpaDiscovery(ocel, tau=tau)

    no_activity: set[str] = set()
    has_activity: set[str] = set()

    for _level, views in process_view_with_events.items():
        for obj_types, activities in views:
            for t in obj_types:
                if activities:
                    has_activity.add(t)
                else:
                    no_activity.add(t)

    remove = sorted(no_activity - has_activity)
    print(f"[mlpa] Types with no activities → will be removed: {remove}")
    return remove

def run_filter(input_path: str, output_path: str, remove_types: list[str]) -> None:
    cmd = [
        sys.executable, str(_ROOT / "ocel_filter.py"),
        "auto", input_path, output_path,
        *remove_types,
    ]
    print(f"\n[filter] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_mining(
    ocel_path: str,
    bundle_path: str,
    leading: str,
    miner: str,
    kpis: list[str],
    s_min: float,
    s_max: float,
    support_abs: bool,
    quiet: bool,
    extra_args: list[str]
) -> None:
    cmd = [
        sys.executable, str(_ROOT / "rho_lift.py"),
        ocel_path,
        "--leading", leading,
        "--miner", miner,
        "--bundle", bundle_path,
        "--s-min", str(s_min),
        "--s-max", str(s_max),
    ]
    for kpi in kpis:
        cmd += ["--kpi", kpi]
    if support_abs:
        cmd.append("--support-abs")
    if quiet:
        cmd.append("--quiet")
    cmd.extend(extra_args)

    print(f"\n[mine] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_explorer(bundle_path: str, host: str = "127.0.0.1", port: int = 8050) -> None:
    cmd = [
        sys.executable, str(_ROOT / "explorer" / "oc_explorer.py"),
        bundle_path,
        "--host", host,
        "--port", str(port),
    ]
    print(f"\n[explorer] {' '.join(cmd)}")
    print(f"[explorer] Open http://{host}:{port}/ in your browser")
    subprocess.run(cmd, check=True)

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="OC-behaviors pipeline: mlpa → filter → mine → (explore)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument(
        "ocel", nargs="?", default=None,
        help="Input OCEL path (.sqlite). Not required with --explore-only.",
    )

    ap.add_argument(
        "--explore-only", metavar="BUNDLE",
        help="Skip everything and open the explorer on an existing .pkl bundle.",
    )

    mlpa = ap.add_argument_group("mlpa")
    mlpa.add_argument(
        "--tau", type=float, default=0.9,
        help="mlpaDiscovery tau threshold (default: 0.9).",
    )
    mlpa.add_argument(
        "--remove-types", nargs="*", metavar="TYPE",
        help="Object types to remove. If omitted, auto-detected from mlpa output.",
    )
    mlpa.add_argument(
        "--skip-mlpa", action="store_true",
        help="Skip the mlpa step entirely (requires --remove-types or --skip-filter).",
    )

    flt = ap.add_argument_group("filter")
    flt.add_argument(
        "--filtered", metavar="PATH",
        help="Output path for the filtered OCEL. "
             "Default: <input>_filtered.sqlite next to the input file.",
    )
    flt.add_argument(
        "--skip-filter", action="store_true",
        help="Skip the filter step and mine the raw input OCEL directly.",
    )

    mine = ap.add_argument_group("mining")
    mine.add_argument(
        "--bundle", metavar="PATH",
        help="Output .pkl bundle for the interactive explorer.",
    )
    mine.add_argument(
        "--leading", metavar="TYPE",
        help="Leading object type for execution extraction (required for mining).",
    )
    mine.add_argument(
        "--miner", choices=["gspan", "subdue"], default="gspan",
    )
    mine.add_argument(
        "--kpi", action="append", dest="kpis", metavar="KPI",
        help="KPI(s) to compute. Repeatable. Default: duration + n_events.",
    )
    mine.add_argument(
        "--s-min", type=float, default=None,
        help="Min support. Default: 5%% of process executions (absolute)."
    )
    mine.add_argument(
        "--s-max", type=float, default=None,
        help="Max support. Default: 80%% of process executions (absolute)."
    )
    mine.add_argument(
        "--support-abs", action="store_true",
        help="Interpret --s-min/--s-max as absolute counts. "
             "Enabled automatically when defaults are used.",
    )
    mine.add_argument("--quiet", action="store_true", help="Silence rho_lift progress logs.")

    exp = ap.add_argument_group("explorer")
    exp.add_argument(
        "--explore", action="store_true",
        help="Launch the interactive explorer automatically after mining.",
    )
    exp.add_argument("--host", default="127.0.0.1")
    exp.add_argument("--port", type=int, default=8050)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args, extra_mining_args = ap.parse_known_args(argv)

    if args.explore_only:
        run_explorer(args.explore_only, host=args.host, port=args.port)
        return 0

    if args.ocel is None:
        ap.error("ocel positional argument is required (or use --explore-only).")

    ocel_path = args.ocel
    inp = Path(ocel_path)

    filtered_path = args.filtered or str(inp.parent / (inp.stem + "_filtered.sqlite"))
    kpis = args.kpis or ["duration", "n_events"]

    if args.explore_only is None and not args.skip_mlpa:
        if args.remove_types is not None:
            remove_types = list(args.remove_types)
            print(f"[mlpa] Using manual --remove-types: {remove_types}")
        else:
            remove_types = detect_types_to_remove(ocel_path, tau=args.tau)
    else:
        remove_types = list(args.remove_types or [])

    if args.skip_filter:
        mining_input = ocel_path
        print(f"[filter] Skipped — mining from original: {mining_input}")
    else:
        run_filter(ocel_path, filtered_path, remove_types)
        mining_input = filtered_path

    if args.bundle is None:
        ap.error("--bundle is required for mining.")
    if args.leading is None:
        ap.error("--leading is required for mining.")

    s_min = args.s_min
    s_max = args.s_max
    support_abs = args.support_abs
    if s_min is None or s_max is None:
        n_exec = count_process_executions(mining_input, args.leading)
        if n_exec > 0:
            if s_min is None:
                s_min = max(1, math.floor(0.05 * n_exec))
            if s_max is None:
                s_max = max(1, math.floor(0.80 * n_exec))
            support_abs = True
            print(f"[pipeline] Process executions: {n_exec} "
                  f"-> s-min={s_min}, s-max={s_max} (absolute)")
        else:
            s_min = s_min if s_min is not None else 0.05
            s_max = s_max if s_max is not None else 0.80
            print(f"[pipeline] Could not count executions; using fractions "
                  f"s-min={s_min}, s-max={s_max}")

    run_mining(
        mining_input, args.bundle, args.leading,
        args.miner, kpis,
        s_min, s_max, support_abs,
        args.quiet, extra_mining_args,
    )

    print(f"\n[pipeline] Bundle saved to: {args.bundle}")

    if args.explore:
        run_explorer(args.bundle, host=args.host, port=args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
