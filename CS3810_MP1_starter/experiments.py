"""
CS3810 Mini-Project 1 - Part 4: Experiment Harness (30 points)
==============================================================

The measurement plumbing is written for you: this file runs every
grid x algorithm x heuristic combination in a subprocess with a timeout,
and writes the raw numbers to results.csv.

    python experiments.py                 # full run, 60s timeout per config
    python experiments.py --quick         # small grids only, for a fast check
    python experiments.py --timeout 120   # be more patient
    python experiments.py --grids g5_rooms g6_corridor

What is left for YOU is at the bottom of the file: turning results.csv into
the table and the two plots your report needs. That part is graded; this
part is not.

A note on timeouts. Some configurations are genuinely infeasible - IDA* with
a weak heuristic on the larger grids will re-expand millions of nodes. A row
marked TIMEOUT is a real finding and belongs in your table. Do not delete it,
and do not raise the timeout until the number is "nicer". Discuss it.

A note on runtime. Compare NODE COUNTS across algorithms; they are a property
of the algorithm. Wall-clock time is a property of your laptop, your Python
version, and what else you had open. Use it as supporting evidence only.
"""

import argparse
import csv
import multiprocessing as mp
import queue as queue_mod
import sys
import time

from test_grids import GRIDS, EXAMPLE, parse_grid

# Every combination the harness will try.
ALGORITHMS = ['dfs', 'astar', 'idastar']
HEURISTIC_NAMES = ['h0', 'h1', 'h2', 'h3']

# DFS ignores the heuristic, so it is run once per grid with this label.
NO_HEURISTIC = '-'

QUICK_GRIDS = ['g1_tiny', 'g2_open', 'g3_blocks']

CSV_FIELDS = [
    'grid', 'rows', 'cols', 'n_dirty',
    'algorithm', 'heuristic', 'status',
    'cost', 'nodes_expanded', 'max_frontier', 'iterations', 'seconds',
]


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------

def build_problem(grid_name):
    """Construct a VacuumWorld for a named grid."""
    from vacuum_world import VacuumWorld
    art = EXAMPLE if grid_name == 'example' else GRIDS[grid_name]
    grid, start, dirty = parse_grid(art)
    return VacuumWorld(grid, start, dirty)


def run_single(algorithm, grid_name, heuristic_name):
    """Run one configuration and return a result dict. Runs in a subprocess."""
    from search import dfs_search, astar_search, idastar_search
    from heuristics import HEURISTICS

    problem = build_problem(grid_name)
    started = time.perf_counter()

    if algorithm == 'dfs':
        plan, expanded, frontier = dfs_search(problem)
        iterations = None
    elif algorithm == 'astar':
        plan, expanded, frontier = astar_search(problem, HEURISTICS[heuristic_name])
        iterations = None
    elif algorithm == 'idastar':
        plan, expanded, iterations = idastar_search(problem, HEURISTICS[heuristic_name])
        frontier = None
    else:
        raise ValueError("unknown algorithm: %s" % algorithm)

    elapsed = time.perf_counter() - started

    return {
        'status': 'ok',
        'cost': None if plan is None else len(plan),
        'nodes_expanded': expanded,
        'max_frontier': frontier,
        'iterations': iterations,
        'seconds': round(elapsed, 4),
    }


def _worker(out, algorithm, grid_name, heuristic_name):
    """Subprocess entry point. Puts (status, payload) on the queue."""
    sys.setrecursionlimit(100000)
    try:
        out.put(('ok', run_single(algorithm, grid_name, heuristic_name)))
    except NotImplementedError as exc:
        out.put(('skip', str(exc)))
    except Exception as exc:  # noqa: BLE001 - report anything the student hits
        out.put(('error', '%s: %s' % (type(exc).__name__, exc)))


def run_with_timeout(algorithm, grid_name, heuristic_name, timeout):
    """Run one configuration in a subprocess, killing it after `timeout` seconds."""
    out = mp.Queue()
    proc = mp.Process(target=_worker,
                      args=(out, algorithm, grid_name, heuristic_name))
    started = time.perf_counter()
    proc.start()
    try:
        status, payload = out.get(timeout=timeout)
    except queue_mod.Empty:
        status, payload = 'timeout', None
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()

    elapsed = round(time.perf_counter() - started, 4)

    if status == 'ok':
        return payload
    if status == 'timeout':
        return {'status': 'TIMEOUT', 'cost': None, 'nodes_expanded': None,
                'max_frontier': None, 'iterations': None, 'seconds': elapsed}
    if status == 'skip':
        return {'status': 'SKIP', 'cost': None, 'nodes_expanded': None,
                'max_frontier': None, 'iterations': None, 'seconds': None}
    return {'status': 'ERROR (%s)' % payload, 'cost': None,
            'nodes_expanded': None, 'max_frontier': None,
            'iterations': None, 'seconds': elapsed}


# ---------------------------------------------------------------------------
# The full sweep
# ---------------------------------------------------------------------------

def configurations(grid_names):
    """Yield (algorithm, grid, heuristic) triples to measure."""
    for grid_name in grid_names:
        yield ('dfs', grid_name, NO_HEURISTIC)
        for algorithm in ('astar', 'idastar'):
            for heuristic_name in HEURISTIC_NAMES:
                yield (algorithm, grid_name, heuristic_name)


def sweep(grid_names, timeout, out_path='results.csv'):
    """Run every configuration and write results.csv. Returns the rows."""
    rows = []
    print("%-13s %-8s %-4s %-10s %s" % (
        'grid', 'algo', 'h', 'status', 'nodes / cost / time'))
    print('-' * 68)

    for algorithm, grid_name, heuristic_name in configurations(grid_names):
        problem_art = EXAMPLE if grid_name == 'example' else GRIDS[grid_name]
        grid, _, dirty = parse_grid(problem_art)

        result = run_with_timeout(algorithm, grid_name, heuristic_name, timeout)
        row = {
            'grid': grid_name,
            'rows': len(grid),
            'cols': len(grid[0]),
            'n_dirty': len(dirty),
            'algorithm': algorithm,
            'heuristic': heuristic_name,
        }
        row.update(result)
        rows.append(row)

        if result['status'] == 'ok':
            detail = "%s nodes, cost %s, %.2fs" % (
                result['nodes_expanded'], result['cost'], result['seconds'])
        else:
            detail = ''
        print("%-13s %-8s %-4s %-10s %s" % (
            grid_name, algorithm, heuristic_name, result['status'], detail))

    with open(out_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})

    print('-' * 68)
    print("wrote %d rows to %s" % (len(rows), out_path))
    return rows


# ---------------------------------------------------------------------------
# YOUR WORK STARTS HERE
# ---------------------------------------------------------------------------

def make_table(rows):
    """TODO: build the results table for your report.

    The handout asks for solution cost, nodes expanded, max frontier size
    (DFS and A*) or iterations (IDA*), and runtime, for every configuration.

    A readable table is not a CSV dump. Decide what goes in rows and what
    goes in columns, and make it possible to compare algorithms at a glance.
    pandas.DataFrame(rows) and .pivot_table() will do most of the work, or
    write it out by hand - either is fine.
    """
    raise NotImplementedError("Part 4: build your results table")


def plot_scaling(rows):
    """TODO: plot #1 - how does each algorithm scale with the amount of dirt?

    Suggested shape: x = number of dirty cells, y = nodes expanded, one line
    per algorithm (hold the heuristic fixed at h2). A log scale on y will
    probably help; say in the caption why you chose it.

    Save to figures/scaling.png.
    """
    raise NotImplementedError("Part 4: plot nodes expanded vs. problem size")


def plot_heuristics(rows):
    """TODO: plot #2 - what does a better heuristic buy you?

    Suggested shape: A* nodes expanded under h0 vs h1 vs h2 (and h3 if you
    did the bonus), grouped by grid. This is the evidence for your answer to
    discussion question 2.

    Save to figures/heuristics.png.
    """
    raise NotImplementedError("Part 4: plot the effect of the heuristic")


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    parser.add_argument('--timeout', type=float, default=60.0,
                        help="seconds per configuration (default: 60)")
    parser.add_argument('--quick', action='store_true',
                        help="only the three smallest grids")
    parser.add_argument('--grids', nargs='+', metavar='NAME',
                        help="specific grids to run (default: all six)")
    parser.add_argument('--out', default='results.csv',
                        help="where to write the raw results")
    parser.add_argument('--analyze', action='store_true',
                        help="also run your table and plot functions")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.grids:
        grid_names = args.grids
    elif args.quick:
        grid_names = QUICK_GRIDS
    else:
        grid_names = list(GRIDS)

    unknown = [g for g in grid_names if g != 'example' and g not in GRIDS]
    if unknown:
        print("unknown grid(s): %s" % ', '.join(unknown))
        print("available: example, %s" % ', '.join(GRIDS))
        return 2

    rows = sweep(grid_names, args.timeout, args.out)

    if args.analyze:
        make_table(rows)
        plot_scaling(rows)
        plot_heuristics(rows)

    return 0


if __name__ == "__main__":
    # The multiprocessing guard above is required on Windows and macOS.
    sys.exit(main())
