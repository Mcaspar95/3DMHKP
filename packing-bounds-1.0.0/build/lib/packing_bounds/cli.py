"""JSON-in/JSON-out command line interface."""
import argparse
from contextlib import contextmanager, redirect_stdout
import json
import os
from pathlib import Path
import sys
from . import STAGES, solve, solve_isolated, build_model, __version__

@contextmanager
def solver_output_to_stderr():
    # Keep machine-readable stdout clean even when native solvers print a banner.
    sys.stdout.flush()
    try:fd=sys.stdout.fileno()
    except (AttributeError,OSError):
        with redirect_stdout(sys.stderr):yield
        return
    saved=os.dup(fd)
    try:
        os.dup2(sys.stderr.fileno(),fd)
        yield
        sys.stdout.flush()
    finally:
        os.dup2(saved,fd);os.close(saved)


def main(argv=None):
    parser=argparse.ArgumentParser(description='2D/3D orthogonal knapsack upper bounds')
    parser.add_argument('input',type=Path,help='JSON instance')
    parser.add_argument('--stage',choices=(*STAGES,'all'),default='full')
    parser.add_argument('--mode',choices=('ip','lp','both'),default='ip')
    parser.add_argument('--time-limit',type=float,default=30,help='Solver seconds per model, excluding construction')
    parser.add_argument('--wall-time-limit',type=float,help='Optional total seconds per model, including construction')
    parser.add_argument('--output','-o',type=Path,help='JSON output file (default: stdout)')
    parser.add_argument('--matrix',type=Path,help='Export selected-stage sparse model to a directory')
    parser.add_argument('--version',action='version',version=__version__)
    args=parser.parse_args(argv)
    if args.matrix and args.stage=='all':parser.error('--matrix requires a single stage')
    try:
        instance=json.loads(args.input.read_text())
        stages=STAGES if args.stage=='all' else (args.stage,)
        modes=('ip','lp') if args.mode=='both' else (args.mode,)
        results=[]
        with solver_output_to_stderr():
            for stage in stages:
                for mode in modes:
                    if args.wall_time_limit is None:
                        result=solve(instance,stage,mode,args.time_limit)
                    else:
                        result=solve_isolated(instance,stage,mode,args.time_limit,args.wall_time_limit)
                    results.append(result)
            if args.matrix:build_model(instance,args.stage).export(args.matrix)
        text=json.dumps(results[0] if len(results)==1 else results,indent=2,allow_nan=False)+'\n'
        if args.output:args.output.write_text(text)
        else:sys.stdout.write(text)
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        parser.exit(2,f'packing-bounds: {exc}\n')
