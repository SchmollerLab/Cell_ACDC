import inspect, os, datetime, sys, traceback
import atexit
import linecache
from collections import defaultdict

from . import cellacdc_path, myutils

import gc
import psutil
import time
import functools

_LINE_BENCHMARK_TRACE_LIMIT = 10000

_LINE_BENCHMARK_STATS = defaultdict(
    lambda: {
        'count': 0,
        'traced_count': 0,
        'untracked_count': 0,
        'total_time': 0.0,
        'min_time': float('inf'),
        'max_time': 0.0,
        'filename': None,
        'line_stats': defaultdict(
            lambda: {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
            }
        ),
    }
)

def _get_benchmark_line_snippet(filename, lineno, max_chars=30):
    if lineno == 'return':
        return '<return>'
    if not filename:
        return '<unknown>'

    line = linecache.getline(filename, lineno).strip()
    if not line:
        return '<blank>'

    if len(line) <= max_chars:
        # fill up to max_chars for better alignment
        line = line.ljust(max_chars)
        return line
    return f'{line[:max_chars-3]}...'

def _print_line_benchmark_session_stats():
    if not _LINE_BENCHMARK_STATS:
        return

    print('\nLine benchmark session summary:')
    for func_name, stats in sorted(_LINE_BENCHMARK_STATS.items()):
        total_count = stats['count']
        traced_count = stats['traced_count']
        untracked_count = stats['untracked_count']
        if total_count == 0:
            continue

        if traced_count:
            mean_time = stats['total_time'] / traced_count
            print(
                f'{func_name}: n={total_count} | '
                f'traced={traced_count} | '
                f'untracked={untracked_count} | '
                f'mean={mean_time*1000:.3f} ms | '
                f'min={stats["min_time"]*1000:.3f} ms | '
                f'max={stats["max_time"]*1000:.3f} ms | '
                f'total={stats["total_time"]*1000:.3f} ms'
            )
        else:
            print(
                f'{func_name}: n={total_count} | '
                f'traced=0 | '
                f'untracked={untracked_count}'
            )

        line_stats = stats['line_stats']
        top_lines = sorted(
            line_stats.items(),
            key=lambda item: item[1]['total_time'],
            reverse=True
        )[:10]
        filename = stats['filename']
        for (start_line, end_line), line_stat in top_lines:
            line_mean = line_stat['total_time'] / line_stat['count']
            line_snippet = _get_benchmark_line_snippet(filename, start_line)
            print(
                f'  {line_snippet:<30} {start_line} -> {end_line}: '
                f'n={line_stat["count"]} | '
                f'mean={line_mean*1000:.3f} ms | '
                f'total={line_stat["total_time"]*1000:.3f} ms'
            )

atexit.register(_print_line_benchmark_session_stats)

def showRefGraph(object_str:str, debug:bool=True):
    """Save a reference graph of the given object type.


    Parameters
    ----------
    object_str : str
        For example `loadData` (So the class name, not the instance name).
    debug : bool, optional
        If `False`, the function does nothing. Default is `True`.
    """
    if not debug:
        return
    
    try:
        import objgraph
    except ImportError:
        conda_prefix, pip_prefix = myutils.get_pip_conda_prefix()

        print(f"objgraph is not installed. Install it with '{pip_prefix} objgraph' to use reference graph features, as well as https://www.graphviz.org/")
        return

    caller_func = inspect.currentframe().f_back.f_code.co_name
    caller_file = inspect.currentframe().f_back.f_code.co_filename
    caller_file = os.path.basename(caller_file).rstrip('.py')
    caller_line = inspect.currentframe().f_back.f_lineno
    timestap = datetime.datetime.now().strftime('%H_%M_%S')

    ref_graph_path = os.path.join(
        os.path.dirname(cellacdc_path),
        '.ref_graphs'
    )

    os.makedirs(ref_graph_path, exist_ok=True)
    
    filename = os.path.join(ref_graph_path, f'ref_graph_{timestap}_{object_str}_from_{caller_file}_{caller_func}_{caller_line}.svg')

    timestap = datetime.datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    callingframe = outerframes[1].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filepath = callingframe_info.filename
    fileinfo_str = (
        f'File "{filepath}", line {callingframe_info.lineno} - {timestap}:'
    )


    gc.collect()
    instances = objgraph.by_type(object_str)
    if instances:
        objgraph.show_backrefs(instances, max_depth=500, filename=filename)
        print(fileinfo_str, f'Graph saved as "{filename}" \n for {len(instances)} instances of "{object_str}"')
    else:
        print(fileinfo_str, f'No {object_str} instances found.')

def print_largest_attributes(
    obj, top_n=10, return_list=False, show_percent=True, process_mem=None
):
    attrs = []
    total = 0
    for attr in dir(obj):
        if attr.startswith('__'):
            continue
        try:
            val = getattr(obj, attr)
            size = sys.getsizeof(val)
            attrs.append((attr, size, type(val)))
            total += size
        except Exception:
            continue
    # Sort by size descending
    attrs.sort(key=lambda x: x[1], reverse=True)
    if process_mem is not None:
        print(f"Total process memory: {process_mem:,} bytes")
    print(f"Total attribute memory: {total:,} bytes")
    for attr, size, typ in attrs[:top_n]:
        percent = (size / total * 100) if total > 0 else 0
        proc_percent = (size / process_mem * 100) if process_mem else 0
        if show_percent and process_mem:
            print(f"{attr:30} {size:10,} bytes  {percent:6.2f}% of obj  {proc_percent:6.2f}% of proc   {typ}")
        elif show_percent:
            print(f"{attr:30} {size:10,} bytes  {percent:6.2f}%   {typ}")
        else:
            print(f"{attr:30} {size:10,} bytes   {typ}")
    if return_list:
        return attrs[:top_n]

def print_call_stack(debug=True, depth=None):
    if not debug:
        return
    stack = traceback.format_stack()
    stack = stack[:-1]
    if depth:
        depth = depth + 1
        stack = stack[-depth:] 
    print("Call stack:")
    for line in stack:
        print(line.strip())

def print_largest_attributes_for_all_classes(package_prefix="cellacdc", top_n=5):
    # Find all classes defined in your package
    classes = set()
    for obj in gc.get_objects():
        if isinstance(obj, type):
            if getattr(obj, "__module__", "").startswith(package_prefix):
                classes.add(obj)
    # For each class, find all instances and print largest attributes
    for cls in classes:
        instances = [o for o in gc.get_objects() if isinstance(o, cls)]
        if not instances:
            continue
        print(f"\nClass: {cls.__module__}.{cls.__name__} ({len(instances)} instances)")
        for i, inst in enumerate(instances):
            print(f"  Instance {i+1}:")
            try:
                print_largest_attributes(inst, top_n=top_n)
            except Exception as e:
                print(f"    Could not inspect instance: {e}")

def print_largest_classes(package_prefix="cellacdc", top_n=10, max_instances=100):
    """
    Print classes (optionally filtered by module prefix) sorted by total memory usage.
    Uses pympler.asizeof for deep size estimation.
    """

    import gc
    import psutil
    import os
    try:
        from pympler import asizeof
    except ImportError:
        print("pympler not installed. Run: pip install pympler")
        return

    process = psutil.Process(os.getpid())
    process_mem = process.memory_info().rss

    class_to_instances = {}

    # ✅ Single pass over objects
    for obj in gc.get_objects():
        try:
            cls = type(obj)
            module = getattr(cls, "__module__", None)

            if package_prefix is not None:
                if not isinstance(module, str) or not module.startswith(package_prefix):
                    continue

            class_to_instances.setdefault(cls, []).append(obj)

        except Exception:
            continue

    class_mem = []

    for cls, instances in class_to_instances.items():
        n = len(instances)

        # ✅ Safe sampling
        if n > max_instances:
            step = max(1, n // max_instances)
            sampled = instances[::step]
        else:
            sampled = instances

        total_size = 0
        counted = 0

        for inst in sampled:
            try:
                size = asizeof.asizeof(inst)
                total_size += size
                counted += 1
            except Exception:
                continue

        # scale up if sampled
        if counted > 0 and n > counted:
            total_size *= (n / counted)

        if total_size > 0:
            class_mem.append((cls, total_size, n))

    # ✅ Sort by memory
    class_mem.sort(key=lambda x: x[1], reverse=True)

    print(f"Total process memory: {process_mem/1024**2:.1f} MB")
    print(f"{'Class':60} {'Instances':>10} {'Total MB':>12} {'% of proc':>10}")

    for cls, total_size, n in class_mem[:top_n]:
        percent = (total_size / process_mem * 100) if process_mem else 0

        name = f"{cls.__module__}.{cls.__name__}"

        print(f"{name:<60} {n:10} {total_size/1024**2:12.2f} {percent:9.2f}%")


# Example usage:
# print_largest_classes("cellacdc", top_n=10)

# Return a benchmark checkpoint with caller line information.
def return_timer_and_line(benchmarking=True):
    if not benchmarking:
        return None
    timestamp = time.perf_counter()
    line = inspect.currentframe().f_back.f_lineno # is super fast!
    return (timestamp, line)

def print_benchmarks(timers, benchmarking=True):
    if not benchmarking:
        return
    checkpoints = [timer for timer in timers if timer is not None]
    if len(checkpoints) < 2:
        return

    print("Benchmarks:")
    for (start_time, start_line), (end_time, end_line) in zip(
        checkpoints, checkpoints[1:]
    ):
        duration = end_time - start_time
        print(
            f"Line {start_line} -> {end_line}: "
            f"{duration:.6f} seconds"
        )

    total_duration = checkpoints[-1][0] - checkpoints[0][0]
    print(f"Total: {total_duration:.6f} seconds")
    
def line_benchmark(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        stats_key = f'{func.__module__}.{func.__qualname__}'
        stats = _LINE_BENCHMARK_STATS[stats_key]
        stats['count'] += 1

        if stats['traced_count'] >= _LINE_BENCHMARK_TRACE_LIMIT:
            stats['untracked_count'] += 1
            return func(*args, **kwargs)

        target_code = func.__code__
        filename = target_code.co_filename
        checkpoints = []
        last_time = None
        last_line = None

        def tracer(frame, event, arg):
            nonlocal last_time, last_line

            if frame.f_code is not target_code:
                return tracer

            now = time.perf_counter()

            if event == "call":
                last_time = now
                last_line = frame.f_lineno
                return tracer

            if event == "line":
                if last_time is not None and last_line is not None:
                    checkpoints.append((last_line, frame.f_lineno, now - last_time))
                last_time = now
                last_line = frame.f_lineno
                return tracer

            if event == "return":
                if last_time is not None and last_line is not None:
                    checkpoints.append((last_line, "return", now - last_time))
                return tracer

            return tracer

        old_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            result = func(*args, **kwargs)
        finally:
            sys.settrace(old_trace)

        total = sum(dt for _, _, dt in checkpoints)
        stats['traced_count'] += 1
        stats['total_time'] += total
        stats['min_time'] = min(stats['min_time'], total)
        stats['max_time'] = max(stats['max_time'], total)
        stats['filename'] = filename

        for start_line, end_line, dt in checkpoints:
            line_stat = stats['line_stats'][(start_line, end_line)]
            line_stat['count'] += 1
            line_stat['total_time'] += dt
            line_stat['min_time'] = min(line_stat['min_time'], dt)
            line_stat['max_time'] = max(line_stat['max_time'], dt)

        return result

    return wrapper