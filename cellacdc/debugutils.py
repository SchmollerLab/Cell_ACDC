import inspect, os, datetime, sys, traceback

from . import cellacdc_path, myutils

import gc
import psutil

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
    Print the classes defined in the given package prefix, sorted by their total memory usage
    (sum of up to max_instances per class), as a percentage of the total process memory.
    Uses pympler.asizeof for deep memory measurement.
    """
    try:
        from pympler import asizeof
    except ImportError:
        conda_prefix, pip_prefix = myutils.get_pip_conda_prefix()

        print(f"pympler is not installed. Install it with '{pip_prefix}l pympler' to use this function.")
        return
    process = psutil.Process()
    process_mem = process.memory_info().rss

    # First, collect all classes and build a mapping to their instances in one pass
    classes = set()
    class_to_instances = {}
    for obj in gc.get_objects():
        if isinstance(obj, type):
            if getattr(obj, "__module__", "").startswith(package_prefix):
                classes.add(obj)
    for obj in gc.get_objects():
        obj_type = type(obj)
        if obj_type in classes:
            class_to_instances.setdefault(obj_type, []).append(obj)

    class_mem = []

    for cls, instances in class_to_instances.items():
        # Only use up to max_instances per class
        num_instances = len(instances)
        step_instances = num_instances / max_instances if num_instances > max_instances else 1

        limited_instances = instances[::int(step_instances)] if step_instances > 1 else instances
        total_size = 0
        for inst in limited_instances:
            try:
                total_size += asizeof.asizeof(inst)
            except Exception:
                continue
        if total_size > 0:
            class_mem.append((cls, total_size, len(instances)))
    class_mem.sort(key=lambda x: x[1], reverse=True)
    print(f"Total process memory: {process_mem:,} bytes")
    print(f"{'Class':50} {'Instances':>10} {'Total bytes':>15} {'% of proc':>12}")
    for cls, total_size, n in class_mem[:top_n]:
        percent = (total_size / process_mem * 100) if process_mem else 0
        print(f"{cls.__module__+'.'+cls.__name__:<50} {n:10} {total_size:15,} {percent:11.2f}%")


# Example usage:
# print_largest_classes("cellacdc", top_n=10)
