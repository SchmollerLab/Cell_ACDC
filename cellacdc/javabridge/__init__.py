"""__init__.py - the javabridge package

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

"""

import os.path

try:
    from _version import __version__
except ImportError:
    # We're running in a tree that doesn't have a _version.py, so we don't know what our version is.
    __version__ = "0.0.0"

# We must dynamically find libjvm.so since its unlikely to be in the same place
# as it was on the distribution on which javabridge was built.
import sys
if sys.platform.startswith('linux'):
    from .locate import find_jre_bin_jdk_so
    _, jdk_so = find_jre_bin_jdk_so()
    if jdk_so:
        import ctypes
        ctypes.cdll.LoadLibrary(jdk_so)

_jars_dir = os.path.join(os.path.dirname(__file__), 'jars')

#: List of absolute paths to JAR files that are required for the
#: Javabridge to work.
JARS = [os.path.realpath(os.path.join(_jars_dir, name + '.jar'))
        for name in ['rhino-1.7R4', 'runnablequeue', 'cpython']]


from .jutil import start_vm, kill_vm, vm, activate_awt, deactivate_awt

from .jutil import attach, detach, get_env


# JavaScript
from .jutil import run_script, unwrap_javascript


# Operations on Java objects
from .jutil import call, get_static_field, static_call, \
    is_instance_of, make_instance, set_static_field, to_string, \
    get_field, set_field, make_static_call

# Make Python object that wraps a Java object
from .jutil import make_method, make_new, make_call, box
from .wrappers import JWrapper, JClassWrapper, JProxy

from .jutil import get_nice_arg

# Useful collection wrappers
from .jutil import get_dictionary_wrapper, jdictionary_to_string_dictionary, \
    jenumeration_to_string_list, get_enumeration_wrapper, iterate_collection, \
    iterate_java, make_list, get_collection_wrapper, make_future_task, \
    make_map, get_map_wrapper

# Reflection. (These use make_method or make_new internally.)
from .jutil import get_class_wrapper, get_field_wrapper, class_for_name, \
    get_constructor_wrapper, get_method_wrapper

# Ensure that callables, runnables and futures that use AWT run in the
# AWT main thread, which is not accessible from Python.
from .jutil import execute_callable_in_main_thread, \
    execute_runnable_in_main_thread, execute_future_in_main_thread, \
    get_future_wrapper

# Exceptions
from .jutil import JavaError, JavaException, JVMNotFoundError

from ._javabridge import mac_enter_run_loop, mac_stop_run_loop, mac_run_loop_init
    
# References
from .jutil import create_jref, redeem_jref, create_and_lock_jref,\
     lock_jref, unlock_jref

# Don't expose: AtExit, get_nice_args,
# make_run_dictionary, run_in_main_thread, split_sig, unwrap_javascript,
# print_all_stack_traces


# Low-level API
from ._javabridge import JB_Env, JB_Object, JB_Class
# JNI helpers.
from ._javabridge import jni_enter, jni_exit, jvm_enter
