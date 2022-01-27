# -*- Encoding: utf-8 -*-
'''jutil.py - high-level interface to the JVM

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

'''
from __future__ import print_function


import gc
import inspect
import logging
import numpy as np
import os
import threading
import traceback
import re    
import subprocess
import sys
import uuid
from .locate import find_javahome
import javabridge
import weakref


# long and int are the same type in Py3
if sys.version_info[0] >= 3:
    long = int

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 0, 0):
    # basestring -> str and unicode -> str in Python 3
    basestring = str
    unicode = str


class JavaError(ValueError):
    '''An error caused by using the Javabridge incorrectly'''
    def __init__(self, message=None):
        super(JavaError,self).__init__(message)


class JVMNotFoundError(JavaError):
    '''Failed to find The Java Runtime Environment'''
    def __init__(self):
        super(JVMNotFoundError, self).__init__("Can't find the Java Virtual Machine")
        

class JavaException(Exception):
    '''Represents a Java exception thrown inside the JVM'''
    def __init__(self, throwable):
        '''Initialize by calling exception_occurred'''
        env = get_env()
        env.exception_describe()
        self.throwable = throwable
        try:
            if self.throwable is None:
                raise ValueError("Tried to create a JavaException but there was no current exception")
            #
            # The following has to be done by hand because the exception can't be
            # cleared at this point
            #
            klass = env.get_object_class(self.throwable)
            method_id = env.get_method_id(klass, 'getMessage', 
                                          '()Ljava/lang/String;')
            if method_id is not None:
                message = env.call_method(self.throwable, method_id)
                if message is not None:
                    message = env.get_string_utf(message)
                    super(JavaException, self).__init__(message)
        finally:
            env.exception_clear()


def _find_jvm_windows():
    # Look for JAVA_HOME and in the registry
    java_home = find_javahome()
    jvm_dir = None
    if java_home is not None:
        found_jvm = False
        for jre_home in (java_home, os.path.join(java_home, "jre")):
            jre_bin = os.path.join(jre_home, 'bin')
            for place_to_look in ('client','server'):
                jvm_dir = os.path.join(jre_bin, place_to_look)
                if os.path.isfile(os.path.join(jvm_dir, "jvm.dll")):
                    new_path = ';'.join((os.environ['PATH'], jvm_dir, jre_bin))
                    if isinstance(os.environ['PATH'], str) and \
                       isinstance(new_path, unicode) and \
                       sys.version_info < (3, 0, 0):
                        # Don't inadvertantly set an environment variable
                        # to unicode: causes subprocess.check_call to fail
                        # in Python 2
                        new_path = new_path.encode("utf-8")
                    os.environ['PATH'] = new_path
                    found_jvm = True
                    break
            if found_jvm:
                break
        if not found_jvm:
            jvm_dir = None
    return jvm_dir

def _find_mac_lib(library):
    jvm_dir = find_javahome()
    for extension in (".dylib", ".jnilib"):
        try:
            cmd = ["find", os.path.dirname(jvm_dir), "-name", library+extension]
            result = subprocess.check_output(cmd)
            if type(result) == bytes:
                lines = result.decode('utf-8').split("\n")
            else:
                lines = result.split("\n")
            if len(lines) > 0 and len(lines[0]) > 0:
                library_path = lines[0].strip()
                return library_path
        except Exception as e:
            logger.error("Failed to execute \"%s\" when searching for %s" % 
                         (cmd, library), exc_info=1)
    logger.error("Failed to find %s (jvmdir: %s" % (library, jvm_dir))
    return
    
def _find_jvm_mac():
    # Load libjvm.dylib and lib/jli/libjli.dylib if it exists
    jvm_dir = find_javahome()
    return jvm_dir

def _find_jvm():
    jvm_dir = None
    if sys.platform.startswith('win'):
        jvm_dir = _find_jvm_windows()
        if jvm_dir is None:
            raise JVMNotFoundError()
    elif sys.platform == 'darwin':
        jvm_dir = _find_jvm_mac()
    return jvm_dir


if sys.platform == "win32":
    # Need to fix up executable path to pick up jvm.dll
    # Also need path to JAVA_HOME\bin in order to find
    # msvcr...
    #
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + _find_jvm() + \
       os.pathsep + os.path.join(find_javahome(), "bin")
    try:
        os.add_dll_directory(_find_jvm())
        os.add_dll_directory(os.path.join(find_javahome(), "bin"))
    except AttributeError:
        logger.debug("DLL directories not added to environment, may cause problems in Python 3.8+")
    
elif sys.platform == "darwin":
    # Has side-effect of preloading dylibs
    _find_jvm_mac()
    
import javabridge._javabridge as _javabridge
__dead_event = threading.Event()
__kill = [False]
__main_thread_closures = []
__run_headless = False

RQCLS = "org/cellprofiler/runnablequeue/RunnableQueue"


class AtExit(object):
    '''AtExit runs a function as the main thread exits from the __main__ function
    
    We bind a reference to self to the main frame's locals. When
    the frame exits, "__del__" is called and the function runs. This is an
    alternative to "atexit" which only runs when all threads die.
    '''
    def __init__(self, fn):
        self.fn = fn
        stack = inspect.stack()
        for f, filename, lineno, module_name, code, index in stack:
            if (module_name == '<module>' and
                f.f_locals.get("__name__") == "__main__"):
                f.f_locals["X" + uuid.uuid4().hex] = self
                break
                
    def __del__(self):
        self.fn()
        
__start_thread = None        

class vm():
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def __enter__(self):
        start_vm(*self.args, **self.kwds)

    def __exit__(self, type, value, traceback):
        kill_vm()

def start_vm(args=None, class_path=None, max_heap_size=None, run_headless=False):
    '''Start the Java Virtual Machine.

    :param args: a list of strings, encoding arbitrary startup options
      for the VM. In particular, strings on the form
      ``"-D<name>=<value>"`` are used to set Java system
      properties. For other startup options, see `"The Invocation API"
      <http://docs.oracle.com/javase/6/docs/technotes/guides/jni/spec/invocation.html>`_. Options
      that set the class path (``-cp``, ``-classpath``, and
      ``-Djava.class.path``) are not allowed here; instead, use the
      `class_path` keyword argument.

    :param class_path: a list of strings constituting a class search
      path. Each string can be a directory, JAR archive, or ZIP
      archive. The default value, `None`, causes the class path in
      ``javabridge.JARS`` to be used.

    :param max_heap_size: string that specifies the maximum size, in
      bytes, of the memory allocation pool. This value must be a multiple
      of 1024 greater than 2MB. Append the letter k or K to indicate
      kilobytes, or m or M to indicate megabytes.

    :param run_headless: if true, set the ``java.awt.headless`` Java
      property. See `"Using Headless Mode in the Java SE Platform"
      <http://www.oracle.com/technetwork/articles/javase/headless-136834.html>`_.

    :throws: :py:exc:`javabridge.JVMNotFoundError`

    '''
    global __start_thread
    
    if args == None:
        args = []

    # Put this before the __vm check so the unit test can test it even
    # though the JVM is already started.
    if '-cp' in args or '-classpath' in args or any(arg.startswith('-Djava.class.path=') for arg in args):
        raise ValueError("Cannot set Java class path in the \"args\" argument to start_vm. Use the class_path keyword argument to javabridge.start_vm instead.")

    _find_jvm()
    
    if _javabridge.get_vm().is_active():
        return
    start_event = threading.Event()

    if class_path is None:
        class_path = javabridge.JARS
    if len(class_path) > 0:
        args.append("-Djava.class.path=" + os.pathsep.join(class_path))
    if max_heap_size:
        args.append("-Xmx" + max_heap_size)
    
    def start_thread(args=args, run_headless=run_headless):
        global __i_am_the_main_thread
        global __kill
        global __main_thread_closures
        global __run_headless
        
        args = list(args)
        if run_headless:
            __run_headless = True
            args = args + [r"-Djava.awt.headless=true"]

        logger.debug("Creating JVM object")
        _javabridge.set_thread_local("is_main_thread", True)
        vm = _javabridge.get_vm()
        #
        # We get local copies here and bind them in a closure to guarantee
        # that they exist past atexit.
        #
        kill = __kill
        main_thread_closures = __main_thread_closures
        try:
            if sys.platform == "darwin":
                logger.debug("Launching VM in non-python thread")
                library_path = _find_mac_lib("libjvm")
                libjli_path = _find_mac_lib("libjli")
                if library_path is None:
                    raise Exception("Javabridge failed to find JVM library")
                vm.create_mac(args, RQCLS, library_path, libjli_path)
                logger.debug("Attaching to VM in monitor thread")
                env = _javabridge.jb_attach()
            else:
                env = vm.create(args)
            init_context_class_loader()
        except:
            traceback.print_exc()
            logger.error("Failed to create Java VM")
            return
        finally:
            logger.debug("Signalling caller")
            start_event.set()
        while True:
            _javabridge.wait_for_wake_event()
            _javabridge.reap()
            while(len(main_thread_closures)):
                main_thread_closures.pop()()
            if kill[0]:
                break
        if sys.platform == "darwin":
            #
            # Torpedo the main thread RunnableQueue
            #
            rqcls = env.find_class(RQCLS)
            stop_id = env.get_static_method_id(rqcls, "stop", "()V")
            env.call_static_method(rqcls, stop_id)
            _javabridge.jb_detach()
        else:
            vm.destroy()
        __dead_event.set()
        
    __start_thread = threading.Thread(target=start_thread)
    __start_thread.setName("JVMMonitor")
    __start_thread.start()
    start_event.wait()
    if not _javabridge.get_vm().is_active():
        raise RuntimeError("Failed to start Java VM")
    attach()
    
def unwrap_javascript(o):
    '''Unwrap an object such as NativeJavaObject
    
    :param o: an object, possibly implementing org.mozilla.javascript.Wrapper
    
    :returns: result of calling the wrapper's unwrap method if a wrapper,
              otherwise the unboxed value for boxed types such as
              java.lang.Integer, and if not boxed, return the Java object.
    '''
    if is_instance_of(o, "org/mozilla/javascript/Wrapper"):
        o = call(o, "unwrap", "()Ljava/lang/Object;")
    if not isinstance(o, _javabridge.JB_Object):
        return o
    for class_name, method, signature in (
        ("java/lang/Boolean", "booleanValue", "()Z"),
        ("java/lang/Byte", "byteValue", "()B"),
        ("java/lang/Integer",  "intValue", "()I"),
        ("java/lang/Long", "longValue", "()L"),
        ("java/lang/Float", "floatValue", "()F"),
        ("java/lang/Double", "doubleValue", "()D")):
        if is_instance_of(o, class_name):
            return call(o, method, signature)
    return o
    
def run_script(script, bindings_in = {}, bindings_out = {}, 
               class_loader = None):
    '''Run a piece of JavaScript code.
    
    :param script: script to run
    :type script: string
    
    :param bindings_in: global variable names and values to assign to them.
    :type bindings_in: dict
                  
    :param bindings_out: a dictionary for returning variables. The
                         keys should be global variable names. After
                         the script has run, the values of these
                         variables will be assigned to the appropriate
                         value slots in the dictionary. For instance,
                         ``bindings_out = dict(foo=None)`` to get the
                         value for the "foo" variable on output.
                   
    :param class_loader: class loader for scripting context
    
    :returns: the object that is the result of the evaluation.
    '''
    context = static_call("org/mozilla/javascript/Context", "enter",
                          "()Lorg/mozilla/javascript/Context;")
    try :
        if class_loader is not None:
            call(context, "setApplicationClassLoader",
                 "(Ljava/lang/ClassLoader;)V",
                 class_loader)
        scope = make_instance("org/mozilla/javascript/ImporterTopLevel",
                              "(Lorg/mozilla/javascript/Context;)V",
                              context)
        for k, v in bindings_in.items():
            call(scope, "put", 
                 "(Ljava/lang/String;Lorg/mozilla/javascript/Scriptable;"
                 "Ljava/lang/Object;)V", k, scope, v)
        result = call(context, "evaluateString",
             "(Lorg/mozilla/javascript/Scriptable;"
             "Ljava/lang/String;"
             "Ljava/lang/String;"
             "I"
             "Ljava/lang/Object;)"
             "Ljava/lang/Object;", 
             scope, script, "<java-python-bridge>", 0, None)
        result = unwrap_javascript(result)
        for k in list(bindings_out):
            bindings_out[k] = unwrap_javascript(call(
                scope, "get",
                "(Ljava/lang/String;"
                "Lorg/mozilla/javascript/Scriptable;)"
                "Ljava/lang/Object;", k, scope))
    except JavaException as e:
        if is_instance_of(e.throwable, "org/mozilla/javascript/WrappedException"):
            raise JavaException(call(e.throwable, "unwrap", "()Ljava/lang/Object;"))
        else:
            raise
    finally:
        static_call("org/mozilla/javascript/Context", "exit", "()V")
    return result

def get_future_wrapper(o, fn_post_process=None):
    '''Wrap a ``java.util.concurrent.Future`` as a class.
    
    :param o: the object implementing the Future interface
    
    :param fn_post_process: a post-processing function to run on the object returned
                      from ``o.get()``. If you have ``Future<T>``, this can apply
                      the appropriate wrapper for ``T`` so you get back a
                      wrapped class of the appropriate type.
    '''
    class Future(object):
        def __init__(self):
            self.o = o
        run = make_method("run", "()V")
        cancel = make_method("cancel", "(Z)Z")
        raw_get = make_method(
            "get", "()Ljava/lang/Object;",
            "Waits if necessary for the computation to complete, and then retrieves its result.",
            fn_post_process=fn_post_process)
        isCancelled = make_method("isCancelled", "()Z")
        isDone = make_method("isDone", "()Z")
        if sys.platform != 'darwin':
            get = raw_get
        else:
            def get(self):
                '''Get the future's value after it has come done'''
                return mac_get_future_value(self)
    return Future()

def make_future_task(runnable_or_callable, 
                     result=None, fn_post_process=None):
    '''Make an instance of ``java.util.concurrent.FutureTask``.
    
    :param runnable_or_callable: either a
                           ``java.util.concurrent.Callable`` or a
                           ``java.lang.Runnable`` which is wrapped inside
                           the ``Future``

    :param result: if a ``Runnable``, this is the result that is returned
                   by ``Future.get``.
    
    :param fn_post_process: a postprocessing function run on the
                            result of ``Future.get``.


    Example: Making a future task from a Runnable:

    >>> future = javabridge.make_future_task(
            javabridge.run_script("new java.lang.Runnable() { run: function() {}};"),
            11)
    >>> future.run()
    >>> javabridge.call(future.get(), "intValue", "()I")
    11

    Example: Making a future task from a Callable:

    >>> callable = javabridge.run_script("""
            new java.util.concurrent.Callable() { 
                call: function() { return 2+2; }};""")
    >>> future = javabridge.make_future_task(callable, 
            fn_post_process=jutil.unwrap_javascript)
    >>> future.run()
    >>> future.get()
    4

    '''
    if is_instance_of(runnable_or_callable, 'java/util/concurrent/Callable'):
        o = make_instance('java/util/concurrent/FutureTask',
                          '(Ljava/util/concurrent/Callable;)V',
                          runnable_or_callable)
    else:
        o = make_instance('java/util/concurrent/FutureTask',
                          '(Ljava/lang/Runnable;Ljava/lang/Object;)V',
                          runnable_or_callable, result)
    return get_future_wrapper(o, fn_post_process)

def execute_runnable_in_main_thread(runnable, synchronous=False):
    '''Execute a runnable on the main thread
    
    :param runnable: a Java object implementing ``java.lang.Runnable``.
    
    :param synchronous: ``True`` if we should wait for the runnable to finish.
    
    Hint: to make a runnable using JavaScript::
    
        return new java.lang.Runnable() {
          run: function() {
            <do something here>
          }
        };

    '''
    if sys.platform == "darwin":
        # Assumes that RunnableQueue has been deployed on the main thread
        if synchronous:
            future = make_future_task(runnable)
            execute_future_in_main_thread(future)
        else:
            static_call(RQCLS, "enqueue", "(Ljava/lang/Runnable;)V",
                        runnable)
    else:
        run_in_main_thread(
            lambda: call(runnable, "run", "()V"), synchronous)
            
def execute_future_in_main_thread(future):
    '''Execute a Future in the main thread
    
    :param future: a future, wrapped by :py:func:`javabridge.get_future_wrapper`
    
    Synchronize with the return, running the event loop.

    '''
    # Portions of this were adapted from IPython/lib/inputhookwx.py
    #-----------------------------------------------------------------------------
    #  Copyright (C) 2008-2009  The IPython Development Team
    #
    #  Distributed under the terms of the BSD License.  The full license is in
    #  the file COPYING, distributed as par t of this software.
    #-----------------------------------------------------------------------------
    
    if sys.platform != "darwin":
        run_in_main_thread(future.run, True)
        return future.get()
    
    logger.debug("Enqueueing future on runnable queue")
    static_call(RQCLS, "enqueue", "(Ljava/lang/Runnable;)V", future.o)
    return mac_get_future_value(future)

def mac_get_future_value(future):
    '''Do special event loop processing to wait for future done on OS/X
    
    We need to run the event loop in OS/X while waiting for the
    future to come done to keep the UI event loop alive for message
    processing.
    '''
    if __run_headless:
        return future.raw_get()
    if sys.maxsize > 2**32:
        if _javabridge.mac_is_main_thread():
            #
            # Haven't figured out how to run a modal event loop
            # on OS/X - tried CFRunLoopInMode with 1/4 sec timeout and
            # it never returned.
            #
            raise NotImplementedError("No support for synchronizing futures in Python's startup thread on the OS/X in 64-bit mode.")
        return future.raw_get()
        
    import wx
    import time
    app = wx.GetApp()
    synchronize_without_event_loop = \
        (app is None and not __run_headless) or not _javabridge.mac_is_main_thread()
    if synchronize_without_event_loop:
        logger.debug("Synchronizing without event loop")
        #
        # There could be a deadlock between the GIL being taken
        # by the execution of Future.get() and AWT needing WX to
        # run the event loop. Therefore, we poll before getting.
        #
        while not future.isDone():
            logger.debug("Future is not done")
            time.sleep(.1)
        return future.raw_get()
    elif app is None:
        #
        # So sad - start some GUI if we need it.
        # 
        app = wx.PySimpleApp(True)
    if app.IsMainLoopRunning():
        evtloop = wx.EventLoop()
        logger.debug("Polling for future done within main loop")
        while not future.isDone():
            logger.debug("Future is not done")
            if evtloop.Pending():
                while evtloop.Pending():
                    logger.debug("Processing pending event")
                    evtloop.Dispatch()
            else:
                logger.debug("No pending wx event, run Dispatch anyway")
                evtloop.Dispatch()
            logger.debug("Sleeping")
            time.sleep(.1)
    else:
        logger.debug("Polling for future while running main loop")
        class EventLoopTimer(wx.Timer):
        
            def __init__(self, func):
                self.func = func
                wx.Timer.__init__(self)
        
            def Notify(self):
                self.func()
        
        class EventLoopRunner(object):
        
            def __init__(self, fn):
                self.fn = fn
                
            def Run(self, time):
                self.evtloop = wx.EventLoop()
                self.timer = EventLoopTimer(self.check_fn)
                self.timer.Start(time)
                self.evtloop.Run()
        
            def check_fn(self):
                if self.fn():
                    self.timer.Stop()
                    self.evtloop.Exit()
        event_loop_runner = EventLoopRunner(lambda: future.isDone())
        event_loop_runner.Run(time=10)
    logger.debug("Fetching future value")
    return future.raw_get()        
        
def execute_callable_in_main_thread(jcallable):
    '''Execute a callable on the main thread, returning its value
    
    :param jcallable: a Java object implementing ``java.util.concurrent.Callable``
    
    :returns: the result of evaluating the callable's "call" method in the
              main thread.
    
    Hint: to make a callable using JavaScript::
    
        var my_import_scope = new JavaImporter(java.util.concurrent.Callable);
        with (my_import_scope) {
            return new Callable() {
                call: function {
                    <do something that produces result>
                    return result;
                }
            };

    '''
    if sys.platform == "darwin":
        # Assumes that RunnableQueue has been deployed on the main thread
        future = make_instance(
            "java/util/concurrent/FutureTask",
            "(Ljava/util/concurrent/Callable;)V",
            jcallable)
        return execute_future_in_main_thread(future)
    else:
        return run_in_main_thread(
            lambda: call(jcallable, "call", "()Ljava/lang/Object;"), 
            True)
    

def run_in_main_thread(closure, synchronous):
    '''Run a closure in the main Java thread
    
    :param closure: a callable object (eg lambda : print "hello, world")
    :param synchronous: True to wait for completion of execution

    '''
    global __main_thread_closures
    if _javabridge.get_thread_local("is_main_thread", False):
        return closure()
    
    if synchronous:
        done_event = threading.Event()
        done_event.clear()
        result = [None]
        exception = [None]
        def synchronous_closure():
            try:
                result[0] = closure()
            except Exception as e:
                logger.exception("Caught exception when executing closure")
                exception[0] = e
            done_event.set()
        __main_thread_closures.append(synchronous_closure)
        _javabridge.set_wake_event()
        done_event.wait()
        if exception[0] is not None:
            raise exception[0]
        return result[0]
    else:
        __main_thread_closures.append(closure)
        _javabridge.set_wake_event()
    
def print_all_stack_traces():
    thread_map = static_call("java/lang/Thread","getAllStackTraces",
                             "()Ljava/util/Map;")
    stack_traces = call(thread_map, "values","()Ljava/util/Collection;")
    sta = call(stack_traces, "toArray","()[Ljava/lang/Object;")
    stal = get_env().get_object_array_elements(sta)
    for stak in stal:
        stakes = get_env().get_object_array_elements(stak)
        for stake in stakes:
            print(to_string(stake))
            
CLOSE_ALL_WINDOWS = """
        new java.lang.Runnable() { 
            run: function() {
                var all_frames = java.awt.Frame.getFrames();
                if (all_frames) {
                    for (idx in all_frames) {
                        try {
                            all_frames[idx].dispose();
                        } catch (err) {
                        }
                    }
                }
            }
        };"""

__awt_is_active = False
def activate_awt():
    '''
    Make a trivial AWT call in order to force AWT to initialize.
    
    '''
    global __awt_is_active
    if not __awt_is_active:
        execute_runnable_in_main_thread(run_script(
            """new java.lang.Runnable() {
                   run: function() {
                       java.awt.Color.BLACK.toString();
                   }
               };"""), True)
        __awt_is_active = True
        
def deactivate_awt():
    '''
    Close all AWT windows.
    
    '''
    global __awt_is_active
    if __awt_is_active:
        r = run_script(CLOSE_ALL_WINDOWS)
        execute_runnable_in_main_thread(r, True)
        __awt_is_active = False
def kill_vm():
    '''Kill the JVM. Once it is killed, it cannot be restarted.'''
    if not _javabridge.get_vm().is_active():
        return
    deactivate_awt()
    gc.collect()
    while _javabridge.get_thread_local("attach_count", 0) > 0:
        detach()
    __kill[0] = True
    _javabridge.set_wake_event()
    __dead_event.wait()
    __start_thread.join()
    
def attach():
    '''Attach to the VM, receiving the thread's environment'''
    attach_count = _javabridge.get_thread_local("attach_count", 0)
    _javabridge.set_thread_local("attach_count", attach_count + 1)
    if attach_count == 0:
        _javabridge.jb_attach()
        init_context_class_loader()
    return _javabridge.get_env()
    
def get_env():
    '''Return the thread's environment
    
    Note: call start_vm() and attach() before calling this
    '''
    return _javabridge.get_env()

def detach():
    '''Detach from the VM, releasing the thread's environment'''
    attach_count = _javabridge.get_thread_local("attach_count", 0)
    assert attach_count > 0
    attach_count -= 1
    _javabridge.set_thread_local("attach_count", attach_count)
    if attach_count > 0:
        return
    _javabridge.jb_detach()

def init_context_class_loader():
    '''Set the thread's context class loader to the system class loader
    
    When Java starts, as opposed to the JVM, the thread context class loader
    is set to the system class loader. When you start the JVM, the context
    class loader is null. This initializes the context class loader
    for a thread, if null.
    '''
    current_thread = static_call("java/lang/Thread", "currentThread",
                                 "()Ljava/lang/Thread;")
    loader = call(current_thread, "getContextClassLoader",
                  "()Ljava/lang/ClassLoader;")
    if loader is None:
        loader = static_call("java/lang/ClassLoader",
                             "getSystemClassLoader",
                             "()Ljava/lang/ClassLoader;")
        call(current_thread, "setContextClassLoader",
             "(Ljava/lang/ClassLoader;)V", loader)

def is_instance_of(o, class_name):
    '''Return True if object is instance of class
    
    :param o: object in question
    :param class_name: class in question. Use slash form: java/lang/Object
    
    Note: returns False if o is not a Java object.

    >>> javabridge.is_instance_of(javabridge.get_env().new_string_utf("Foo"), 'java/lang/String')
    True

    '''
    if not isinstance(o, _javabridge.JB_Object):
        return False
    env = get_env()
    klass = env.find_class(class_name)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    result = env.is_instance_of(o, klass)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    return result

def make_call(o, method_name, sig):
    '''Create a function that calls a method
    
    For repeated calls to a method on the same object, this method is
    faster than "call". The function returned takes raw Java objects
    which is significantly faster than "call" which parses the
    signature and casts arguments and return values.
    
    :param o: the object on which to make the call or a class name in slash form
    :param method_name: the name of the method to call
    :param sig: the function signature
    
    :returns: a function that can be called with the object to execute
              the method

    '''
    assert o is not None
    env = get_env()
    if isinstance(o, basestring):
        klass = env.find_class(o)
        bind = False
    else:
        klass = env.get_object_class(o)
        bind = True
    jexception = env.exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    method_id = env.get_method_id(klass, method_name, sig)
    jexception = env.exception_occurred()
    if method_id is None:
        if jexception is not None:
            raise JavaException(jexception)
        raise JavaError('Could not find method name = "%s" '
                        'with signature = "%s"' % (method_name, sig))
    if bind:
        def fn(*args):
            result = env.call_method(o, method_id, *args)
            x = env.exception_occurred()
            if x is not None:
                raise JavaException(x)
            return result
    else:
        def fn(o, *args):
            result = env.call_method(o, method_id, *args)
            x = env.exception_occurred()
            if x is not None:
                raise JavaException(x)
            return result
    return fn
    
def call(o, method_name, sig, *args):
    '''
    Call a method on an object
    
    :param o: object in question
    :param method_name: name of method on object's class
    :param sig: calling signature

    :returns: the result of the method call, converted to Python
              values when possible.

    >>> import javabridge
    >>> jstring = javabridge.get_env().new_string_utf("Hello, world")
    >>> javabridge.call(jstring, "charAt", "(I)C", 0)
    'H'

    '''
    env = get_env()
    fn = make_call(o, method_name, sig)
    args_sig = split_sig(sig[1:sig.find(')')])
    ret_sig = sig[sig.find(')')+1:]
    nice_args = get_nice_args(args, args_sig)
    result = fn(*nice_args)
    x = env.exception_occurred()
    if x is not None:
        raise JavaException(x)
    return get_nice_result(result, ret_sig)    

def make_static_call(class_name, method_name, sig):
    '''Create a function that performs a call of a static method
    
    make_static_call produces a function that is faster than static_call
    but is missing the niceties of preparing the argument and result casting.
    
    :param class_name: name of the class using slashes
    :param method_name: name of the method to call
    :param sig: the signature of the method.

    '''
    env = get_env()
    klass = env.find_class(class_name)
    if klass is None:
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    
    method_id = env.get_static_method_id(klass, method_name, sig)
    if method_id is None:
        raise JavaError('Could not find method name = %s '
                        'with signature = %s' %(method_name, sig))
    def fn(*args):
        result = env.call_static_method(klass, method_id, *args)
        jexception = env.exception_occurred() 
        if jexception is not None:
            raise JavaException(jexception)
        return result
    return fn

def static_call(class_name, method_name, sig, *args):
    '''Call a static method on a class
    
    :param class_name: name of the class, using slashes
    :param method_name: name of the static method
    :param sig: signature of the static method

    >>> javabridge.static_call("Ljava/lang/String;", "valueOf", "(I)Ljava/lang/String;", 123)
    u'123'

    '''
    env = get_env()
    fn = make_static_call(class_name, method_name, sig)
    args_sig = split_sig(sig[1:sig.find(')')])
    ret_sig = sig[sig.find(')')+1:]
    nice_args = get_nice_args(args, args_sig)
    result = fn(*nice_args)
    return get_nice_result(result, ret_sig)

def make_method(name, sig, doc='No documentation', fn_post_process=None):
    '''Return a class method for the given Java class. When called,
    the method expects to find its Java instance object in ``self.o``,
    which is where ``make_new`` puts it.

    :param name: method name
    :param sig: calling signature
    :param doc: doc string to be attached to the Python method
    :param fn_post_process: a function, such as a wrapper, that transforms
                            the method output into something more useable.
    
    '''
    
    def method(self, *args):
        assert isinstance(self.o, _javabridge.JB_Object)
        result = call(self.o, name, sig, *args)
        if fn_post_process is not None:
            result = fn_post_process(result)
        return result

    method.__doc__ = doc
    return method

def get_static_field(klass, name, sig):
    '''Get the value for a static field on a class
    
    :param klass: the class or string name of class
    :param name: the name of the field
    :param sig: the signature, typically, 'I' or 'Ljava/lang/String;'

    >>> javabridge.get_static_field("java/lang/Short", "MAX_VALUE", "S")
    32767

    '''
    env = get_env()
    if isinstance(klass, _javabridge.JB_Object):
        # Get the object's class
        klass = env.get_object_class(klass)
    elif not isinstance(klass, _javabridge.JB_Class):
        class_name = str(klass)
        klass = env.find_class(class_name)
        if klass is None:
            jexception = get_env().exception_occurred()
            raise JavaException(jexception)
    field_id = env.get_static_field_id(klass, name, sig)
    if field_id is None:
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    if sig == 'Z':
        return env.get_static_boolean_field(klass, field_id)
    elif sig == 'C':
        return env.get_static_char_field(klass, field_id)
    elif sig == 'B':
        return env.get_static_byte_field(klass, field_id)
    elif sig == 'S':
        return env.get_static_short_field(klass, field_id)
    elif sig == 'I':
        return env.get_static_int_field(klass, field_id)
    elif sig == 'J':
        return env.get_static_long_field(klass, field_id)
    elif sig == 'F':
        return env.get_static_float_field(klass, field_id)
    elif sig == 'D':
        return env.get_static_double_field(klass, field_id)
    else:
        return get_nice_result(env.get_static_object_field(klass, field_id),
                               sig)
        
def set_static_field(klass, name, sig, value):
    '''
    Set the value for a static field on a class
    
    :param klass: the class or string name of class
    :param name: the name of the field
    :param sig: the signature, typically, 'I' or 'Ljava/lang/String;'
    :param value: the value to set

    '''
    env = get_env()
    if isinstance(klass, _javabridge.JB_Object):
        # Get the object's class
        klass = env.get_object_class(klass)
    elif not isinstance(klass, _javabridge.JB_Class):
        class_name = str(klass)
        klass = env.find_class(class_name)
        if klass is None:
            jexception = get_env().exception_occurred()
            raise JavaException(jexception)
    field_id = env.get_static_field_id(klass, name, sig)
    if field_id is None:
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    if sig == 'Z':
        env.set_static_boolean_field(klass, field_id, value)
    elif sig == 'B':
        env.set_static_byte_field(klass, field_id, value)
    elif sig == 'C':
        assert len(str(value)) > 0
        env.set_static_char_field(klass, field_id, value)
    elif sig == 'S':
        env.set_static_short_field(klass, field_id, value)
    elif sig == 'I':
        env.set_static_int_field(klass, field_id, value)
    elif sig == 'J':
        env.set_static_long_field(klass, field_id, value)
    elif sig == 'F':
        env.set_static_float_field(klass, field_id, value)
    elif sig == 'D':
        env.set_static_double_field(klass, field_id, value)
    else:
        jobject = get_nice_arg(value, sig)
        env.set_static_object_field(klass, field_id, jobject)

def get_field(o, name, sig):
    '''Get the value for a field on an object
    
    :param o: the object
    :param name: the name of the field
    :param sig: the signature, typically 'I' or 'Ljava/lang/String;'

    '''
    assert isinstance(o, javabridge.JB_Object)
    env = get_env()
    klass = env.get_object_class(o)
    field_id = env.get_field_id(klass, name, sig)
    if field_id is None:
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    if sig == 'Z':
        return env.get_boolean_field(o, field_id)
    elif sig == 'C':
        return env.get_char_field(o, field_id)
    elif sig == 'B':
        return env.get_byte_field(o, field_id)
    elif sig == 'S':
        return env.get_short_field(o, field_id)
    elif sig == 'I':
        return env.get_int_field(o, field_id)
    elif sig == 'J':
        return env.get_long_field(o, field_id)
    elif sig == 'F':
        return env.get_float_field(o, field_id)
    elif sig == 'D':
        return env.get_double_field(o, field_id)
    else:
        return get_nice_result(env.get_object_field(o, field_id), sig)
        
def set_field(o, name, sig, value):
    '''Set the value for a field on an object
    
    :param o: the object
    :param name: the name of the field
    :param sig: the signature, typically 'I' or 'Ljava/lang/String;'
    :param value: the value to set
    '''
    assert isinstance(o, javabridge.JB_Object)
    env = get_env()
    klass = env.get_object_class(o)
    field_id = env.get_field_id(klass, name, sig)
    if field_id is None:
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    if sig == 'Z':
        env.set_boolean_field(o, field_id, value)
    elif sig == 'C':
        env.set_char_field(o, field_id, value)
    elif sig == 'B':
        env.set_byte_field(o, field_id, value)
    elif sig == 'C':
        assert len(str(value)) > 0
        env.set_char_field(o, field_id, value)
    elif sig == 'S':
        env.set_short_field(o, field_id, value)
    elif sig == 'I':
        env.set_int_field(o, field_id, value)
    elif sig == 'J':
        env.set_long_field(o, field_id, value)
    elif sig == 'F':
        env.set_float_field(o, field_id, value)
    elif sig == 'D':
        env.set_double_field(o, field_id, value)
    else:
        jobject = get_nice_arg(value, sig)
        env.set_object_field(o, field_id, jobject)
        
def split_sig(sig):
    '''Split a signature into its constituent arguments'''
    split = []
    orig_sig = sig
    while len(sig) > 0:
        match = re.match("\\[*(?:[ZBCSIJFD]|L[^;]+;)",sig)
        if match is None:
            raise ValueError("Invalid signature: %s"%orig_sig)
        split.append(match.group())
        sig=sig[match.end():]
    return split
        
def get_nice_args(args, sig):
    '''Convert arguments to Java types where appropriate
    
    returns a list of possibly converted arguments
    '''
    return [get_nice_arg(arg, subsig)
            for arg, subsig in zip(args, sig)]

def get_nice_arg(arg, sig):
    '''Convert an argument into a Java type when appropriate.

    '''
    env = get_env()
    is_java = (isinstance(arg, _javabridge.JB_Object) or
               isinstance(arg, _javabridge.JB_Class))
    if sig[0] == 'L' and not is_java:
        #
        # Check for the standard packing of java objects into class instances
        #
        if hasattr(arg, "o"):
            return arg.o
    #
    # If asking for an object, try converting basic types into Java-wraps
    # of Java basic types
    #
    if sig == 'Ljava/lang/Object;' and isinstance(arg, bool):
        return make_instance('java/lang/Boolean', '(Z)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, int):
        return make_instance('java/lang/Integer', '(I)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, long):
        return make_instance('java/lang/Long', '(J)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, float):
        return make_instance('java/lang/Double', '(D)V', arg)
    if (sig in ('Ljava/lang/String;','Ljava/lang/Object;') and not
         isinstance(arg, _javabridge.JB_Object)):
        if arg is None:
            return None
        else:
            if sys.version_info.major == 2:
                if isinstance(arg, str):
                    arg = arg.decode("utf-8")
        return env.new_string_utf(arg)
    if sig == 'Ljava/lang/Integer;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Integer', '(I)V', int(arg))
    if sig == 'Ljava/lang/Long' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Long', '(J)V', long(arg))
    if sig == 'Ljava/lang/Boolean;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Boolean', '(Z)V', bool(arg))
    
    if isinstance(arg, np.ndarray):
        if sig == '[Z':
            return env.make_boolean_array(np.ascontiguousarray(arg.flatten(), np.bool8))
        elif sig == '[B':
            return env.make_byte_array(np.ascontiguousarray(arg.flatten(), np.uint8))
        elif sig == '[S':
            return env.make_short_array(np.ascontiguousarray(arg.flatten(), np.int16))
        elif sig == '[I':
            return env.make_int_array(np.ascontiguousarray(arg.flatten(), np.int32))
        elif sig == '[J':
            return env.make_long_array(np.ascontiguousarray(arg.flatten(), np.int64))
        elif sig == '[F':
            return env.make_float_array(np.ascontiguousarray(arg.flatten(), np.float32))
        elif sig == '[D':
            return env.make_double_array(np.ascontiguousarray(arg.flatten(), np.float64))
    elif sig.startswith('L') and sig.endswith(';') and not is_java:
        #
        # Desperately try to make an instance of it with an integer constructor
        #
        if isinstance(arg, (int, long, bool)):
            return make_instance(sig[1:-1], '(I)V', int(arg))
        elif isinstance(arg, (str, unicode)):
            return make_instance(sig[1:-1], '(Ljava/lang/String;)V', arg)
    if sig.startswith('[L') and (not is_java) and hasattr(arg, '__iter__'):
        objs = [get_nice_arg(subarg, sig[1:]) for subarg in arg]
        k = env.find_class(sig[2:-1])
        a = env.make_object_array(len(objs), k)
        for i, obj in enumerate(objs):
            env.set_object_array_element(a, i, obj)
        return a
    return arg

def get_nice_result(result, sig):
    '''Convert a result that may be a java object into a string'''
    if result is None:
        return None
    env = get_env()
    if (sig == 'Ljava/lang/String;' or
        (sig == 'Ljava/lang/Object;' and 
         is_instance_of(result, "java/lang/String"))):
        return env.get_string_utf(result)
    if sig == 'Ljava/lang/Integer;':
        return call(result, 'intValue', '()I')
    if sig == 'Ljava/lang/Long':
        return call(result, 'longValue', '()J')
    if sig == 'Ljava/lang/Boolean;':
        return call(result, 'booleanValue', '()Z')
    if sig == '[B':
        # Convert a byte array into a numpy array
        return env.get_byte_array_elements(result)
    if isinstance(result, _javabridge.JB_Object):
        #
        # Do longhand to prevent recursion
        #
        rklass = env.get_object_class(result)
        m = env.get_method_id(rklass, 'getClass', '()Ljava/lang/Class;')
        rclass = env.call_method(result, m)
        rkklass = env.get_object_class(rclass)
        m = env.get_method_id(rkklass, 'isPrimitive', '()Z')
        is_primitive = env.call_method(rclass, m)
        if is_primitive:
            rc = get_class_wrapper(rclass, True)
            classname = rc.getCanonicalName()
            if classname == 'boolean':
                return to_string(result) == 'true'
            elif classname in ('int', 'byte', 'short', 'long'):
                return int(to_string(result))
            elif classname in ('float', 'double'):
                return float(to_string(result))
            elif classname == 'char':
                return to_string(result)
    return result

def to_string(jobject):
    '''
    Call the toString method on any object.

    :returns: the string representation of the object as a Python string

    >>> jstring = javabridge.get_env().new_string_utf("Hello, world")
    >>> jstring
    <Java object at 0x55116e0>
    >>> javabridge.to_string(jstring)
    u'Hello, world'

    '''
    env = get_env()
    if not isinstance(jobject, _javabridge.JB_Object):
        return str(jobject)
    return call(jobject, 'toString', '()Ljava/lang/String;')

def box(value, klass):
    '''Given a Java class and a value, convert the value to an instance of it
    
    value - value to be converted
    klass - return an object of this class, given the value.
    '''
    wclass = get_class_wrapper(klass, True)
    name = wclass.getCanonicalName()
    if wclass.isPrimitive():
        if name == 'boolean':
            return make_instance('java/lang/Boolean', "(Z)V", value)
        elif name == 'int':
            return make_instance('java/lang/Integer', "(I)V", value)
        elif name == 'byte':
            return make_instance('java/lang/Byte', "(B)V", value)
        elif name == 'short':
            return make_instance('java/lang/Short', "(S)V", value)
        elif name == 'long':
            return make_instance('java/lang/Long', "(J)V", value)
        elif name == 'float':
            return make_instance('java/lang/Float', "(F)V", value)
        elif name == 'double':
            return make_instance('java/lang/Double', "(D)V", value)
        elif name == 'char':
            return make_instance('java/lang/Character', "(C)V", value)
        else:
            raise NotImplementedError("Boxing %s is not implemented" % name)
    sig = "L%s;" % wclass.getCanonicalName().replace(".", "/")
    return get_nice_arg(value, sig)

def get_collection_wrapper(collection, fn_wrapper=None):
    '''Return a wrapper of ``java.util.Collection``
    
    :param collection: an object that implements
                 ``java.util.Collection``. If the object implements the
                 list interface, that is wrapped as well
    
    :param fn_wrapper: if defined, a function that wraps a Java object
    
    The returned value is a Python object, duck-typed as a sequence. Items
    can be retrieved by index or by slicing. You can also iterate through
    the collection::
    
        for o in get_collection_wrapper(jobject):
            # do something
        
    If you supply a function wrapper, indexing and iteration operations
    will return the result of calling the function wrapper on the objects
    in the collection::
    
        for d in get_collection_wrapper(list_of_hashmaps, get_map_wrapper):
            # a map wrapper on the hashmap is returned
            print(d["Foo"])

    '''
    class Collection(object):
        def __init__(self):
            self.o = collection
            
        add = make_method("add", "(Ljava/lang/Object;)Z")
        addAll = make_method("addAll", "(Ljava/util/Collection;)Z")
        clear = make_method("clear", "()V")
        contains = make_method("contains", "(Ljava/lang/Object;)Z")
        containsAll = make_method("containsAll", "(Ljava/util/Collection;)Z")
        isEmpty = make_method("isEmpty", "()Z")
        iterator = make_method("iterator", "()Ljava/util/Iterator;")
        remove = make_method("remove", "(Ljava/lang/Object;)Z")
        removeAll = make_method("removeAll", "(Ljava/util/Collection;)Z")
        retainAll = make_method("retainAll", "(Ljava/util/Collection;)Z")
        size = make_method("size", "()I")
        toArray = make_method(
            "toArray", "()[Ljava/lang/Object;",
            fn_post_process=get_env().get_object_array_elements)
        toArrayC = make_method("toArray", "([Ljava/lang/Object;)[Ljava/lang/Object;")
        
        def __len__(self):
            return self.size()
        
        def __iter__(self):
            return iterate_collection(self.o, fn_wrapper = fn_wrapper)
        
        def __contains__(self, item):
            return self.contains(item)
        
        @staticmethod
        def is_collection(x):
            return (hasattr(x, "o") and 
                    is_instance_of(x.o, "java/util/Collection"))
            
        def __add__(self, items):
            klass = call(self.o, "getClass", "()Ljava/lang/Class;")
            copy = get_collection_wrapper(
                call(klass, "newInstance", "()Ljava/lang/Object;"),
                fn_wrapper = fn_wrapper)
            copy.addAll(self.o)
            if self.is_collection(items):
                copy.addAll(items.o)
            else:
                for item in items:
                    copy.add(item)
            return copy
            
        def __iadd__(self, items):
            if self.is_collection(items):
                self.addAll(items)
            else:
                for item in items:
                    self.add(item)
            return self
        
        if is_instance_of(collection, 'java/util/List'):
            addI = make_method("add", "(ILjava/lang/Object;)V")
            addAllI = make_method("addAll", "(ILjava/util/Collection;)Z")
            indexOf = make_method("indexOf", "(Ljava/lang/Object;)I")
            lastIndexOf = make_method("lastIndexOf", "(Ljava/lang/Object;)I")
            removeI = make_method("remove", "(I)Ljava/lang/Object;", 
                                  fn_post_process=fn_wrapper)
            get = make_method("get", "(I)Ljava/lang/Object;", 
                              fn_post_process=fn_wrapper)
            set = make_method("set", "(ILjava/lang/Object;)Ljava/lang/Object;",
                              fn_post_process=fn_wrapper)
            subList = make_method(
                "subList",
                "(II)Ljava/util/List;",
                fn_post_process=lambda x: get_collection_wrapper(x, fn_wrapper))
            
            def __normalize_idx(self, idx, none_value):
                if idx is None:
                    return none_value
                elif idx < 0:
                    return max(0, self.size()+idx)
                elif idx > self.size():
                    return self.size()
                return idx
            
            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    start = self.__normalize_idx(idx.start, 0)
                    stop = self.__normalize_idx(idx.stop, self.size())
                    if idx.step is None or idx.step == 1:
                        return self.subList(start, stop)
                    return [self[i] for i in range(start, stop, idx.step)]
                return self.get(self.__normalize_idx(idx, 0))
            
            def __setitem__(self, idx, value):
                self.set(idx, value)
                
            def __delitem__(self, idx):
                self.removeI(idx)
            
    return Collection()

array_list_add_method_id = None
def make_list(elements=[]):
    '''Make a wrapped ``java.util.ArrayList``.

    The ``ArrayList`` will contain the specified elements, if any.

    :param elements: the elements to put in the ``ArrayList``.

    Examples::
        >>> mylist = make_list(["Hello", "World", 2])
        >>> print("\\n".join([to_string(o) for o in mylist]))
        Hello
        World
        2
        >>> print("%s, %s." % (mylist[0], mylist[1].lower()))
        Hello, world.
        >>> get_class_wrapper(mylist.o)
        java.util.ArrayList
        public boolean java.util.ArrayList.add(java.lang.Object)
        public void java.util.ArrayList.add(int,java.lang.Object)
        ...
    '''
    global array_list_add_method_id
    
    a = get_collection_wrapper(make_instance("java/util/ArrayList", "()V"))
    env = get_env()
    if len(elements) > 0:
        if array_list_add_method_id is None:
            array_list_class = env.find_class("java/util/ArrayList")
            array_list_add_method_id = env.get_method_id(
                array_list_class, "add", "(Ljava/lang/Object;)Z")
        for element in elements:
            if not isinstance(element, _javabridge.JB_Object):
                element = get_nice_arg(element, "Ljava/lang/Object;")
            env.call_method(a.o, array_list_add_method_id, element)
            x = env.exception_occurred()
            if x is not None:
                raise JavaException(x)
    return a

def get_dictionary_wrapper(dictionary):
    '''Return a wrapper of ``java.util.Dictionary``.

    :param dictionary: Java object that implements the ``java.util.Dictionary`` interface.
    :returns: a Python instance that wraps the Java dictionary.

    >>> jproperties = javabridge.static_call("java/lang/System", "getProperties", "()Ljava/util/Properties;")
    >>> properties = javabridge.get_dictionary_wrapper(jproperties)
    >>> properties.size()
    56

    '''
    env = get_env()
    class Dictionary(object):
        def __init__(self):
            self.o = dictionary
        size = make_method('size', '()I',
                           'Returns the number of entries in this dictionary')
        isEmpty = make_method('isEmpty', '()Z',
                              'Tests if this dictionary has no entries')
        keys = make_method('keys', '()Ljava/util/Enumeration;',
                           'Returns an enumeration of keys in this dictionary')
        elements = make_method('elements',
                               '()Ljava/util/Enumeration;',
                               'Returns an enumeration of elements in this dictionary')
        get = make_method('get',
                          '(Ljava/lang/Object;)Ljava/lang/Object;',
                          'Return the value associated with a key or None if no value')
        put = make_method('put',
                          '(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;',
                          'Associate a value with a key in the dictionary')
    return Dictionary()

def get_map_wrapper(o):
    '''Return a wrapper of ``java.util.Map``
    
    :param o: a Java object that implements the ``java.util.Map`` interface
    
    Returns a Python object duck typed as a dictionary.
    You can fetch values from the Java object using the Python array syntax::
    
        > d = get_map_wrapper(jmap)
        > d["Foo"] = "Bar"
        > print(d["Foo"])
        Bar
    '''
    assert is_instance_of(o, 'java/util/Map')
    class Map(object):
        def __init__(self):
            self.o = o
        clear = make_method("clear", "()V")
        containsKey = make_method("containsKey", "(Ljava/lang/Object;)Z")
        containsValue = make_method("containsValue", "(Ljava/lang/Object;)Z")
        entrySet = make_method("entrySet", "()Ljava/util/Set;")
        get = make_method("get", "(Ljava/lang/Object;)Ljava/lang/Object;")
        isEmpty = make_method("isEmpty", "()Z")
        keySet = make_method("keySet", "()Ljava/util/Set;")
        put = make_method(
            "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;")
        putAll = make_method("putAll", "(Ljava/util/Map;)V")
        remove = make_method("remove", "(Ljava/lang/Object;)Ljava/lang/Object;")
        size = make_method("size", "()I")
        values = make_method("values", "()Ljava/util/Collection;")
        
        def __len__(self):
            return self.size()
        
        def __getitem__(self, key):
            return self.get(key)
        
        def __setitem__(self, key, value):
            self.put(key, value)
            
        def __iter__(self):
            return iterate_collection(self.keySet())
        
        def keys(self):
            return tuple(iterate_collection(self.keySet(self)))
        
    return Map()

def make_map(**kwargs):
    '''Create a wrapped ``java.util.HashMap`` from arbitrary keyword arguments.

    Example::
    
        > d = make_map(foo="Bar")
        > print(d["foo"])
        Bar
        > get_class_wrapper(d.o)
        java.util.HashMap
        public java.lang.Object java.util.HashMap.get(java.lang.Object)
        public java.lang.Object java.util.HashMap.put(java.lang.Object,java.lang.Object)
    '''
    hashmap = get_map_wrapper(make_instance('java/util/HashMap', "()V"))
    for k, v in kwargs.items():
        hashmap[k] = v
    return hashmap

def jdictionary_to_string_dictionary(hashtable):
    '''Convert a Java dictionary to a Python dictionary.
    
    Convert each key and value in the Java dictionary to a string and
    construct a Python dictionary from the result.

    :param hashtable: Java object that implements the ``java.util.Hashtable`` interface.
    :returns: a Python ``dict`` with strings as keys and values

    >>> jproperties = javabridge.static_call("java/lang/System", "getProperties", "()Ljava/util/Properties;")
    >>> properties = javabridge.jdictionary_to_string_dictionary(jproperties)
    >>> properties['java.specification.vendor']
    'Sun Microsystems Inc.'

    '''
    jhashtable = get_dictionary_wrapper(hashtable)
    jkeys = jhashtable.keys()
    keys = jenumeration_to_string_list(jkeys)
    result = {}
    for key in keys:
        result[key] = to_string(jhashtable.get(key))
    return result

def get_enumeration_wrapper(enumeration):
    '''Return a wrapper of java.util.Enumeration
    
    Given a JB_Object that implements java.util.Enumeration,
    return an object that wraps the class methods.

    >>> jproperties = javabridge.static_call("java/lang/System", "getProperties", "()Ljava/util/Properties;")
    >>> keys = javabridge.call(jproperties, "keys", "()Ljava/util/Enumeration;")
    >>> enum = javabridge.get_enumeration_wrapper(keys)
    >>> while enum.hasMoreElements():
    ...     if javabridge.to_string(enum.nextElement()) == 'java.vm.name':
    ...         print("Has java.vm.name")
    ... 
    Has java.vm.name

    '''
    env = get_env()
    class Enumeration(object):
        def __init__(self):
            '''Call the init method with the JB_Object'''
            self.o = enumeration
        hasMoreElements = make_method('hasMoreElements', '()Z',
                                      'Return true if the enumeration has more elements to retrieve')
        nextElement = make_method('nextElement', 
                                  '()Ljava/lang/Object;')
    return Enumeration()

iterator_has_next_id = None
iterator_next_id = None
def iterate_java(iterator, fn_wrapper=None):
    '''Make a Python iterator for a Java iterator
    
    >>> jiterator = javabridge.run_script("""var al = new java.util.ArrayList(); al.add("Foo"); al.add("Bar"); al.iterator()""")
    >>> [x for x in javabridge.iterate_java(jiterator)]
    [u'Foo', u'Bar']

    '''
    global iterator_has_next_id, iterator_next_id
    env = get_env()
    iterator_class = env.find_class("java/util/Iterator")
    if not isinstance(iterator, _javabridge.JB_Object):
        raise JavaError("%s is not a Javabridge JB_Object" % repr(iterator))
    if not env.is_instance_of(iterator, iterator_class):
        raise JavaError("%s does not implement the java.util.Iterator interface" %
                        get_class_wrapper(iterator).getCanonicalName())
    if iterator_has_next_id is None:
        iterator_has_next_id = env.get_method_id(iterator_class, "hasNext", "()Z")
        iterator_next_id = env.get_method_id(iterator_class, "next", "()Ljava/lang/Object;")
    while(True):
        result = env.call_method(iterator, iterator_has_next_id)
        x = env.exception_occurred()
        if x is not None:
            raise JavaException(x)
        if not result:
            break;
        item = env.call_method(iterator, iterator_next_id)
        x = env.exception_occurred()
        if x is not None:
            raise JavaException(x)
        yield item if fn_wrapper is None else fn_wrapper(item)
        
def iterate_collection(c, fn_wrapper=None):
    '''
    Make a Python iterator over the elements of a Java collection

    >>> al = javabridge.run_script("""var al = new java.util.ArrayList(); al.add("Foo"); al.add("Bar"); al;""")
    >>> [x for x in javabridge.iterate_java(al)]
    [u'Foo', u'Bar']

    '''
    return iterate_java(call(c, "iterator", "()Ljava/util/Iterator;"),
                        fn_wrapper=fn_wrapper)
        
def jenumeration_to_string_list(enumeration):
    '''Convert a Java enumeration to a Python list of strings
    
    Convert each element in an enumeration to a string and return them
    as a Python list.

    >>> jproperties = javabridge.static_call("java/lang/System", "getProperties", "()Ljava/util/Properties;")
    >>> keys = javabridge.call(jproperties, "keys", "()Ljava/util/Enumeration;")
    >>> 'java.vm.name' in javabridge.jenumeration_to_string_list(keys)
    True

    '''
    jenumeration = get_enumeration_wrapper(enumeration)
    result = []
    while jenumeration.hasMoreElements():
        result.append(to_string(jenumeration.nextElement()))
    return result

def make_new(class_name, sig):
    '''
    Make a function that creates a new instance of the class. When
    called, the function does not return the new instance, but stores
    it at ``self.o``.
    
    A typical init function looks like this::

        new_fn = make_new("java/lang/Integer", '(I)V')
        def __init__(self, i):
            new_fn(i)

    '''
    def constructor(self, *args):
        self.o = make_instance(class_name, sig, *args)
    return constructor

def make_instance(class_name, sig, *args):
    '''Create an instance of a class
    
    :param class_name: name of class in foo/bar/Baz form (not foo.bar.Baz)
    :param sig: signature of constructor
    :param args: arguments to constructor

    >>> javabridge.make_instance("java/lang/Integer", "(I)V", 42)
    <Java object at 0x55116dc>

    '''
    args_sig = split_sig(sig[1:sig.find(')')])
    klass = get_env().find_class(class_name)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    method_id = get_env().get_method_id(klass, '<init>', sig)
    jexception = get_env().exception_occurred()
    if method_id is None:
        if jexception is None:
            raise JavaError('Could not find constructor '
                            'with signature = "%s' % sig)
        else:
            raise JavaException(jexception)
    result = get_env().new_object(klass, method_id, 
                                  *get_nice_args(args, args_sig))
    jexception = get_env().exception_occurred() 
    if jexception is not None:
        raise JavaException(jexception)
    return result

def class_for_name(classname, ldr="system"):
    '''Return a ``java.lang.Class`` for the given name.
    
    :param classname: the class name in dotted form, e.g. "java.lang.String"

    '''
    if ldr == "system":
        ldr = static_call('java/lang/ClassLoader', 'getSystemClassLoader',
                          '()Ljava/lang/ClassLoader;')
    return static_call('java/lang/Class', 'forName', 
                       '(Ljava/lang/String;ZLjava/lang/ClassLoader;)'
                       'Ljava/lang/Class;', 
                       classname, True, ldr)

def get_class_wrapper(obj, is_class = False):
    '''Return a wrapper for an object's class (e.g., for
    reflection). The returned wrapper class will have the following
    methods:

    ``getAnnotation()``
       ``java.lang.annotation.Annotation``
    ``getAnnotations()``
       array of ``java.lang.annotation.Annotation``
    ``getCanonicalName()``
       ``java.lang.String``
    ``getClasses()``
       array of ``java.lang.Class``
    ``getContructor(signature)``
       ``java.lang.reflect.Constructor``
    ``getFields()``
       array of ``java.lang.reflect.Field``
    ``getField(field_name)``
       ``java.lang.reflect.Field``
    ``getMethods()``
       array of ``java.lang.reflect.Method``
    ``getMethod(method_name)``
       ``java.lang.reflect.Method``
    ``cast(class)``
       object
    ``isPrimitive()``
       boolean
    ``newInstance()``
       object
 
    '''
    if is_class:
        class_object = obj
    elif isinstance(obj, (str, unicode)):
        class_object = class_for_name(obj)
    else:
        class_object = call(obj, 'getClass','()Ljava/lang/Class;')
    class Klass(object):
        def __init__(self):
            self.o = class_object
        getAnnotation = make_method('getAnnotation',
                                    '(Ljava/lang/Class;)Ljava/lang/annotation/Annotation;',
                                    "Returns this element's annotation if present")
        getAnnotations = make_method('getAnnotations',
                                     '()[Ljava/lang/annotation/Annotation;')
        getCanonicalName = make_method('getCanonicalName',
                                       '()Ljava/lang/String;',
                                       'Returns the canonical name of the class')
        getClasses = make_method('getClasses','()[Ljava/lang/Class;',
                                 'Returns an array containing Class objects representing all the public classes and interfaces that are members of the class represented by this Class object.')
        getConstructor = make_method(
            'getConstructor', 
            '([Ljava/lang/Class;)Ljava/lang/reflect/Constructor;',
            'Return a constructor with the given signature')
        getConstructors = make_method('getConstructors','()[Ljava/lang/reflect/Constructor;')
        getFields = make_method('getFields','()[Ljava/lang/reflect/Field;')
        getField = make_method('getField','(Ljava/lang/String;)Ljava/lang/reflect/Field;')
        getMethod = make_method('getMethod','(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;')
        getMethods = make_method('getMethods','()[Ljava/lang/reflect/Method;')
        cast = make_method('cast', '(Ljava/lang/Object;)Ljava/lang/Object;',
                           'Throw an exception if object is not castable to this class')
        isPrimitive = make_method('isPrimitive', '()Z',
                                  'Return True if the class is a primitive such as boolean or int')
        newInstance = make_method('newInstance', '()Ljava/lang/Object;',
                                  'Make a new instance of the object with the default constructor')
        def __repr__(self):
            methods = get_env().get_object_array_elements(self.getMethods())
            return "%s\n%s" % (
                self.getCanonicalName(), 
                "\n".join([to_string(x) for x in methods]))

    return Klass()

MOD_ABSTRACT  = 'ABSTRACT'
MOD_FINAL = 'FINAL'
MOD_INTERFACE = 'INTERFACE'
MOD_NATIVE = 'NATIVE'
MOD_PRIVATE = 'PRIVATE'
MOD_PROTECTED = 'PROTECTED'
MOD_PUBLIC = 'PUBLIC'
MOD_STATIC = 'STATIC'
MOD_STRICT = 'STRICT'
MOD_SYCHRONIZED = 'SYNCHRONIZED'
MOD_TRANSIENT = 'TRANSIENT'
MOD_VOLATILE = 'VOLATILE'
MOD_ALL = [MOD_ABSTRACT, MOD_FINAL, MOD_INTERFACE, MOD_NATIVE,
           MOD_PRIVATE, MOD_PROTECTED, MOD_PUBLIC, MOD_STATIC,
           MOD_STRICT, MOD_SYCHRONIZED, MOD_TRANSIENT, MOD_VOLATILE]

def get_modifier_flags(modifier_flags):
    '''Parse out the modifiers from the modifier flags from getModifiers'''
    result = []
    for mod in MOD_ALL:
        if modifier_flags & get_static_field('java/lang/reflect/Modifier',
                                             mod, 'I'):
            result.append(mod)
    return result

def get_field_wrapper(field):
    '''
    Return a wrapper for the java.lang.reflect.Field class. The
    returned wrapper class will have the following methods:

    ``getAnnotation()``
       java.lang.annotation.Annotation
    ``getBoolean()``
       bool
    ``getByte``
       byte
    ``getChar``
       char
    ``getDouble``
       double
    ``getFloat``
       float
    ``getInt``
       int
    ``getShort``
       short
    ``getLong``
       long
    ``getDeclaredAnnotations()``
       array of java.lang.annotation.Annotation
    ``getGenericType``
       java.lang.reflect.Type
    ``getModifiers()``
       Python list of strings indicating the modifier flags
    ``getName()``
       java.lang.String()
    ``getType()``
       java.lang.Class()
    ``set(object, object)``
       void
    ``setBoolean(bool)``
       void
    ``setByte(byte)``
       void
    ``setChar(char)``
       void
    ``setDouble(double)``
       void
    ``setFloat(float)``
       void
    ``setInt(int)``
       void
    ``setShort(short)``
       void
    ``setLong(long)``
       void

    '''
    class Field(object):
        def __init__(self):
            self.o = field
            
        get = make_method('get', '(Ljava/lang/Object;)Ljava/lang/Object;',
                          'Returns the value of the field represented by this '
                          'Field, on the specified object.')
        def getAnnotation(self, annotation_class):
            """Returns this element's annotation for the specified type
            
            annotation_class - find annotations of this class
            
            returns the annotation or None if not annotated"""
            
            if isinstance(annotation_class, (str, unicode)):
                annotation_class = class_for_name(annotation_class)
            return call(self.o, 'getAnnotation', 
                        '(Ljava/lang/Class;)Ljava/lang/annotation/Annotation;',
                        annotation_class)
        
        getBoolean = make_method('getBoolean', '(Ljava/lang/Object;)Z',
                                 'Read a boolean field from an object')
        getByte = make_method('getByte', '(Ljava/lang/Object;)B',
                              'Read a byte field from an object')
        getChar = make_method('getChar', '(Ljava/lang/Object;)C')
        getDouble = make_method('getDouble', '(Ljava/lang/Object;)D')
        getFloat = make_method('getFloat', '(Ljava/lang/Object;)F')
        getInt = make_method('getInt', '(Ljava/lang/Object;)I')
        getShort = make_method('getShort', '(Ljava/lang/Object;)S')
        getLong = make_method('getLong', '(Ljava/lang/Object;)J')
        getDeclaredAnnotations = make_method(
            'getDeclaredAnnotations',
            '()[Ljava/lang/annotation/Annotation;')
        getGenericType = make_method('getGenericType', 
                                     '()Ljava/lang/reflect/Type;')
        def getModifiers(self):
            return get_modifier_flags(call(self.o, 'getModifiers','()I'))
        getName = make_method('getName', '()Ljava/lang/String;')
        
        getType = make_method('getType', '()Ljava/lang/Class;')
        set = make_method('set', '(Ljava/lang/Object;Ljava/lang/Object;)V')
        setBoolean = make_method('setBoolean', '(Ljava/lang/Object;Z)V',
                                 'Set a boolean field in an object')
        setByte = make_method('setByte', '(Ljava/lang/Object;B)V',
                              'Set a byte field in an object')
        setChar = make_method('setChar', '(Ljava/lang/Object;C)V')
        setDouble = make_method('setDouble', '(Ljava/lang/Object;D)V')
        setFloat = make_method('setFloat', '(Ljava/lang/Object;F)V')
        setInt = make_method('setInt', '(Ljava/lang/Object;I)V')
        setShort = make_method('setShort', '(Ljava/lang/Object;S)V')
        setLong = make_method('setLong', '(Ljava/lang/Object;J)V')
    return Field()

def get_constructor_wrapper(obj):
    '''
    Get a wrapper for calling methods on the constructor object. The
    wraper class will have the following methods:

    ``getParameterTypes()``
       array of ``java.lang.Class``
    ``getName()``
       ``java.lang.String``
    ``newInstance(array of objects)``
       object
    ``getAnnotation()``
       ``java.lang.annotation.Annotation``
    ``getModifiers()``
       Python list of strings indicating the modifier flags

    '''
    class Constructor(object):
        def __init__(self):
            self.o = obj
            
        getParameterTypes = make_method('getParameterTypes',
                                        '()[Ljava/lang/Class;',
                                        'Get the types of the constructor parameters')
        getName = make_method('getName', '()Ljava/lang/String;')
        newInstance = make_method('newInstance',
                                  '([Ljava/lang/Object;)Ljava/lang/Object;')
        getAnnotation = make_method('getAnnotation', 
                                    '()Ljava/lang/annotation/Annotation;')
        getModifiers = make_method('getModifiers', '()I')
    return Constructor()
        
def get_method_wrapper(obj):
    '''
    Get a wrapper for calling methods on the method object. The
    wrapper class will have the following methods:

    ``getParameterTypes()``
       array of ``java.lang.Class``
    ``getName()``
       ``java.lang.String``
    ``invoke(this, arguments)``
       object
    ``getAnnotation()``
       ``java.lang.annotation.Annotation``
    ``getModifiers()``
       Python list of strings indicating the modifier flags

    '''
    class Method(object):
        def __init__(self):
            self.o = obj
            
        getParameterTypes = make_method('getParameterTypes',
                                        '()[Ljava/lang/Class;',
                                        'Get the types of the constructor parameters')
        getName = make_method('getName', '()Ljava/lang/String;')
        invoke = make_method('invoke',
                             '(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;')
        getAnnotation = make_method('getAnnotation', 
                                    '()Ljava/lang/annotation/Annotation;')
        getModifiers = make_method('getModifiers', '()I')
    return Method()
        
def make_run_dictionary(jobject):
    '''Support function for Py_RunString - jobject -> globals / locals
    
    jobject_address - address of a Java Map of string to object
    '''
    result = {}
    jmap = javabridge.JWrapper(jobject)
    jentry_set = jmap.entrySet()
    jentry_set_iterator = jentry_set.iterator()
    while jentry_set_iterator.hasNext():
        entry = jentry_set_iterator.next()
        key, value = [o if not isinstance(o, javabridge.JWrapper) else o.o
                      for o in (entry.getKey(), entry.getValue())]
        result[to_string(key)] = value
    return result

__weakrefdict = weakref.WeakValueDictionary()
__strongrefdict = {}

class _JRef(object):
    '''A reference to some Python value for Java scripting
    
    A Java script executed using org.cellprofiler.javabridge.CPython.exec
    might want to maintain and refer to objects and values. This class
    wraps the value so that it can be referred to later.
    '''
    def __init__(self, value):
        self.__value = value
        
    def __call__(self):
        return self.__value
    
def create_jref(value):
    '''Create a weak reference to a Python value
    
    This routine lets the Java method, CPython.exec(), create weak references
    which can be redeemed by passing a token to redeem_weak_reference. The
    routine returns a reference to the value as well and this reference must
    be stored somewhere (e.g. a global value in a module) for the token to
    be valid upon redemption.
    
    :param value: The value to be redeemed
    
    :returns: a tuple of a string token and a reference that must be maintained
              in order to retrieve it later
    '''
    ref = _JRef(value)
    ref_id = uuid.uuid4().hex
    __weakrefdict[ref_id] = ref
    return ref_id, ref

def create_and_lock_jref(value):
    '''Create and lock a value in one step
    
    :param value: the value to be redeemed
    :returns: a ref_id that can be used to redeem the value and to unlock it.
    '''
    ref_id, ref = create_jref(value)
    lock_jref(ref_id)
    return ref_id

def redeem_jref(ref_id):
    '''Redeem a reference created using create_jref
    
    Raises KeyError if the reference could not be found, possibly because
    someone didn't hold onto it
    
    :param ref_id: the token returned by create_jref for the reference
    
    :returns: the value
    '''
    return __weakrefdict[ref_id]()

def lock_jref(ref_id):
    '''Lock a reference to maintain it across CPython.exec() invocations
    
    Lock a reference into memory until unlock_jref is called. lock_jref()
    can be called repeatedly on the same reference and the reference will
    be held until an equal number of unlock_jref() calls have been made.
    
    :param ref_id: the ID returned from create_ref
    '''
    if ref_id not in __strongrefdict:
        __strongrefdict[ref_id] = []
    __strongrefdict[ref_id].append(__weakrefdict[ref_id])
    
def unlock_jref(ref_id):
    '''Unlock a reference locked by lock_jref

    Unlock and potentially dereference a reference locked by lock_jref()
    
    :param ref_id: the ID used to lock the reference
    '''
    refs = __strongrefdict[ref_id]
    if len(refs) == 1:
        del __strongrefdict[ref_id]
    else:
        refs.pop()
        
if __name__=="__main__":
    import wx
    app = wx.PySimpleApp(False)
    frame = wx.Frame(None)
    frame.Sizer = wx.BoxSizer(wx.HORIZONTAL)
    start_button = wx.Button(frame, label="Start VM")
    frame.Sizer.Add(start_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    def fn_start(event):
        start_vm([])
        start_button.Enable(False)
    start_button.Bind(wx.EVT_BUTTON, fn_start)
    
    launch_button = wx.Button(frame, label="Launch AWT frame")
    frame.Sizer.Add(launch_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    
    def fn_launch_frame(event):
        execute_runnable_in_main_thread(run_script("""
        new java.lang.Runnable() {
            run: function() {
                with(JavaImporter(java.awt.Frame)) Frame().setVisible(true);
            }
        };"""))
    launch_button.Bind(wx.EVT_BUTTON, fn_launch_frame)
    
    stop_button = wx.Button(frame, label="Stop VM")
    frame.Sizer.Add(stop_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    def fn_stop(event):
        def do_kill_vm():
            attach()
            kill_vm()
            wx.CallAfter(stop_button.Enable, False)
        thread = threading.Thread(target=do_kill_vm)
        thread.start()
    stop_button.Bind(wx.EVT_BUTTON, fn_stop)
    frame.Layout()
    frame.Show()
    app.MainLoop()
        
    
