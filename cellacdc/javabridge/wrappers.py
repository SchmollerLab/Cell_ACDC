# -*- Encoding: utf-8 -*-
'''wrappers.py - Wrappers for Java classes and instances

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

'''

import inspect
import sys
import numpy as np
import javabridge as J

try:
    basestring            # Python 2
except NameError:
    basestring = (str, )  # Python 3


class JWrapper(object):
    '''A class that wraps a Java object
    
    JWrapper uses Java reflection to find a Java object's methods and fields.
    You can then use dot notation to call the methods and get references
    to the fields. If methods return Java objects, these will also be
    wrapped, so you can use the wrapper to do almost anything.
    
    When a class has overloaded methods with the same name, JWrapper will
    try to pick the one that matches the types of the arguments.
    
    To access static methods and fields, use JClassWrapper.
    
    `self.o` is the JB_Object wrapped by the wrapper. You
    can use `self.o` as an argument for the collection wrappers or anywhere
    else you might use a JB_Object.
    
    Usage:
    
        >>> a = JWrapper(javabridge.make_instance("java/util/ArrayList", "()V"))
        >>> a.add("Hello")
        >>> a.add("World")
        >>> a.size()
        2
        >>> a.get(0).toLowerCase()
        hello
    '''
    def __init__(self, o):
        '''Initialize the JWrapper with a Java object
        
        :param o: a Java object (class = JB_Object)
        '''
        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        self.o = o
        self.class_wrapper = J.get_class_wrapper(o)
        env = J.get_env()
        jmethods = env.get_object_array_elements(self.class_wrapper.getMethods())
        methods = {}
        for jmethod in jmethods:
            if (J.call(jmethod, "getModifiers", "()I") & STATIC) == STATIC:
                continue
            method = J.get_method_wrapper(jmethod)
            name = method.getName()
            if name not in methods:
                methods[name] = []
                fn = lambda naame=name: lambda *args: self.__call(naame, *args)
                fn = fn()
                fn.__doc__ = J.to_string(jmethod)
                setattr(self, name, fn)
            else:
                fn = getattr(self, name)
                fn.__doc__ = fn.__doc__ +"\n"+J.to_string(jmethod)
            methods[name].append(method)
        jfields = env.get_object_array_elements(
            self.class_wrapper.getFields(self))
        field_class = env.find_class("java/lang/reflect/Field")
        method_id = env.get_method_id(
            field_class, "getName", "()Ljava/lang/String;")
        self.field_names = [ 
            env.get_string_utf(env.call_method(o, method_id)) for o in jfields]
        self.methods = methods
        
    def __getattr__(self, name):
        if name in ("o", "class_wrapper", "methods", "field_names"):
            raise AttributeError()
        if not hasattr(self, "methods") or not hasattr(self, "field_names"):
            # not initialized
            raise AttributeError()
        if name not in self.field_names:
            raise AttributeError()
        try:
            jfield = self.class_wrapper.getField(name)
        except:
            raise AttributeError()
    
        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        if (J.call(jfield, "getModifiers", "()I") & STATIC) == STATIC:
            raise AttributeError()
        klass = J.call(jfield, "getType", "()Ljava/lang/Class;")
        result = J.get_field(self.o, name, sig(klass))
        if isinstance(result, J.JB_Object):
            result = JWrapper(result)
        return result
    
    def __setattr__(self, name, value):
        if name in ("o", "class_wrapper", "methods", "field_names") or \
           not hasattr(self, "methods"):
            object.__setattr__(self, name, value)
            return
        try:
            jfield = self.class_wrapper.getField(name)
        except:
            object.__setattr__(self, name, value)
            return
    
        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        if (J.call(jfield, "getModifiers", "()I") & STATIC) == STATIC:
            raise AttributeError()
        klass = J.call(jfield, "getType", "()Ljava/lang/Class;")
        result = J.set_field(self.o, name, sig(klass), value)
            
    def __call(self, method_name, *args):
        '''Call the appropriate overloaded method with the given name
        
        :param method_name: the name of the method to call
        :param *args: the arguments to the method, which are used to
                      disambiguate between similarly named methods
        '''
        env = J.get_env()
        last_e = None
        for method in self.methods[method_name]:
            params = env.get_object_array_elements(method.getParameterTypes())
            is_var_args = J.call(method.o, "isVarArgs", "()Z")
            if len(args) < len(params) - (1 if is_var_args else 0):
                continue
            if len(args) > len(params) and not is_var_args:
                continue
            if is_var_args:
                pm1 = len(params)-1
                args1 = list(args[:pm1]) + [args[pm1:]]
            else:
                args1 = args
            try:
                cargs = [cast(o, klass) for o, klass in zip(args1, params)]
            except:
                last_e = sys.exc_info()[1]
                continue
            rtype = J.call(method.o, "getReturnType", "()Ljava/lang/Class;")
            args_sig = "".join(map(sig, params))
            rsig = sig(rtype)
            msig = "(%s)%s" % (args_sig, rsig)
            result =  J.call(self.o, method_name, msig, *cargs)
            if isinstance(result, J.JB_Object):
                result = JWrapper(result)
            return result
        raise TypeError("No matching method found for %s" % method_name)
    
    def __repr__(self):
        classname = J.call(J.call(self.o, "getClass", "()Ljava/lang/Class;"), 
                           "getName", "()Ljava/lang/String;")
        return "Instance of %s: %s" % (classname, J.to_string(self.o))
    
    def __str__(self):
        return J.to_string(self.o)
    
    def __int__(self):
        return self.intValue()

    def __float__(self):
        return self.floatValue()

    def __len__(self):
        if not J.is_instance_of(self.o,'java/util/Collection'):
            raise TypeError("%s is not a Collection and does not support __len__" % self)
        return self.size()

    def __getitem__(self, i):
        if not J.is_instance_of(self.o,'java/util/Collection'):
            raise TypeError("%s is not a Collection and does not support __getitem__" % self)
        return self.get(i)

    def __setitem__(self, i, v):
        if not J.is_instance_of(self.o,'java/util/Collection'):
            raise TypeError("%s is not a Collection and does not support __setitem__" % self)
        return self.set(i, v) 
    
    class Iterator:
        def __init__(self, o):
            self.o = o
            self.i = 0

        def next(self):
            return self.__next__()

        def __next__(self):
            if self.i == len(self.o):
                raise StopIteration
            self.i = self.i +1
            return self.o[self.i-1]

    def __iter__(self):
        if not J.is_instance_of(self.o,'java/util/Collection'):
            raise TypeError("%s is not a Collection and does not support __iter__" % self)
        return self.Iterator(self)

class JClassWrapper(object):
    '''Wrapper for a class
    
    JWrapper uses Java reflection to find a Java object's methods and fields.
    You can then use dot notation to call the static methods and get references
    to the static fields. If methods return Java objects, these will also be
    wrapped, so you can use the wrapper to do almost anything.
    
    When a class has overloaded methods with the same name, JWrapper will
    try to pick the one that matches the types of the arguments.
    
    >>> Integer = JClassWrapper("java.lang.Integer")
    >>> Integer.MAX_VALUE
    2147483647
    '''
    def __init__(self, class_name):
        '''Initialize to wrap a class name
        
        :param class_name: name of class in dotted form, e.g. java.lang.Integer
        '''
        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        self.cname = class_name.replace(".", "/")
        self.klass = J.get_class_wrapper(J.class_for_name(class_name), True)
        self.static_methods = {}
        env = J.get_env()
        jmethods = env.get_object_array_elements(self.klass.getMethods())
        methods = {}
        for jmethod in jmethods:
            if (J.call(jmethod, "getModifiers", "()I") & STATIC) != STATIC:
                continue
            method = J.get_method_wrapper(jmethod)
            name = method.getName()
            if name not in methods:
                methods[name] = []
                fn = lambda naame=name: lambda *args: self.__call_static(naame, *args)
                fn = fn()
                fn.__doc__ = J.to_string(jmethod)
                setattr(self, name, fn)
            else:
                fn = getattr(self, name)
                fn.__doc__ = fn.__doc__ +"\n"+J.to_string(jmethod)
            methods[name].append(method)
        jfields = env.get_object_array_elements(self.klass.getFields(self))
        field_class = env.find_class("java/lang/reflect/Field")
        method_id = env.get_method_id(
            field_class, "getName", "()Ljava/lang/String;")
        self.field_names = [ 
            env.get_string_utf(env.call_method(o, method_id)) for o in jfields]
        self.methods = methods
        
    def __getattr__(self, name):
        if name in ("klass", "static_methods", "methods", "cname", 
                    "field_names"):
            raise AttributeError()
        if not hasattr(self, "methods") or not hasattr(self, "field_names"):
            raise AttributeError()
        if name not in self.field_names:
            raise AttributeError("Cound not find field %s" % name)
        try:
            jfield = self.klass.getField(name)
        except:
            raise AttributeError("Could not find field %s" % name)
    
        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        if (J.call(jfield, "getModifiers", "()I") & STATIC) != STATIC:
            raise AttributeError("Field %s is not static" % name)
        klass = J.call(jfield, "getType", "()Ljava/lang/Class;")
        result = J.get_static_field(self.cname, name, sig(klass))
        if isinstance(result, J.JB_Object):
            result = JWrapper(result)
        return result
    
    def __setattr__(self, name, value):
        if name in ("klass", "static_methods", "methods", "cname", 
                    "field_names") or not hasattr(self, "methods"):
            object.__setattr__(self, name, value)
            return
        try:
            jfield = self.klass.getField(name)
        except:
            return object.__setattr__(self, name, value)

        STATIC = J.get_static_field("java/lang/reflect/Modifier", "STATIC", "I")
        if (J.call(jfield, "getModifiers", "()I") & STATIC) != STATIC:
            raise AttributeError()
        klass = J.call(jfield, "getType", "()Ljava/lang/Class;")
        result = J.set_static_field(self.cname, name, sig(klass), value)
    
    def __call_static(self, method_name, *args):
        '''Call the appropriate overloaded method with the given name
        
        :param method_name: the name of the method to call
        :param *args: the arguments to the method, which are used to
                      disambiguate between similarly named methods
        '''
        env = J.get_env()
        last_e = None
        for method in self.methods[method_name]:
            params = env.get_object_array_elements(method.getParameterTypes())
            is_var_args = J.call(method.o, "isVarArgs", "()Z")
            if len(args) < len(params) - (1 if is_var_args else 0):
                continue
            if len(args) > len(params) and not is_var_args:
                continue
            if is_var_args:
                pm1 = len(params)-1
                args1 = list(args[:pm1]) + [args[pm1:]]
            else:
                args1 = args
            try:
                cargs = [cast(o, klass) for o, klass in zip(args1, params)]
            except:
                last_e = sys.exc_info()[1]
                continue
            rtype = J.call(method.o, "getReturnType", "()Ljava/lang/Class;")
            args_sig = "".join(map(sig, params))
            rsig = sig(rtype)
            msig = "(%s)%s" % (args_sig, rsig)
            result =  J.static_call(self.cname, method_name, msig, *cargs)
            if isinstance(result, J.JB_Object):
                result = JWrapper(result)
            return result
        raise TypeError("No matching method found for %s" % method_name)

    def __call__(self, *args):
        '''Constructors'''
        env = J.get_env()
        jconstructors = self.klass.getConstructors()
        for jconstructor in env.get_object_array_elements(jconstructors):
            constructor = J.get_constructor_wrapper(jconstructor)
            params = env.get_object_array_elements(
                constructor.getParameterTypes())
            is_var_args = J.call(constructor.o, "isVarArgs", "()Z")
            if len(args) < len(params) - (1 if is_var_args else 0):
                continue
            if len(args) > len(params) and not is_var_args:
                continue
            if is_var_args:
                pm1 = len(params)-1
                args1 = list(args[:pm1]) + [args[pm1:]]
            else:
                args1 = args
            try:
                cargs = [cast(o, klass) for o, klass in zip(args1, params)]
            except:
                last_e = sys.exc_info()[1]
                continue
            args_sig = "".join(map(sig, params))
            msig = "(%s)V" % (args_sig)
            result =  J.make_instance(self.cname, msig, *cargs)
            result = JWrapper(result)
            return result
        raise TypeError("No matching constructor found")
    
class JProxy(object):
    '''A wrapper around java.lang.reflect.Proxy
    
    The wrapper takes a dictionary of either method name or a
    `java.lang.reflect.Method` instance to a callable that handles
    the method. You can also subclass JProxy and define methods
    with the same names as the Java methods and they will be called.    

    An example:

        >>> import javabridge
        >>> import sys
        >>> runnable = javabridge.JProxy(
                'java.lang.Runnable',
                dict(run=lambda:sys.stderr.write("Hello, world.\\n"))))
        >>> javabridge.JWrapper(runnable.o).run()
        
    Another example:

        >>> import javabridge
        >>> import sys
        >>> class MyRunnable(javabridge.JProxy):
                def __init__(self):
                    javabridge.JProxy.__init__(self, 'java.lang.Runnable')
                def run(self):
                    sys.stderr.write("Hello, world.\\n")
        >>> proxy = MyRunnable()
        >>> javabridge.JWrapper(runnable.o).run()
    '''
    def __init__(self, base_class_name, d=None):
        '''Initialize the proxy with the interface name and methods
        
        :param base_class_name: the class name of the interface to implement
                                in dotted form (e.g. java.lang.Runnable)
        :param d: an optional dictionary of method name to implementation
        '''
        self.ref_id, self.ref = J.create_jref(self)
        self.__d = d or {}
        jclass = J.class_for_name(base_class_name)
        loader = J.call(jclass, "getClassLoader",
                        "()Ljava/lang/ClassLoader;")
        env = J.get_env()
        classes = env.make_object_array(1, env.find_class("java/lang/Class"))
        env.set_object_array_element(classes, 0, jclass)
        handler = J.make_instance(
            "org/cellprofiler/javabridge/CPythonInvocationHandler",
            "(Ljava/lang/String;)V", self.ref_id)
        self.o = J.static_call(
            "java/lang/reflect/Proxy",
            "newProxyInstance",
            "(Ljava/lang/ClassLoader;"
            "[Ljava/lang/Class;"
            "Ljava/lang/reflect/InvocationHandler;)"
            "Ljava/lang/Object;",
            loader, classes, handler)
        
    def __call__(self, proxy, method, jargs):
        name = J.call(method, "getName", "()Ljava/lang/String;")
        env = J.get_env()
        args = tuple(env.get_object_array_elements(jargs))
        if name in self.__d:
            result = self.__d[name](*args)
        else:
            result = getattr(self, name)(*args)
        retclass = J.call(method, "getReturnType", "()Ljava/lang/Class;")
        return cast(result, retclass)
    
def importClass(class_name, import_name = None):
    '''Import a wrapped class into the global context
    
    :param class_name: a dotted class name such as java.lang.String
    :param import_name: if defined, use this name instead of the class's name
    '''
    if import_name is None:
        if "." in class_name:
            import_name = class_name.rsplit(".", 1)[1]
        else:
            import_name = class_name
    frame = inspect.currentframe(1)
    frame.f_locals[import_name] = JClassWrapper(class_name)

def sig(klass):
    '''Return the JNI signature for a class'''
    name = J.call(klass, "getName", "()Ljava/lang/String;")
    if not (J.call(klass, "isPrimitive", "()Z") or 
            J.call(klass, "isArray", "()Z")):
        name = "L%s;" % name
    if name == 'void':
        return "V"
    if name == 'int':
        return "I"
    if name == 'byte':
        return "B"
    if name == 'boolean':
        return "Z"
    if name == 'long':
        return "J"
    if name == 'float':
        return "F"
    if name == 'double':
        return "D"
    if name == 'char':
        return "C"
    if name == 'short':
        return "S"
    return name.replace(".", "/")

def cast(o, klass):
    '''Cast the given object to the given class
    
    :param o: either a Python object or Java object to be cast
    :param klass: a java.lang.Class indicating the target class
    
    raises a TypeError if the object can't be cast.
    '''
    if J.call(klass, "getName", "()Ljava/lang/String;") == 'void':
        return None
    is_primitive = J.call(klass, "isPrimitive", "()Z")
    csig = sig(klass)
    if o is None:
        if not is_primitive:
            return None
        else:
            raise TypeError("Can't cast None to a primitive type")
        
    if isinstance(o, J.JB_Object):
        if J.call(klass, "isInstance", "(Ljava/lang/Object;)Z", o):
            return o
        classname = J.run_script("o.getClass().getCanonicalName()", dict(o=o))
        klassname = J.run_script("klass.getCanonicalName()", dict(klass=klass))
        raise TypeError("Object of class %s cannot be cast to %s",
                        classname, klassname)
    elif hasattr(o, "o"):
        return cast(o.o, klass)
    elif not np.isscalar(o):
        component_type = J.call(klass, "getComponentType", "()Ljava/lang/Class;")
        if component_type is None:
            raise TypeError("Argument must not be a sequence")
        if len(o) > 0:
            # Test if an element can be cast to the array type
            cast(o[0], component_type)
        return J.get_nice_arg(o, csig)
    elif is_primitive or csig in \
         ('Ljava/lang/String;', 'Ljava/lang/CharSequence;', 
          'Ljava/lang/Object;'):
        if csig == 'Ljava/lang/CharSequence;':
            csig = 'Ljava/lang/String;'
        elif csig == 'C' and isinstance(o, basestring) and len(o) != 1:
            raise TypeError("Failed to convert string of length %d to char" %
                            len(o))
        return J.get_nice_arg(o, csig)
    raise TypeError("Failed to convert argument to %s" % csig)

all = [JWrapper, JClassWrapper]
