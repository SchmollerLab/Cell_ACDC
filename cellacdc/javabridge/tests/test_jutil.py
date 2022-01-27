'''test_jutil.py - test the high-level interface

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

'''

import gc
import os
import numpy as np
import threading
import unittest
import sys

import javabridge

# Monkey patch some half-corrent implementations of methods that only
# appeared in Python 2.7.
if not hasattr(unittest.TestCase, 'assertIn'):
    unittest.TestCase.assertIn = lambda self, a, b: self.assertTrue(a in b)
if not hasattr(unittest.TestCase, 'assertNotIn'):
    unittest.TestCase.assertNotIn = lambda self, a, b: self.assertTrue(a not in b)
if not hasattr(unittest.TestCase, 'assertSequenceEqual'):
    unittest.TestCase.assertSequenceEqual = lambda self, a, b: self.assertTrue([aa == bb for aa, bb in zip(a, b)])


class TestJutil(unittest.TestCase):

    def setUp(self):
        self.env = javabridge.attach()
    
    def tearDown(self):
        javabridge.detach()
        
    def test_01_01_to_string(self):
        jstring = self.env.new_string_utf("Hello, world")
        self.assertEqual(javabridge.to_string(jstring), "Hello, world")
        
    def test_01_02_make_instance(self):
        jobject = javabridge.make_instance("java/lang/Object", "()V")
        self.assertTrue(javabridge.to_string(jobject).startswith("java.lang.Object"))
        
    def test_01_03_call(self):
        jstring = self.env.new_string_utf("Hello, world")
        self.assertEqual(javabridge.call(jstring, "charAt", "(I)C", 0), "H")
        
    def test_01_03_01_static_call(self):
        result = javabridge.static_call("Ljava/lang/String;", "valueOf", 
                               "(I)Ljava/lang/String;",123)
        self.assertEqual(result, "123")
        
    def test_01_04_make_method(self):
        env = self.env
        class String(object):
            def __init__(self):
                self.o = env.new_string_utf("Hello, world")
                
            charAt = javabridge.make_method("charAt", "(I)C", "My documentation")
            
        s = String()
        self.assertEqual(s.charAt.__doc__, "My documentation")
        self.assertEqual(s.charAt(0), "H")
    
    def test_01_05_00_get_static_field(self):
        klass = self.env.find_class("java/lang/Short")
        self.assertEqual(javabridge.get_static_field(klass, "MAX_VALUE", "S"), 2**15 - 1)

    def test_01_05_01_no_field_for_get_static_field(self):
        def fn():
            javabridge.get_static_field(
                'java/lang/Object', "NoSuchField", "I")
        self.assertRaises(javabridge.JavaException, fn)
        
    def test_01_05_02_no_class_for_get_static_field(self):
        def fn():
            javabridge.get_static_field(
                'no/such/class', "field", "I")
        self.assertRaises(javabridge.JavaException, fn)
        
    def test_01_05_03_set_static_field(self):
        class_name = "org/cellprofiler/javabridge/test/RealRect"
        test_cases = (
            ("fs_char", "C", "A"),
            ("fs_byte", "B", 3),
            ("fs_short", "S", 15),
            ("fs_int", "I", 392),
            ("fs_long", "J", -14),
            ("fs_float", "F", 1.03),
            ("fs_double", "D", -889.1),
            ("fs_object", "Ljava/lang/Object;", 
             javabridge.make_instance("java/lang/Integer", "(I)V", 15)),
            ("fs_object", "Ljava/lang/Object;", None))
        for field_name, signature, value in test_cases:
            javabridge.set_static_field(class_name, field_name, signature, value)
            v = javabridge.get_static_field(class_name, field_name, signature)
            if isinstance(value, float):
                self.assertAlmostEqual(v, value)
            elif isinstance(value, javabridge.JB_Object):
                self.assertTrue(javabridge.call(
                    value, "equals", "(Ljava/lang/Object;)Z", v))
            else:
                self.assertEqual(v, value)
                
    def test_01_05_04_no_field_for_set_static_field(self):
        def fn():
            javabridge.set_static_field(
                'java/lang/Object', "NoSuchField", "I", 5)
        self.assertRaises(javabridge.JavaException, fn)
        
    def test_01_05_05_no_class_for_set_static_field(self):
        def fn():
            javabridge.set_static_field(
                'no/such/class', "field", "I", 5)
        self.assertRaises(javabridge.JavaException, fn)
    
    def test_01_06_get_enumeration_wrapper(self):
        properties = javabridge.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        keys = javabridge.call(properties, "keys", "()Ljava/util/Enumeration;")
        enum = javabridge.get_enumeration_wrapper(keys)
        has_java_vm_name = False
        while(enum.hasMoreElements()):
            key = javabridge.to_string(enum.nextElement())
            if key == "java.vm.name":
                has_java_vm_name = True
        self.assertTrue(has_java_vm_name)
        
    def test_01_07_get_dictionary_wrapper(self):
        properties = javabridge.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = javabridge.get_dictionary_wrapper(properties)
        self.assertTrue(d.size() > 10)
        self.assertFalse(d.isEmpty())
        keys = javabridge.get_enumeration_wrapper(d.keys())
        values = javabridge.get_enumeration_wrapper(d.elements())
        n_elems = d.size()
        for i in range(n_elems):
            self.assertTrue(keys.hasMoreElements())
            key = javabridge.to_string(keys.nextElement())
            self.assertTrue(values.hasMoreElements())
            value = javabridge.to_string(values.nextElement())
            self.assertEqual(javabridge.to_string(d.get(key)), value)
        self.assertFalse(keys.hasMoreElements())
        self.assertFalse(values.hasMoreElements())
        
    def test_01_08_jenumeration_to_string_list(self):
        properties = javabridge.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = javabridge.get_dictionary_wrapper(properties)
        keys = javabridge.jenumeration_to_string_list(d.keys())
        enum = javabridge.get_enumeration_wrapper(d.keys())
        for i in range(d.size()):
            key = javabridge.to_string(enum.nextElement())
            self.assertEqual(key, keys[i])
    
    def test_01_09_jdictionary_to_string_dictionary(self):
        properties = javabridge.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = javabridge.get_dictionary_wrapper(properties)
        pyd = javabridge.jdictionary_to_string_dictionary(properties)
        keys = javabridge.jenumeration_to_string_list(d.keys())
        for key in keys:
            value = javabridge.to_string(d.get(key))
            self.assertEqual(pyd[key], value)
            
    def test_01_10_make_new(self):
        env = self.env
        class MyClass:
            new_fn = javabridge.make_new("java/lang/Object", '()V')
            def __init__(self):
                self.new_fn()
        my_instance = MyClass()
        
    def test_01_11_class_for_name(self):
        c = javabridge.class_for_name('java.lang.String')
        name = javabridge.call(c, 'getCanonicalName', '()Ljava/lang/String;')
        self.assertEqual(name, 'java.lang.String')
    
    def test_02_01_access_object_across_environments(self):
        #
        # Create an object in one environment, close the environment,
        # open a second environment, then use it and delete it.
        #
        env = self.env
        self.assertTrue(isinstance(env,javabridge.JB_Env))
        class MyInteger:
            new_fn = javabridge.make_new("java/lang/Integer",'(I)V')
            def __init__(self, value):
                self.new_fn(value)
            intValue = javabridge.make_method("intValue", '()I')
        my_value = 543
        my_integer=MyInteger(my_value)
        def run(my_integer = my_integer):
            env = javabridge.attach()
            self.assertEqual(my_integer.intValue(),my_value)
            javabridge.detach()
        t = threading.Thread(target = run)
        t.start()
        t.join()
        
    def test_02_02_delete_in_environment(self):
        env = self.env
        self.assertTrue(isinstance(env, javabridge.JB_Env))
        class MyInteger:
            new_fn = javabridge.make_new("java/lang/Integer",'(I)V')
            def __init__(self, value):
                self.new_fn(value)
            intValue = javabridge.make_method("intValue", '()I')
        my_value = 543
        my_integer=MyInteger(my_value)
        def run(my_integer = my_integer):
            env = javabridge.attach()
            self.assertEqual(my_integer.intValue(),my_value)
            del my_integer
            javabridge.detach()
        t = threading.Thread(target = run)
        t.start()
        t.join()
        
    def test_02_03_death_and_resurrection(self):
        '''Put an object into another in Java, delete it in Python and recover it'''
        
        np.random.seed(24)
        my_value = np.random.randint(0, 1000)
        jobj = javabridge.make_instance("java/lang/Integer", "(I)V", my_value)
        integer_klass = self.env.find_class("java/lang/Integer")
        jcontainer = self.env.make_object_array(1, integer_klass)
        self.env.set_object_array_element(jcontainer, 0, jobj)
        del jobj
        gc.collect()
        jobjs = self.env.get_object_array_elements(jcontainer)
        jobj = jobjs[0]
        self.assertEqual(javabridge.call(jobj, "intValue", "()I"), my_value)
    
    def test_02_04_non_java_thread_deletes_it(self):
        '''Delete a Java object on a not-Java thread'''
        refs = [javabridge.make_instance("java/lang/Integer", "(I)V", 5)]
        def run():
            del refs[0]
            gc.collect()
        t = threading.Thread(target = run)
        t.start()
        t.join()
        
    def test_03_01_cw_from_class(self):
        '''Get a class wrapper from a class'''
        c = javabridge.get_class_wrapper(javabridge.make_instance('java/lang/Integer', '(I)V',
                                                14))
    
    def test_03_02_cw_from_string(self):
        '''Get a class wrapper from a string'''
        c = javabridge.get_class_wrapper("java.lang.Number")
        
    def test_03_03_cw_get_classes(self):
        c = javabridge.get_class_wrapper('java.lang.Number')
        classes = c.getClasses()
        self.assertEqual(len(javabridge.get_env().get_object_array_elements(classes)), 0)
        
    def test_03_04_cw_get_annotation(self):
        c = javabridge.get_class_wrapper('java.security.Identity')
        annotation = c.getAnnotation(javabridge.class_for_name('java.lang.Deprecated'))
        self.assertTrue(annotation is not None)
    
    def test_03_05_cw_get_annotations(self):
        c = javabridge.get_class_wrapper('java.security.Identity')
        annotations = c.getAnnotations()
        annotations = javabridge.get_env().get_object_array_elements(annotations)
        self.assertEqual(len(annotations), 1)
        self.assertTrue(javabridge.to_string(annotations[0]).startswith('@java.lang.Deprecated'))
        
    def test_03_06_cw_get_constructors(self):
        c = javabridge.get_class_wrapper('java.lang.String')
        constructors = c.getConstructors()
        constructors = javabridge.get_env().get_object_array_elements(constructors)
        self.assertEqual(len(constructors), 15)
        
    def test_03_07_cw_get_fields(self):
        c = javabridge.get_class_wrapper('java.lang.String')
        fields = c.getFields()
        fields = javabridge.get_env().get_object_array_elements(fields)
        self.assertEqual(len(fields), 1)
        self.assertEqual(javabridge.call(fields[0], 'getName', '()Ljava/lang/String;'),
                         "CASE_INSENSITIVE_ORDER")
        
    def test_03_08_cw_get_field(self):
        c = javabridge.get_class_wrapper('java.lang.String')
        field = c.getField('CASE_INSENSITIVE_ORDER')
        modifiers = javabridge.call(field, 'getModifiers', '()I')
        static = javabridge.get_static_field('java/lang/reflect/Modifier','STATIC','I')
        self.assertEqual((modifiers & static), static)
        
    def test_03_09_cw_get_method(self):
        sclass = javabridge.class_for_name('java.lang.String')
        iclass = javabridge.get_static_field('java/lang/Integer', 'TYPE', 
                                    'Ljava/lang/Class;')
        c = javabridge.get_class_wrapper('java.lang.String')
        m = c.getMethod('charAt', [ iclass ])
        self.assertEqual(javabridge.to_string(javabridge.call(m, 'getReturnType', '()Ljava/lang/Class;')), 'char')
        m = c.getMethod('concat', [ sclass])
        self.assertEqual(javabridge.to_string(javabridge.call(m, 'getReturnType', '()Ljava/lang/Class;')), 
                         'class java.lang.String')
        
    def test_03_10_cw_get_methods(self):
        c = javabridge.get_class_wrapper('java.lang.String')
        mmm = javabridge.get_env().get_object_array_elements(c.getMethods())
        self.assertTrue(any([javabridge.call(m, 'getName', '()Ljava/lang/String;') == 'concat'
                             for m in mmm]))
        
    def test_03_11_cw_get_constructor(self):
        c = javabridge.get_class_wrapper('java.lang.String')
        sclass = javabridge.class_for_name('java.lang.String')
        constructor = c.getConstructor([sclass])
        self.assertEqual(javabridge.call(constructor, 'getName', '()Ljava/lang/String;'),
                         'java.lang.String')
        
    def test_04_01_field_get(self):
        c = javabridge.get_class_wrapper('java.lang.Byte')
        f = javabridge.get_field_wrapper(c.getField('MAX_VALUE'))
        v = f.get(None)
        self.assertEqual(javabridge.to_string(v), '127')
        
    def test_04_02_field_name(self):
        c = javabridge.get_class_wrapper('java.lang.Byte')
        f = javabridge.get_field_wrapper(c.getField('MAX_VALUE'))
        self.assertEqual(f.getName(), 'MAX_VALUE')
        
    def test_04_03_field_type(self):
        c = javabridge.get_class_wrapper('java.lang.Byte')
        f = javabridge.get_field_wrapper(c.getField('MAX_VALUE'))
        t = f.getType()
        self.assertEqual(javabridge.to_string(t), 'byte')
        
    def test_05_01_run_script(self):
        self.assertEqual(javabridge.run_script("2+2"), 4)
        
    def test_05_02_run_script_with_inputs(self):
        self.assertEqual(javabridge.run_script("a+b", bindings_in={"a":2, "b":3}), 5)
        
    def test_05_03_run_script_with_outputs(self):
        outputs = { "result": None}
        javabridge.run_script("var result = 2+2;", bindings_out=outputs)
        self.assertEqual(outputs["result"], 4)
        
    def test_06_01_execute_asynch_main(self):
        javabridge.execute_runnable_in_main_thread(javabridge.run_script(
            "new java.lang.Runnable() { run:function() {}};"))
        
    def test_06_02_execute_synch_main(self):
        javabridge.execute_runnable_in_main_thread(javabridge.run_script(
            "new java.lang.Runnable() { run:function() {}};"), True)
        
    def test_06_03_future_main(self):
        c = javabridge.run_script("""
        new java.util.concurrent.Callable() {
           call: function() { return 2+2; }};""")
        result = javabridge.execute_future_in_main_thread(
            javabridge.make_future_task(c, fn_post_process=javabridge.unwrap_javascript))
        self.assertEqual(result, 4)
        
    def test_07_01_wrap_future(self):
        future = javabridge.run_script("""
        new java.util.concurrent.FutureTask(
            new java.util.concurrent.Callable() {
               call: function() { return 2+2; }});""")
        wfuture = javabridge.get_future_wrapper(
            future, fn_post_process=javabridge.unwrap_javascript)
        self.assertFalse(wfuture.isDone())
        self.assertFalse(wfuture.isCancelled())
        wfuture.run()
        self.assertTrue(wfuture.isDone())
        self.assertEqual(wfuture.get(), 4)
        
    def test_07_02_cancel_future(self):
        future = javabridge.run_script("""
        new java.util.concurrent.FutureTask(
            new java.util.concurrent.Callable() {
               call: function() { return 2+2; }});""")
        wfuture = javabridge.get_future_wrapper(
            future, fn_post_process=javabridge.unwrap_javascript)
        wfuture.cancel(True)
        self.assertTrue(wfuture.isCancelled())
        self.assertRaises(javabridge.JavaException, wfuture.get)
        
    def test_07_03_make_future_task_from_runnable(self):
        future = javabridge.make_future_task(
            javabridge.run_script("new java.lang.Runnable() { run: function() {}};"),
            11)
        future.run()
        self.assertEqual(javabridge.call(future.get(), "intValue", "()I"), 11)
        
    def test_07_04_make_future_task_from_callable(self):
        call_able = javabridge.run_script("""
        new java.util.concurrent.Callable() { 
            call: function() { return 2+2; }};""")
        future = javabridge.make_future_task(
            call_able, fn_post_process=javabridge.unwrap_javascript)
        future.run()
        self.assertEqual(future.get(), 4)
        
    def test_08_01_wrap_collection(self):
        c = javabridge.make_instance("java/util/HashSet", "()V")
        w = javabridge.get_collection_wrapper(c)
        self.assertFalse(hasattr(w, "addI"))
        self.assertEqual(w.size(), 0)
        self.assertEqual(len(w), 0)
        self.assertTrue(w.isEmpty())
        
    def test_08_02_add(self):
        c = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        self.assertTrue(c.add("Foo"))
        self.assertEqual(len(c), 1)
        self.assertFalse(c.isEmpty())
        
    def test_08_03_contains(self):
        c = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c.add("Foo")
        self.assertTrue(c.contains("Foo"))
        self.assertFalse(c.contains("Bar"))
        self.assertIn("Foo", c)
        self.assertNotIn("Bar", c)
        
    def test_08_04_addAll(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Baz")
        c2.addAll(c1.o)
        self.assertIn("Foo", c2)
        
    def test_08_05__add__(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Baz")
        c3 = c1 + c2
        for k in ("Foo", "Bar", "Baz"):
            self.assertIn(k, c3)
        
        c4 = c3 + ["Hello", "World"]
        self.assertIn("Hello", c4)
        self.assertIn("World", c4)
        
    def test_08_06__iadd__(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Baz")
        c2 += c1
        for k in ("Foo", "Bar", "Baz"):
            self.assertIn(k, c2)
        c2 += ["Hello", "World"]
        self.assertIn("Hello", c2)
        self.assertIn("World", c2)
        
    def test_08_07_contains_all(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Baz")
        self.assertFalse(c2.containsAll(c1.o))
        c2 += c1
        self.assertTrue(c2.containsAll(c1.o))
        
    def test_08_08_remove(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c1.remove("Foo")
        self.assertNotIn("Foo", c1)
        
    def test_08_09_removeAll(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Foo")
        c1.removeAll(c2)
        self.assertNotIn("Foo", c1)
        
    def test_08_10_retainAll(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        c2 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c2.add("Foo")
        c1.retainAll(c2)
        self.assertIn("Foo", c1)
        self.assertNotIn("Bar", c1)
        
    def test_08_11_toArray(self):
        c1 = javabridge.get_collection_wrapper(javabridge.make_instance("java/util/HashSet", "()V"))
        c1.add("Foo")
        c1.add("Bar")
        result = [javabridge.to_string(x) for x in c1.toArray()]
        self.assertIn("Foo", result)
        self.assertIn("Bar", result)
        
    def test_08_12_make_list(self):
        l = javabridge.make_list(["Foo", "Bar"])
        self.assertSequenceEqual(l, ["Foo", "Bar"])
        self.assertTrue(hasattr(l, "addI"))
        
    def test_08_13_addI(self):
        l = javabridge.make_list(["Foo", "Bar"])
        l.addI(1, "Baz")
        self.assertSequenceEqual(l, ["Foo", "Baz", "Bar"])
        
    def test_08_14_addAllI(self):
        l = javabridge.make_list(["Foo", "Bar"])
        l.addAllI(1, javabridge.make_list(["Baz"]))
        self.assertSequenceEqual(l, ["Foo", "Baz", "Bar"])
        
    def test_08_15_indexOf(self):
        l = javabridge.make_list(["Foo", "Bar"])
        self.assertEqual(l.indexOf("Bar"), 1)
        self.assertEqual(l.lastIndexOf("Foo"), 0)
        
    def test_08_16_get(self):
        l = javabridge.make_list(["Foo", "Bar"])
        self.assertEqual(l.get(1), "Bar")
        
    def test_08_17_set(self):
        l = javabridge.make_list(["Foo", "Bar"])
        l.set(1, "Baz")
        self.assertEqual(l.get(1), "Baz")
        
    def test_08_18_subList(self):
        l = javabridge.make_list(["Foo", "Bar", "Baz", "Hello", "World"])
        self.assertSequenceEqual(l.subList(1, 3), ["Bar", "Baz"])
        
    def test_08_19__getitem__(self):
        l = javabridge.make_list(["Foo", "Bar", "Baz", "Hello", "World"])
        self.assertEqual(l[1], "Bar")
        self.assertEqual(l[-2], "Hello")
        self.assertSequenceEqual(l[1:3], ["Bar", "Baz"])
        self.assertSequenceEqual(l[::3], ["Foo", "Hello"])
        
    def test_08_20__setitem__(self):
        l = javabridge.make_list(["Foo", "Bar"])
        l[1] = "Baz"
        self.assertEqual(l.get(1), "Baz")
        
    def test_08_21__delitem__(self):
        l = javabridge.make_list(["Foo", "Bar", "Baz"])
        del l[1]
        self.assertSequenceEqual(l, ["Foo", "Baz"])
        
    def test_09_01_00_get_field(self):
        o = javabridge.make_instance("org/cellprofiler/javabridge/test/RealRect", "(DDDD)V", 1, 2, 3, 4)
        self.assertEqual(javabridge.get_field(o, "x", "D"), 1)
        
    def test_09_02_get_field_no_such_field(self):
        def fn():
            o = javabridge.make_instance("java/lang/Object", "()V")
            javabridge.get_field(o, "NoSuchField", "I")
        self.assertRaises(javabridge.JavaException, fn)
        
    def test_09_03_set_field(self):
        class_name = "org/cellprofiler/javabridge/test/RealRect"
        o = javabridge.make_instance(class_name, "()V")
        test_cases = (
            ("f_char", "C", "A"),
            ("f_byte", "B", 3),
            ("f_short", "S", 15),
            ("f_int", "I", 392),
            ("f_long", "J", -14),
            ("f_float", "F", 1.03),
            ("f_double", "D", -889.1),
            ("f_object", "Ljava/lang/Object;", 
             javabridge.make_instance("java/lang/Integer", "(I)V", 15)),
            ("f_object", "Ljava/lang/Object;", None))
        for field_name, signature, value in test_cases:
            javabridge.set_field(o, field_name, signature, value)
            v = javabridge.get_field(o, field_name, signature)
            if isinstance(value, float):
                self.assertAlmostEqual(v, value)
            elif isinstance(value, javabridge.JB_Object):
                self.assertTrue(javabridge.call(
                    value, "equals", "(Ljava/lang/Object;)Z", v))
            else:
                self.assertEqual(v, value)

    def test_09_04_set_field_no_such_field(self):
        def fn():
            o = javabridge.make_instance("java/lang/Object", "()V")
            javabridge.set_field(o, "NoSuchField", "I", 1)
        self.assertRaises(javabridge.JavaException, fn)
        
    def test_10_01_iterate_java_on_non_iterator(self):
        #
        # Regression test of issue #11: the expression below segfaulted
        #
        def fn():
            list(javabridge.iterate_java(javabridge.make_list(range(10)).o))
        self.assertRaises(javabridge.JavaError, fn)

    def test_10_01_class_path(self):
        for arg in ['-cp', '-classpath', '-Djava.class.path=foo']:
            self.assertRaises(ValueError, lambda: javabridge.start_vm([arg]))

    def test_11_01_make_run_dictionary(self):
        from javabridge.jutil import make_run_dictionary
        o = javabridge.make_instance("java/util/Hashtable", "()V")
        a = javabridge.make_instance("java/util/ArrayList", "()V")
        javabridge.call(
            o, "put", 
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            "foo", "bar")
        javabridge.call(
            o, "put", 
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            "baz", a)
        d = make_run_dictionary(o)
        self.assertIn("foo", d)
        self.assertEquals(d["foo"], "bar")
        self.assertIn("baz", d)
        self.assertTrue(javabridge.call(d["baz"], "equals", 
                                        "(Ljava/lang/Object;)Z", a))
        
    def test_12_01_jref(self):
        o = dict(foo="bar", baz="2")
        ref_id, ref = javabridge.create_jref(o)
        alt = javabridge.redeem_jref(ref_id)
        o["bar"] = "bunny"
        for key in o:
            self.assertTrue(key in alt)
            self.assertEqual(o[key], alt[key])
        
    def test_12_02_jref_lost(self):
        o = dict(foo="bar", baz="2")
        ref_id, ref = javabridge.create_jref(o)
        del ref
        self.assertRaises(KeyError, javabridge.redeem_jref, ref_id)
        
    def test_12_03_jref_create_and_lock(self):
        cpython = javabridge.JClassWrapper(
            'org.cellprofiler.javabridge.CPython')()
        d = javabridge.JClassWrapper('java.util.Hashtable')()
        result = javabridge.JClassWrapper('java.util.ArrayList')()
        d.put("result", result)
        ref_self = javabridge.create_and_lock_jref(self)
        d.put("self", ref_self)
        cpython.execute(
            'import javabridge\n'
            'x = { "foo":"bar"}\n'
            'ref_id = javabridge.create_and_lock_jref(x)\n'
            'javabridge.JWrapper(result).add(ref_id)', d, d)
        cpython.execute(
            'import javabridge\n'
            'ref_id = javabridge.JWrapper(result).get(0)\n'
            'self = javabridge.redeem_jref(javabridge.to_string(self))\n'
            'self.assertEqual(javabridge.redeem_jref(ref_id)["foo"], "bar")\n'
            'javabridge.unlock_jref(ref_id)', d, d)
        javabridge.unlock_jref(ref_self)
        self.assertRaises(KeyError, javabridge.redeem_jref, ref_self)
        
    def test_13_01_unicode_arg(self):
        # On 2.x, check that a unicode argument is properly prepared
        s = u"Hola ni\u00F1os"
        s1, s2 = s.split(" ")
        if sys.version_info.major == 2:
            s2 = s2.encode("utf-8")
        env = javabridge.get_env()
        js1 = env.new_string(s1+" ")
        result = javabridge.call(
            js1, "concat", "(Ljava/lang/String;)Ljava/lang/String;", s2)
        self.assertEqual(s, result)
        
if __name__=="__main__":
    unittest.main()
