'''test_wrappers.py test JWrapper and JClassWrapper

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

'''
import unittest
import javabridge as J

class TestJWrapper(unittest.TestCase):
    def test_01_01_init(self):
        jobj = J.get_env().new_string(u"Hello, world.")
        obj = J.JWrapper(jobj)
        self.assertEquals(jobj, obj.o)

    def test_01_02_call_noargs(self):
        jobj = J.get_env().new_string(u"Hello, world.")
        obj = J.JWrapper(jobj)
        self.assertEquals(obj.toLowerCase(), "hello, world.")

    def test_01_03_call_args(self):
        jobj = J.get_env().new_string(u"Hello, world.")
        obj = J.JWrapper(jobj)
        result = obj.replace("Hello,", "Goodbye cruel")
        self.assertEquals(result, "Goodbye cruel world.")

    def test_01_04_call_varargs(self):
        sclass = J.JWrapper(J.class_for_name("java.lang.String"));
        for constructor in J.get_env().get_object_array_elements(
            sclass.getConstructors().o):
            wconstructor = J.JWrapper(constructor)
            parameter_types = J.get_env().get_object_array_elements(
                wconstructor.getParameterTypes().o)
            c1 = sclass.getConstructor(*parameter_types)
            self.assertTrue(c1.equals(constructor))

    def test_02_01_get_field(self):
        obj = J.JClassWrapper("org.cellprofiler.javabridge.test.RealRect")(
            1.5, 2.5, 3.5, 4.5)
        self.assertEquals(obj.x, 1.5)

    def test_02_02_set_field(self):
        obj = J.JClassWrapper("org.cellprofiler.javabridge.test.RealRect")(
            1.5, 2.5, 3.5, 4.5)
        obj.x = 2.5
        self.assertEquals(obj.x, 2.5)

class TestJClassWrapper_Unboxing(unittest.TestCase):
    def setUp(self):
        self.i = J.JClassWrapper('java.lang.Integer')(3)

    def test_01_01_int(self):
        self.assertEquals(int(self.i), 3)
    
    def test_01_02_float(self):
        self.assertEquals(float(self.i),3.0)
    
    def test_01_03_str(self):
        self.assertEquals(str(self.i), '3')

class TestJClassWrapper_Collection(unittest.TestCase):
    def setUp(self):
        self.a = J.JClassWrapper('java.util.ArrayList')()
        self.assertEquals(len(self.a), 0)
        self.ints = [0,1,2,4,8,16]
        self.assertEquals(len(self.ints), 6)
        for i in self.ints:
            self.a.add(i)

    def test_01_01_get_len(self):
        self.assertEquals(len(self.a), len(self.ints))

    def test_01_02_iterate(self):
        for x,y in zip(self.a, self.ints):
            self.assertEquals(x.intValue(), y)

    def test_01_03_get_index(self):
        for i in range(len(self.a)):
            self.assertEquals(self.a[i].intValue(), self.ints[i])

    def test_01_04_set_index(self):
        for i in range(len(self.a)):
            self.a[i] = 10
        for i in range(len(self.a)):
            self.assertEquals(self.a[i].intValue(), 10)

class TestJClassWrapper(unittest.TestCase):
    def test_01_01_init(self):
        c = J.JClassWrapper("java.lang.Integer")

    def test_01_02_field(self):
        c = J.JClassWrapper("java.lang.Short")
        field = c.MAX_VALUE
        self.assertEquals(field, (1 << 15)-1)

    def test_02_03_static_call(self):
        c = J.JClassWrapper("java.lang.Integer")
        self.assertEquals(c.toString(123), "123")

    def test_02_04_static_call_varargs(self):
        #
        # Test calling a static function with a variable number of
        # arguments.
        #
        c = J.JClassWrapper("java.lang.String")
        self.assertEquals(c.format("Hello, %s.", "world"),
                                   "Hello, world.")
        self.assertEquals(c.format("Goodbye %s %s.", "cruel", "world"),
                          "Goodbye cruel world.")

    def test_02_05_constructor_varargs(self):
        # Regression test of issue #41
        #
        args = ("foo", "bar")
        f = J.JClassWrapper(
            "javax.swing.filechooser.FileNameExtensionFilter")("baz", *args)
        exts = J.get_env().get_object_array_elements(f.getExtensions().o)
        self.assertEqual(args[0], J.to_string(exts[0]))
        self.assertEqual(args[1], J.to_string(exts[1]))

class TestJProxy(unittest.TestCase):
    def test_01_01_init(self):
        def whatever():
            pass
        J.JProxy('java.lang.Runnable', dict(run=whatever))

    def test_01_02_runnable(self):
        magic = []
        def whatever(magic=magic):
            magic.append("bus")
        runnable = J.JProxy('java.lang.Runnable',
                            dict(run=whatever))
        J.JWrapper(runnable.o).run()
        self.assertEqual(magic[0], "bus")

    def test_01_03_runnable_class(self):
        class MyProxy(J.JProxy):
            def __init__(self):
                J.JProxy.__init__(self, 'java.lang.Runnable')
                self.esteem = 0

            def run(self):
                self.esteem = "through the roof"

        proxy = MyProxy()
        J.JWrapper(proxy.o).run()
        self.assertEqual(proxy.esteem, "through the roof")

    def test_01_04_args(self):
        my_observable = J.make_instance("java/util/Observable", "()V")
        def update(observable, obj):
            self.assertTrue(J.JWrapper(observable).equals(my_observable))
            self.assertTrue(J.JWrapper(obj).equals("bar"))
        proxy = J.JProxy('java.util.Observer', dict(update=update))
        J.JWrapper(proxy.o).update(my_observable, "bar")

    def test_01_05_return_value(self):
        def call():
            return "foo"
        proxy = J.JProxy('java.util.concurrent.Callable',
                         dict(call = call))
        self.assertEquals(J.JWrapper(proxy.o).call(), "foo")

if __name__=="__main__":
    import javabridge
    javabridge.start_vm()
    try:
        unittest.main()
    finally:
        javabridge.kill_vm()
