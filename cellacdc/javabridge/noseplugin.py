"""noseplugin.py - start and stop JVM when running unit tests

python-javabridge is licensed under the BSD license.  See the
accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

"""

import logging
from nose.plugins import Plugin
import os
import numpy as np
np.seterr(all='ignore')
import sys


log = logging.getLogger(__name__)


class JavabridgePlugin(Plugin):
    '''Javabridge nose test plugin
    
    This plugin starts the JVM before running tests and kills it when
    the tests are done. The plugin is necessary because the JVM cannot
    be restarted once it is killed, so unittest's setUp() and
    tearDown() methods cannot be used to start and stop the JVM.
    '''
    enabled = False
    name = "javabridge"
    score = 100
    extra_jvm_args = []

    def begin(self):
        import javabridge

        javabridge.start_vm(self.extra_jvm_args,
                            class_path=self.class_path.split(os.pathsep),
                            run_headless=self.headless,
                            max_heap_size=self.max_heap_size)
        if not self.headless:
            javabridge.activate_awt()

    def options(self, parser, env=os.environ):
        super(JavabridgePlugin, self).options(parser, env=env)
        parser.add_option("--classpath", action="store",
                          default=env.get('NOSE_CLASSPATH'),
                          metavar="PATH",
                          dest="classpath",
                          help="Additional class path for JVM [NOSE_CLASSPATH]")
        parser.add_option("--no-headless", action="store_true",
                          default=bool(env.get('NOSE_NO_HEADLESS')),
                          dest="no_headless",
                          help="Set Java environment variable, java.awt.headless to false to allow AWT calls [NOSE_NO_HEADLESS]")
        parser.add_option("--max-heap-size", action="store",
                          default=env.get('NOSE_MAX_HEAP_SIZE'),
                          dest="max_heap_size",
                          help="Set the maximum heap size argument to the JVM as in the -Xmx command-line argument [NOSE_MAX_HEAP_SIZE]")

    def configure(self, options, conf):
        import javabridge
        super(JavabridgePlugin, self).configure(options, conf)
        self.class_path = os.pathsep.join(javabridge.JARS)
        if options.classpath:
            self.class_path = os.pathsep.join([options.classpath, self.class_path])
        self.headless = not options.no_headless
        self.max_heap_size = options.max_heap_size

    def prepareTestRunner(self, testRunner):
        '''Need to make the test runner call finalize if in Wing
        
        Wing IDE's XML test runner fails to call finalize, so we
        wrap it and add that function here
        '''
        if (getattr(testRunner, "__module__","unknown") == 
            "wingtest_common"):
            outer_self = self
            class TestRunnerProxy(object):
                def run(self, test):
                    result = testRunner.run(test)
                    outer_self.finalize(testRunner.result)
                    return result
                
                @property
                def result(self):
                    return testRunner.result
            return TestRunnerProxy()
            
    def finalize(self, result):
        import javabridge
        javabridge.kill_vm()
