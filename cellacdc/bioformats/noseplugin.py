# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

from __future__ import absolute_import, unicode_literals

import logging
from nose.plugins import Plugin

import javabridge


log = logging.getLogger(__name__)


class Log4JPlugin(Plugin):
    '''
    Plugin that initializes Log4J in order to avoid Bioformats error
    messages.

    '''
    enabled = False
    name = "log4j"
    score = 90 # Less than the score of javabridge.nosetests.JavaBridgePlugin

    def begin(self):
        javabridge.static_call("org/apache/log4j/BasicConfigurator",
                               "configure", "()V")
        log4j_logger = javabridge.static_call("org/apache/log4j/Logger",
                                              "getRootLogger",
                                              "()Lorg/apache/log4j/Logger;")
        warn_level = javabridge.get_static_field("org/apache/log4j/Level","WARN",
                                                 "Lorg/apache/log4j/Level;")
        javabridge.call(log4j_logger, "setLevel", "(Lorg/apache/log4j/Level;)V",
                        warn_level)
