import os
import javabridge as jv

from cellacdc import bioformats

# import bioformats

print(bioformats.__file__)

# path = r'"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\Cell_ACDC\cellacdc\bioformats\jars\bioformats_package.jar"'
jars = bioformats.JARS
print(jars)
jv.start_vm(class_path=jars)
paths = jv.JClassWrapper('java.lang.System').getProperty('java.class.path').split(";")

for path in paths:
    print("%s: %s" %("exists" if os.path.isfile(path) else "missing", path))

jv.kill_vm()
