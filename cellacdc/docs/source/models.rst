Models for automatic segmentation and tracking
==============================================

.. contents::

Cell-ACDC has several models which can be used for segmentation of your data, as well as tracking of objects. Using the Segmentation module, or directly in the GUI, these models can be accessed. 

Available models
----------------

Each model has its own strengths and weaknesses, so make sure to try out 
different models to find the right fit for you. 

Give the corresponding publications a read for further information:

.. include:: _models_list.rst

YeaZ
~~~~
For greater detail, please read the `publication <https://www.nature.com/articles/s41467-020-19557-4>`__, or visit their `GitHub page <https://github.com/rahi-lab/YeaZ-GUI>`__.

YeaZ is a **convolutional neural network**. As the name suggests, it was developed and trained on images of yeast. This means that it **works great with yeast**, but not so much with other things. YeaZ can be used for both **segmentation and tracking**.

However, YeaZ does not work well with bright field images, as it was not trained with such images. **YeaZ2** should have **improved performance for bright field**, so the second version should be a good solution when working with yeast.

Cellpose
~~~~~~~~

For greater detail, please read the `publication <https://www.nature.com/articles/s41592-020-01018-x>`__ or their `web page <https://www.cellpose.org/>`__.

Cellpose was trained on a **more diverse** data set than YeaZ, and thus can be used for segmentation of a wider range of data. If YeaZ is not the right fit for you, Cellpose might be.

Adding a new Model
------------------

If none of the included models suits your purposes, adding your own model might be a good idea.


Adding a segmentation model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a segmentation model in a few steps:

1. Create a **new folder** with the models's name (e.g., YeastMate) inside the ``/cellacdc/models`` folder.

.. tip:: 
    If you **don't know where Cell-ACDC was installed**, open the main launcher and click on the ``Help --> About Cell-ACDC`` menu on the top menu bar.

2. Create a ``__init__.py`` file inside the model folder. In this file you can handle **automatic installation** of the module required by your model. For example, if your model requires the module tensorflow you can install it manually, or programmatically as follows:

    .. code-block:: python
        :linenos:

        import os 
        import sys 
        
        try: 
            import tensorflow 
            # Tries to import the module, in this case btrack

        except ModuleNotFoundError:
            subprocess.check_call( 
                [sys.executable, '-m', 'pip', 'install', 'tensorflow'] 
                # If the model is not found (so its not installed), 
                # it will try installing it using btrack
            )

    Add any line of code needed to initialize correct import of the model.

3. Create a new file called ``acdcSegment.py`` in the model folder with the following template code:

    .. code-block:: python
        :linenos:
        
        import module1 #here put the modules that the tracker needs
        import module2

        class Model:
            def __init__(self, **init_kwargs):
                script_path = os.path.dirname(os.path.realpath(__file__))
                weights_path = os.path.join(script_path, 'model', 'weights')

                self.model = MyModel(
                    weights_path, **init_kwargs
                )

            def segment(self, image, **segment_kwargs):
                lab = self.model.eval(image, **segment_kwargs)
                return lab

The **model parameters** will be **automatically inferred from the class you created** in the ``acdcSegment.py`` file, and a widget with those parameters will pop-up. In this widget you can set the model parameters (or press Ok without changing anything if you want to go with default parameters).

Have a loot at the ``/cellacdc/models`` folder `here <https://github.com/SchmollerLab/Cell_ACDC/tree/main/cellacdc/models>`__ for **examples**. You can for example see the ``__init__.py`` file `here <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/models/YeaZ_v2/__init__.py>`__ and the ``acdcSegment.py`` file `here <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/models/YeaZ_v2/acdcSegment.py>`__ for YeaZ2.


Adding a tracker
~~~~~~~~~~~~~~~~

This only takes a few minutes:

1. Create a **new folder** with the trackers's name (e.g., YeaZ) inside the ``/cellacdc/trackers`` folder.

.. tip:: 
    If you **don't know where Cell-ACDC was installed**, open the main launcher and click on the ``Help --> About Cell-ACDC`` menu on the top menu bar.

2. Create a ``__init__.py`` file inside the folder. In this file you can handle **automatic installation** of the module required by your tracker. For example, if your tracker requires the module ``btrack`` you can install it manually, or programmatically as follows:

    .. code-block:: python
        :linenos:

        import os
        import sys
        
        try:
            import btrack
            # Tries to import the module, in this case btrack

        except ModuleNotFoundError:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'btrack']
                # If the model is not found (so its not installed),
                # it will try installing it using btrack
            )

    Add any line of code needed to initialize correct import of the model.

3. Create a new file called ``trackerName_tracker.py`` (e.g.,`` YeaZ_tracker.py``) in the tracker folder with the following template code:

    .. code-block:: python
        :linenos:

        import module1 #here put the modules that the tracker needs
        import module2 
        
        class tracker: 
            def __init__(self): 
                '''here put the code to initialize tracker''' 
                
            def track(self, segm_video, signals=None, export_to=None):
                '''here put the code to that from a 
                   segmented video Returns the tracked video 
                ''' 
                return tracked_video

Have a look at the already implemented trackers. The ``cellacdc/trackers`` folder can be found `here <https://github.com/SchmollerLab/Cell_ACDC/tree/main/cellacdc/trackers>`__, with `YeaZ_tracker.py <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/trackers/YeaZ/YeaZ_tracker.py>`__ as an example for YeaZ.

That's it. Next time you launch the segmentation module you will be able to select your new tracker.