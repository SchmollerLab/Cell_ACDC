.. _python-guide: https://focalplane.biologists.com/2022/12/08/managing-scientific-python-environments-using-conda-mamba-and-friends/

Installation
============

Here you will find a detailed guide on how to install Cell-ACDC. In general, 
you should be fine with installing the stable version, however, Cell-ACDC 
development is still quite rapid and if you want to try out the latest 
features we recommend installing the latest version. On the other hand, 
installing from source is required only if you plan to contribute to Cell-ACDC 
development. In that case see this section :ref:`contributing-guide`.

.. tip:: 
    
    If you are **new to Python** or you need a **refresher** on how to manage 
    scientific Python environments, I highly recommend reading 
    `this guide <python-guide>`__ by Dr. Robert Haase.

* :ref:`Install stable version <install-stable-version>`
* :ref:`Install latest version <install-latest-version>`
* :ref:`Install from source <install-from-source-developer-version>`

.. _install-stable-version:

Install stable version
----------------------

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal app.

3. **Update conda** by running the following command:
    
    .. code-block:: 
    
        conda update conda
    
    This will update all packages that are part of conda.

4. **Create a virtual environment** with the following command:
   
    .. code-block:: 
   
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

5. **Activate the virtual environment** with the following command:
   
    .. code-block:: 
   
        conda activate acdc
    
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

    .. important:: 

       Before moving to the next steps make sure that you always activate 
       the ``acdc`` environment. If you close the terminal and reopen it, 
       always run the command ``conda activate acdc`` before installing any 
       package. To know whether the right environment is active, the line 
       on the terminal where you type commands should start with the text 
       ``(acdc)``, like in this screenshot:

       .. tabs::

            .. tab:: Windows

                .. figure:: images/conda_activate_acdc_windows.png
                    :width: 100%

                    Anaconda Prompt after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.
            
            .. tab:: macOS

                .. figure:: images/conda_activate_acdc_macOS.png
                    :width: 100%

                    Terminal app after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.


6. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, Cell-ACDC is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7.  **Install Cell-ACDC** with the following command:
   
    .. code-block:: 
        
        pip install "cellacdc"
        
    This tells pip to install Cell-ACDC.

8. **Install the GUI libraries**:

    After successful installation, you should be able to **run Cell-ACDC with 
    the command** ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC. 

    .. include:: _gui_packages.rst

Updating to the latest stable version of Cell-ACDC 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of Cell-ACDC , open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade cellacdc


.. _install-latest-version:

Install latest version
----------------------

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal app.

3. **Update conda** by running the following command:
    
    .. code-block:: 
    
        conda update conda
    
    This will update all packages that are part of conda.

4. **Create a virtual environment** with the following command:
   
    .. code-block:: 
   
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

5. **Activate the virtual environment** with the following command:
   
    .. code-block:: 
   
        conda activate acdc
    
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

    .. important:: 

       Before moving to the next steps make sure that you always activate 
       the ``acdc`` environment. If you close the terminal and reopen it, 
       always run the command ``conda activate acdc`` before installing any 
       package. To know whether the right environment is active, the line 
       on the terminal where you type commands should start with the text 
       ``(acdc)``, like in this screenshot:

       .. tabs::

            .. tab:: Windows

                .. figure:: images/conda_activate_acdc_windows.png
                    :width: 100%

                    Anaconda Prompt after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.
            
            .. tab:: macOS

                .. figure:: images/conda_activate_acdc_macOS.png
                    :width: 100%

                    Terminal app after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.


6. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, Cell-ACDC is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7.  **Install Cell-ACDC** directly from the GitHub repo with the following command:
   
    .. code-block:: 
        
        pip install "git+https://github.com/SchmollerLab/Cell_ACDC.git"
    
    .. tip:: 

        If you **already have the stable version** and you want to upgrade to the 
        latest version run the following command instead:

        .. code-block::

            pip install --upgrade "git+https://github.com/SchmollerLab/Cell_ACDC.git"

    This tells pip to install Cell-ACDC.

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

8. **Install the GUI libraries**:

    After successful installation, you should be able to **run Cell-ACDC with 
    the command** ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC.  

    .. include:: _gui_packages.rst

Updating to the latest version of Cell-ACDC 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of Cell-ACDC , open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade "git+https://github.com/SchmollerLab/Cell_ACDC.git"


.. _install-from-source-developer-version:

Install from source (developer version)
---------------------------------------

If you want to try out experimental features (and, if you have time, maybe report a bug or two :D), you can install the developer version from source as follows:

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal.

3. **Clone the source code** with the following command:
   
    .. code-block:: 
    
        git clone https://github.com/SchmollerLab/Cell_ACDC.git

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

4. **Navigate to the Cell_ACDC folder** with the following command:
   
    .. code-block:: 
   
        cd Cell_ACDC

    The command ``cd`` stands for "change directory" and it allows you to move 
    between directories in the terminal. 

5. **Update conda** with the following command:
   
    .. code-block:: 

        conda update conda
    
    This will update all packages that are part of conda.

6. Create a **virtual environment** with the following command:
   
    .. code-block:: 
    
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

7. **Activate the virtual environment** with the following command:
   
    .. code-block:: 
    
        conda activate acdc

    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

    .. important:: 

       Before moving to the next steps make sure that you always activate 
       the ``acdc`` environment. If you close the terminal and reopen it, 
       always run the command ``conda activate acdc`` before installing any 
       package. To know whether the right environment is active, the line 
       on the terminal where you type commands should start with the text 
       ``(acdc)``, like in this screenshot:

       .. tabs::

            .. tab:: Windows

                .. figure:: images/conda_activate_acdc_windows.png
                    :width: 100%

                    Anaconda Prompt after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.
            
            .. tab:: macOS

                .. figure:: images/conda_activate_acdc_macOS.png
                    :width: 100%

                    Terminal app after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.

8. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, Cell-ACDC is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

9.  **Install Cell-ACDC** with the following command:
   
    .. code-block:: 
   
        pip install -e "."

    The ``.`` at the end of the command means that you want to install from 
    the current folder in the terminal. This must be the ``Cell_ACDC`` folder 
    that you cloned before. 

10. **Install the GUI libraries**:

    After successful installation, you should be able to **run Cell-ACDC with 
    the command** ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC. 

    .. include:: _gui_packages.rst


Updating Cell-ACDC installed from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update Cell-ACDC installed from source, open a terminal window, navigate to the 
Cell-ACDC folder with the command ``cd Cell_ACDC`` and run ``git pull``.

Since you installed with the ``-e`` flag, pulling with ``git`` is enough.