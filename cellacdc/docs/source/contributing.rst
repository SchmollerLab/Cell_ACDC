.. _contributing-guide:

How to contribute to Cell-ACDC
==============================

Contributions to Cell-ACDC are **always very welcome**! If you have questions about 
contributing feel free to open a new thread on our 
`forum <https://github.com/SchmollerLab/Cell_ACDC/discussions>`_.

Development process
-------------------

1. If this is the first time you contribute:

   * Go to our `GitHub page <https://github.com/SchmollerLab/Cell_ACDC>`_ 
     and click the "fork" button to create your own copy of the project.

   * Open a terminal window. On Windows I recommend using the `PowerShell 7 
     <https://learn.microsoft.com/en-ie/powershell/scripting/install/installing-powershell-on-windows>`_

   * Clone the forked project to your local computer (remember to replace `your-username` in the link below)::

        git clone https://github.com/your-username/Cell_ACDC.git

   * Navigate to the ``Cell_ACDC`` directory::

        cd Cell_ACDC

   * Add the upstream repository::

        git remote add upstream https://github.com/SchmollerLab/Cell_ACDC.git

   * Now, you have remote repositories named:

     - ``upstream``, which refers to the original ``Cell_ACDC`` repository
     - ``origin``, which refers to your personal fork

   * Install the cloned Cell-ACDC in developer mode (i.e. editable) in a 
     virtual environment using ``venv`` or ``conda``:

     * ``venv`` (more info `here <https://docs.python.org/3/library/venv.html>`_)
  
       ::

         # Navigate to a folder where you want to create the virtual env
         cd ~/.virtualenvs

         # Create a virtual env with the name you like, e.g., ``acdc-dev``
         python -m pip venv acdc-dev

         # Activate the environment (on PowerShell replace ``source`` with dot ``.``)
         source ~/.virtualenvs/acdc-dev/bin/activate

         # Navigate to the cloned folder path (parent folder of ``cellacdc``)
         cd <path_to_Cell_ACDC>

         # Install Cell-ACDC in developer mode
         pip install -e .
  
     * ``conda`` (Anaconda, Miniconda, Miniforge etc.)

       ::

         # Create a virtual env with the name you like, e.g., ``acdc-dev``
         conda create -n acdc-dev python=3.10

         # Activate the environment
         conda activate acdc-dev

         # Navigate to the cloned folder path (parent folder of ``cellacdc``)
         cd <path_to_Cell_ACDC>

         # Install Cell-ACDC in developer mode
         pip install -e .

2. Develop your contribution:

   * Navigate to the cloned folder path (parent folder of ``cellacdc``)::
        
        cd <path_to_Cell_ACDC>
    
   * Pull the latest changes from upstream::

        git checkout main
        git pull upstream main

   * Create a branch with the name you prefer, such as 'new-segm-model'::

        git checkout -b new-segm-model

   * Commit locally as you progress (with ``git add`` and ``git commit``). Please write `good commit messages <https://vxlabs.com/software-development-handbook/#good-commit-messages>`_.

3. Submit your contribution through a Pull Request (PR):

   * Push your changes back to your fork on GitHub::

        git push origin new-segm-model

   * Go to GitHub. The new branch will show up with a green "pull request" button -- click it.
  
Note that if you want to modify the code of the Pull Request, you can simply 
commit and push to the same branch. GitHub will automatically add to the open 
Pull Request.


