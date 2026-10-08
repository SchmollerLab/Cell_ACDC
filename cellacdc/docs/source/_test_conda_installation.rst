Run the following command

.. code-block:: 

    conda

If you get the error message ``command not found: conda`` or ``conda : The term 'conda' is not recognized``, you first need 
to initialize ``conda``. To do so, follow the instructions below specific for your OS:

.. tabs::

        .. tab:: Windows

            On Windows, if ``conda`` cannot be found, it is likely you are 
            not using the **Miniforge prompt**. This is fine, but in that case 
            we recommend using the Powershell. You can find it by searching for it.
            Once you open the Powershell, to initialise ``conda``, you need to locate the ``miniforge3`` folder where conda was installed. This is typically 
            located in your user folder. You can test this by running this command 
            
            .. code-block:: 

                cd ~
            
            and then run the command ``ls`` to list all the folders present. 
            If you see the folder ``miniforge3`` in the list, then you found it. 
            If not, feel free to contact us and we will help you locating your Miniforge installation. 

            If you found it, run this command to initialize ``conda``
            
            .. code-block:: 

                ~\miniforge3\condabin\conda.bat init
            
            After that, restart the Powershell and see if the ``conda`` command 
            can be found.  
        
        .. tab:: macOS/Linux

            On macOS, you will very likely have to initialize ``conda``. To do so, you first need to locate the ``miniforge3`` folder where conda was installed. This is typically 
            located in your user folder. You can test this by running this command 
            
            .. code-block:: 

                cd ~
            
            and then run the command ``ls`` to list all the folders present. 
            If you see the folder ``miniforge3`` in the list, then you found it. 
            If not, feel free to contact us and we will help you locating your Miniforge installation. 

            If you found it, run this command to initialize ``conda``
            
            .. code-block:: 

                ~/miniforge3/bin/conda init zsh
        
            After that, restart the Terminal and see if the ``conda`` command 
            can be found. If you still get the error ``conda: command not found``, try this command:

            .. code-block:: 

                ~/miniforge3/bin/conda init