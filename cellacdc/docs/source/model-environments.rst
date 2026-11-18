.. _model-environments:

Model environments and model servers (developer guide)
=======================================================

.. contents::
   :local:
   :depth: 2

This page explains how Cell-ACDC runs segmentation models in **isolated
environments**, how the code is organized, how to **add a new model**, and how
the model (server) side communicates requests to the user (client) side.

Overview
--------

Every segmentation model can run in its own Python environment, named
``_acdc_<model folder name>`` (for example ``_acdc_cellpose_v3``). Inside that
environment a small HTTP service (the *worker*) loads the model. The GUI, the
command line and the workflow kernel talk to it over HTTP.

.. code-block:: text

    Cell-ACDC (main environment: GUI, CLI)
       │   ModelClient  (same interface as acdcSegment.Model)
       ├── LocalBackend  → starts <model env>/python -m cellacdc.service.worker --server
       │                   and connects to http://127.0.0.1:<free port>
       └── RemoteBackend → connects to http://<server>:<port> (same worker code)

Running a model locally already looks like running it on a server. Moving it
to a remote machine only changes the URL. This gives three benefits:

* Model dependencies (PyTorch, TensorFlow, specific ``numpy`` versions, ...)
  never conflict with each other or with Cell-ACDC.
* Each model can use its own Python version.
* The same worker code runs locally and on a GPU server.

Execution modes
~~~~~~~~~~~~~~~

There are two separate, explicit execution paths. Cell-ACDC never silently
falls back from one to the other:

* **Isolated environment** (``USE_MODEL_ENV = True`` in
  ``cellacdc/service/model_execution.py``, the default). The model runs in its
  worker. If the environment cannot be created or the worker fails to start, a
  clear error is raised.
* **In-process** (``USE_MODEL_ENV = False``). The model is imported and run in
  the main Cell-ACDC environment, as in older versions.

If the packages of a model are already installed in the main environment,
Cell-ACDC asks (once per model and session) whether to use them or to set up
an isolated environment (recommended). Installed packages are detected from
package metadata only (``model_installed_in_main_env``); the model's
``__init__.py`` is never imported for this check.

Server mode
~~~~~~~~~~~

The worker is started with the ``--server`` command-line flag::

    python -m cellacdc.service.worker --server --model cellpose_v3 --port 8520

``cellacdc/__init__.py`` reads this flag at import time and sets
``cellacdc.RUNNING_AS_SERVER``. Use ``myutils.is_server_mode()`` to check it in
code. In server mode:

* Qt, ``cellacdc.widgets`` and ``cellacdc.apps`` are never imported. The model
  environment does not even have Qt installed.
* User configuration files are not read (an empty config is used).
* Nothing is installed and nobody is asked anything.
  ``myutils.check_install_package`` and ``check_install_torch`` raise a
  ``ModuleNotFoundError`` that says which package is missing and points to
  ``cellacdc/models/_env_config.py``. The client must prepare the environment
  *before* the worker starts (see `Communicating from the server side to the
  client side`_).

Code structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``cellacdc/models/_env_config.py``
     - **The only place for per-model environment configuration**: Python
       version, PyTorch, packages, additional setup commands and required
       environment variables. It is loaded by file path, so it must not import
       ``cellacdc``.
   * - ``cellacdc/service/environment.py``
     - Creates and updates model environments (conda if Cell-ACDC runs in
       conda, otherwise a venv in the Cell-ACDC user folder). Installs the
       lightweight Cell-ACDC runtime, the worker dependencies, PyTorch, the
       model packages and the ``setup_steps``, and asks for ``required_env``
       variables. Checks that the environment runs the client's Cell-ACDC
       build and repairs it otherwise (see `Same Cell-ACDC build in the
       client and the environment`_). Main entry point:
       ``get_or_create_model_env``.
   * - ``cellacdc/service/worker.py``
     - FastAPI worker running *inside* the model environment. It loads
       ``acdcSegment.Model`` and serves ``/health``, ``/version``, ``/spec``,
       ``/info``, ``/init``, ``/segment``, ``/segment3DT``, ``/add_prompt``,
       ``/second_ch_img_to_stack``, ``/predictCcaState``, ``/jobs/...`` and
       ``/shutdown``. Requests from clients with another protocol version are
       rejected (see `Protocol versions`_). A job's result
       can be fetched once (then the job is removed); ``DELETE /jobs/<id>``
       cancels a pending/running job or deletes a finished one, and only the
       newest finished jobs are kept in memory.
   * - ``cellacdc/service/client.py``
     - ``ModelClient`` (drop-in replacement for an ``acdcSegment.Model``
       instance), ``LocalBackend`` (starts and stops a local worker) and
       ``RemoteBackend`` (connects to a running worker).
       ``check_worker_version`` raises ``WorkerVersionError`` if a worker uses
       another protocol version.
   * - ``cellacdc/service/protocol.py``
     - ``PROTOCOL_VERSION`` of the HTTP protocol between client and worker,
       and the ``X-Cell-ACDC-Protocol`` header that carries it (see
       `Protocol versions`_).
   * - ``cellacdc/service/payload.py``
     - Encodes the arguments of every model call (init, segment, prompts)
       as one binary attachment: JSON plus an ``.npz`` archive with ``numpy``
       arrays and the bytes of uploaded files; ``pandas`` DataFrames are
       embedded in the JSON. Arguments that are paths of files on the
       client's disk are uploaded (see `Request format`_ and `5. Data instead
       of file paths`_).
   * - ``cellacdc/service/prompts.py``
     - Interface for asking the user something (``choice``, ``confirm``,
       ``text_input``, ``info``) independently of the user interface: Qt
       dialogs (``QtHandler``), the terminal (``CliHandler``) or defaults
       (``AutoHandler``).
   * - ``cellacdc/service/schema.py``
     - Pydantic models of the HTTP responses.
   * - ``cellacdc/service/setup_models.py``
     - ``acdc-setup-models`` command that prepares model environments ahead
       of time.
   * - ``cellacdc/myutils.py``
     - ``getModelArgSpecFromEnv`` provides standalone parameter inspection in
       a disposable process. GUI execution instead reads ``/spec`` from the
       persistent worker.
       ``resolve_segm_model_init_data`` loads the data that the GUI refers to
       by file name.
   * - ``cellacdc/service/model_execution.py``
     - ``ModelExecution``: shared by every GUI caller. ``prepare`` chooses
       the execution path, prepares the environment, starts a deferred-init
       worker and reads its parameters and help URL from ``/spec``.
       ``init_model`` asks for required variables and initializes that same
       process with ``POST /init`` (or creates the model in-process); ``release``
       gives an instance back and ``stop`` stops all workers.
       ``run_in_background`` runs slow steps in a ``QThread`` without
       freezing the GUI.
   * - ``cellacdc/segm.py``, ``cellacdc/gui.py``, ``cellacdc/workers.py``
     - GUI callers of ``ModelExecution``: the segmentation module
       (``segmWin``), the main GUI (``repeatSegm``, also used by the magic
       labeller, ``repeatSegmVideo`` and the YeastMate budding prediction
       ``autoAssignBud_YeastMate``) and segmentation for lost IDs.
       ``SegForLostIDsWorker`` runs in a ``QThread``, so it asks the GUI
       thread for the model with ``sigInitModel``; these models are reused
       until the settings or the position change.
       Each ``ModelExecution`` is cached per model for the whole GUI session,
       and all workers are stopped when the window closes.
   * - ``cellacdc/cli.py``
     - Workflow kernel. The ``backend`` section of a workflow file (``type``
       = ``in_process``, ``local`` or ``remote``; ``url``; ``env_python``)
       selects the backend. With an ``env_python``, the Cell-ACDC build of
       that environment is checked before the worker starts.

In isolated GUI execution, opening the parameter dialog imports the model
module but does not construct the model or load its weights. After confirmation
the same worker is initialized, so parameter inspection and inference share
the module cache. Cancelling the dialog discards only the uninitialized worker,
not previously initialized models. ``cancel_preparation`` provides this cleanup
for callers; ``stop`` also cleans up any prepared worker. Cached parameter specs
remain available after cancellation. If the worker exits or required environment
variables change before initialization, a new worker is started with the current
environment.

What happens when a model is selected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``ModelExecution.prepare`` checks the main environment (without imports)
   and, if needed, asks which environment to use.
2. ``get_or_create_model_env`` finds or creates ``_acdc_<model>`` and then,
   on every run (steps that are already done are skipped quickly):

   a. links or installs the lightweight Cell-ACDC runtime (Cell-ACDC with
      ``--no-deps`` plus ``WORKER_RUNTIME_DEPENDENCIES``, no GUI packages),
      and reinstalls it if the environment has another Cell-ACDC build (see
      `Same Cell-ACDC build in the client and the environment`_);
   b. installs the worker dependencies (FastAPI, uvicorn, ...);
   c. installs PyTorch if ``torch: True`` (asks which build: CPU or CUDA);
   d. installs ``packages``;
   e. runs ``setup_steps``.

3. ``LocalBackend.prepare`` starts the model environment's Python worker
   (``--server --defer-init``), checks its protocol version with
   ``/version`` (see `Protocol versions`_) and reads parameter metadata and
   the help URL from ``/spec``. The GUI builds the parameter dialog while this worker stays
   alive without constructing the model.
4. The user sets the parameters. ``ModelExecution.init_model`` then:

   a. asks for missing ``required_env`` variables;
   b. replaces file selections (auxiliary segmentation, points table) with the
      loaded data;
   c. gets a worker (see `Reusing running workers`_) and sends the init
      arguments with ``POST /init``.

5. Segmentation calls ``ModelClient.segment``. Images are sent as ``.npy``
   and the segment arguments as an attachment (see `Request format`_);
   labels are returned as ``.npy``.

Reusing running workers
~~~~~~~~~~~~~~~~~~~~~~~

Starting a worker is slow (new Python process, importing e.g. PyTorch,
loading the weights), so ``ModelExecution`` keeps workers running and reuses
them. For every ``init_model`` call it:

1. shares a worker whose model already has exactly the same parameters and
   data (compared with a fingerprint that includes array contents), so
   nothing is reloaded;
2. otherwise re-initializes an unused worker with ``POST /init``
   (``LocalBackend.reinitialize``). The process and its imports are kept;
   only the model is re-created. If this fails, the worker keeps its
   previous model. Callers pass the instance they replace as ``replace``
   (e.g. ``repeatSegm`` with the previous model), so its worker can be
   re-initialized;
3. otherwise initializes the worker retained from parameter inspection, or
   starts a new worker if none is available. Instances used at the same time with
   different parameters (e.g. segmentation and segmentation for lost IDs)
   run in separate workers.

Callers give back instances with ``release`` (or ``replace``). Released
workers keep running; at most ``MAX_IDLE_WORKERS`` (default 1) unused
workers are kept per model because they hold memory, also on the GPU. Workers
that exited are started again. All workers stop with ``ModelExecution.stop``
(when the GUI closes).

HTTP protocol between client and worker
---------------------------------------

Request format
~~~~~~~~~~~~~~

Every request carries the model arguments in **one binary attachment** (a
multipart file part), never in form fields: Starlette (FastAPI foundation), the web 
framework of the worker, limits form fields to 1 MiB, but not files. The 
attachment is built by ``payload.encode`` and read by ``payload.decode``
(``cellacdc/service/payload.py``):

.. code-block:: text

    8 bytes   length N of the JSON text (big-endian)
    N bytes   UTF-8 JSON with the arguments
    rest      optional .npz archive with the binary data that the JSON refers to

* ``numpy`` arrays are stored in the ``.npz`` archive;
* ``pandas`` DataFrames are embedded in the JSON (``orient='split'``);
* files on the client's disk are uploaded: their bytes are stored in the
  ``.npz`` archive (see `5. Data instead of file paths`_).

The keys in the ``.npz`` archive are generated (``array_N``, ``file_N``),
never argument names, so no argument can overwrite another one. The same
container, built by ``payload.encode_value_body``, carries SAM embeddings and
the worker's responses that contain nested values (for example labels
together with updated SAM embeddings).

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Endpoint
     - Multipart file parts
   * - ``/init``, ``/add_prompt``
     - ``kwargs``: the arguments (``payload.encode``).
   * - ``/segment``, ``/segment3DT``, ``/jobs/segment``
     - ``image``: the image saved with ``np.save`` (``.npy``) and
       ``params``: the arguments (``payload.encode``). For SAM embedding
       operations, ``/segment`` also receives ``embeddings``
       (``payload.encode_value_body``).

``LocalBackend`` always starts the worker with ``--defer-init`` and sends the
init arguments with ``POST /init`` (``post_init``); arguments are never passed
on the command line.

Protocol versions
~~~~~~~~~~~~~~~~~

Client and worker must use the same ``PROTOCOL_VERSION``
(``cellacdc/service/protocol.py``):

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Version
     - Changes
   * - 1
     - Arguments as JSON text in form fields.
   * - 2
     - Every argument in a binary attachment (see `Request format`_), files
       on the client's disk are uploaded automatically, ``GET /version``.

* The worker answers ``GET /version`` with ``{"protocol_version": ...,
  "cellacdc_version": ...}``.
* Clients send their version in the ``X-Cell-ACDC-Protocol`` header of every
  request. The worker rejects requests without a matching header with HTTP
  426, except ``/health``, ``/version``, ``/shutdown`` and the documentation
  pages. The interactive ``/docs`` page pre-fills the header.
* Clients call ``check_worker_version`` before they use a worker: in
  ``ModelClient.__init__``, in ``LocalBackend`` right after ``/health`` and in
  ``RemoteBackend.get_client`` before ``/init``. Workers of protocol 1 answer
  ``/version`` with HTTP 404 and count as protocol 1. A mismatch raises
  ``WorkerVersionError``.

Increase ``PROTOCOL_VERSION`` whenever a change to the endpoints or to the
request/response format makes older clients or workers incompatible (for
example a renamed or removed endpoint or part, a new required part, or another
encoding). Add the new version to the history in ``protocol.py`` and to the
table above.

Same Cell-ACDC build in the client and the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model environment runs exactly the Cell-ACDC build of the client.
``environment.ensure_cellacdc_cli`` (called when an environment is created or
repaired) and the combined check in ``_check_environment_dependencies`` (the
fast path of ``get_or_create_model_env``, see `Diagnosing slow environment
resolution`_) compare the environment's Cell-ACDC with the client's. The
protocol versions must always be equal; in addition:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Client's Cell-ACDC
     - The environment must have
   * - Source checkout
     - ``cellacdc`` imported from the same checkout (linked with a ``.pth``
       file).
   * - VCS install (``git+https://...``)
     - The same commit.
   * - Archive or wheel URL install
     - The same URL.
   * - Index (PyPI) install
     - The same version.

On a mismatch, ``environment.repair_cellacdc_runtime`` repairs the
environment:

* released and VCS builds are reinstalled with
  ``pip install --no-deps --force-reinstall <spec>``;
* for a source checkout, a Cell-ACDC copy installed inside the environment
  (which would hide the linked checkout) is uninstalled and the checkout is
  linked again.

Then the runtime dependencies are installed and the result is checked again.
A lock file in the environment prevents several processes (for example
cluster jobs) from repairing the same environment at the same time.

``LocalBackend`` additionally restarts a worker whose ``/version`` reports
another protocol, after repairing its environment. It does so only for
environments created by Cell-ACDC (``_acdc_<model>``, see
``environment.is_managed_model_env``) and never for the client's own
interpreter; for other environments it raises an error asking to install the
same Cell-ACDC version there. The command-line workflow runner
(``cellacdc/cli.py``, workflows with an ``env_python``) checks the build in
the same way before it starts the worker: it repairs managed environments and
warns about environments that the user manages.

.. note::

    Model environments are shared by all Cell-ACDC installations of the user.
    Alternating between two different Cell-ACDC builds therefore reinstalls
    Cell-ACDC in the environment at each switch.

Adding a new model
------------------

The steps below add a model called ``MyModel``. Look at the existing models in
``cellacdc/models`` for complete examples (``cellpose_v4`` and ``StarDist`` for
packages, ``YeastMate`` for ``setup_steps``, ``cellsam`` for
``required_env``, ``pomBseen_nuclear`` for auxiliary segmentation data).

1. Create the model folder
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``cellacdc/models/MyModel/``. The folder name is the model name shown in
the GUI and is used for the environment name (``_acdc_MyModel``).

2. Configure the environment in ``_env_config.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an entry to ``MODEL_ENVS`` in ``cellacdc/models/_env_config.py``. All fields
are optional. A model without an entry gets the Python version of the main
environment and no extra packages.

.. code-block:: python

    'MyModel': {
        # Python version of the environment (default: same as main env)
        'python': '3.11',

        # Install PyTorch with the CPU/CUDA build matching the machine
        'torch': True,

        # Packages, as pip names or as dicts with the arguments of
        # myutils.check_install_package
        'packages': [
            'some-package',
            {'pkg_name': 'mymodel', 'pypi_name': 'mymodel-seg',
             'import_pkg_name': 'mymodel',
             'min_version': '2.0', 'max_version': '3.0',
             'include_lower_version': True},
        ],

        # Optional: only used to detect an existing installation in the main
        # environment (defaults to 'packages')
        'main_env_packages': ['mymodel'],

        # Optional: additional commands run with the environment's Python
        'setup_steps': [
            {
                'name': 'mymodel from GitHub',
                'description': 'Install the development version of mymodel.',
                'check': 'import mymodel',  # skip the step if this succeeds
                'commands': [
                    ['-m', 'pip', 'install',
                     'git+https://github.com/me/mymodel.git'],
                ],
            },
        ],

        # Optional: environment variables needed at runtime (see below)
        'required_env': [
            {
                'name': 'MYMODEL_TOKEN',
                'description': 'Create a token at https://example.org.',
                'secret': True,
                'unless_init_kwargs': ['model_path'],
            },
        ],
    },

Notes:

* ``setup_steps`` run after PyTorch and ``packages``, in order. Each command is
  a list of arguments appended to the environment's Python. Commands never run
  through a shell. If ``check`` (Python code) succeeds, the step is skipped.
  After the step, ``check`` must succeed, otherwise setup fails with the output
  of the check. A step without ``check`` runs on every setup.
* Use ``setup_steps`` for anything that ``packages`` cannot express: installs
  from Git, ``--no-build-isolation`` builds, replacing ``opencv-python`` with
  ``opencv-python-headless``, version fixes, downloads, and so on.
* Never install packages from the model code itself in server mode. Everything
  must be declared here so that the client can prepare and confirm it.

3. Write ``__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~

``__init__.py`` is imported in both execution paths. Use
``myutils.check_install_package`` (it installs with confirmation in-process
and raises a clear error in server mode) instead of calling ``pip`` yourself:

.. code-block:: python

    from cellacdc import myutils

    myutils.check_install_package(
        'mymodel', pypi_name='mymodel-seg', import_pkg_name='mymodel'
    )

If ``__init__.py`` has custom installation code (dialogs, ``subprocess``),
guard it so that it never runs in the worker:

.. code-block:: python

    try:
        import mymodel
    except ModuleNotFoundError:
        from cellacdc import myutils
        if myutils.is_server_mode():
            myutils._raise_server_missing('mymodel (MyModel setup_steps)')
        ...  # in-process installation with Qt dialogs

Do not import Qt or ``cellacdc.widgets`` at module level in ``__init__.py`` or
``acdcSegment.py``. They are not installed in the model environment.

4. Write ``acdcSegment.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from cellacdc import _types

    class ModelType:
        values = ['cyto', 'nuclei']

    class Model:
        def __init__(self, model_type: ModelType = 'cyto', gpu: bool = False):
            import mymodel
            self.model = mymodel.load(model_type, gpu=gpu)

        def segment(self, image, diameter: float = 30.0):
            """Segment `image` and return integer labels of the same shape.

            Parameters
            ----------
            diameter : float
                Expected object diameter in pixels. Default is 30
            """
            return self.model.predict(image, diameter=diameter).astype(np.uint32)

    def url_help():
        return 'https://github.com/me/mymodel'

The GUI builds the parameter dialog from the signatures of ``__init__`` and
``segment``, their type hints and the docstring. Because the parameters are
read inside the model environment and sent as metadata, the type hints must
be types that can be transferred:

* ``bool``, ``int``, ``float``, ``str``, ``os.PathLike`` and ``None``.
* A class with a ``values`` list, shown as a drop-down. Optional attributes are
  ``is_exclusive_with`` and ``default_exclusive``. ``typing.Literal[...]`` also
  works.
* ``_types.Vector``, ``_types.FolderPath``, ``_types.CsvFilePath``,
  ``_types.YeazV2ModelType`` and ``_types.SecondChannelImage``.
* Classes with ``not_a_param = True`` (or ``_types.NotGUIParam``), which are
  hidden from the dialog.

To add a new custom widget type, create a Qt-free marker class in
``cellacdc/_types.py`` with ``isWidget = True`` that creates the widget lazily
in ``__init__`` (see ``CsvFilePath``). Then register it in
``_serialize_model_param_type`` and ``_deserialize_model_param_type`` in
``myutils.py``. Unregistered widget types raise a ``TypeError``. The widget
must implement ``value()``, ``setValue(value)`` and ``cast_dtype``.

Other conventions:

* ``segment`` may accept ``frame_i``, ``start_z_slice`` and similar arguments.
  The worker drops arguments that the method does not accept.
* Optional methods ``segment3DT``, ``second_ch_img_to_stack`` and
  ``predictCcaState`` are detected automatically and reported through
  ``/info``. ``ModelClient`` forwards them to the worker, so the model's own
  implementation always runs (never a reimplementation on the client).
* Validate inputs and raise clear errors (for example a wrong number of
  z-slices). The full error is shown to the user (see below).

5. Data instead of file paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A remote worker cannot read the client's files, so the client never sends its
own paths to the worker. It sends the data instead:

* If ``Model.__init__`` has a required positional argument after ``self``,
  the GUI shows an *Auxiliary segmentation file* selector. The client loads
  that file and sends the array as ``segm_data``. The worker calls
  ``Model(segm_data, **kwargs)``.
* An ``input_points_path`` parameter (``_types.CsvFilePath``) is loaded on the
  client and sent as the ``input_points_df`` DataFrame.
* Files on disk are uploaded automatically. A top-level argument of
  ``__init__``, ``segment``, ``segment3DT``, ``add_prompt`` or of a
  background job is uploaded if it is the path of an existing file on the
  client (for example a custom model's weights). The worker receives the path
  of a temporary copy with the same file name. For ``__init__`` the copy
  lives as long as the model; for the other calls it is deleted after the
  call.

An argument counts as a file only if it is unambiguous as a path: a
path-like object (e.g. ``pathlib.Path``), an absolute path string, or a
relative path string that contains a folder separator (e.g.
``models/weights.pth``). Bare words such as ``None``, ``point`` or ``cyto3``
are never treated as files, even if a file with that name exists in the
working directory. In addition:

* folders are not uploaded;
* only the file itself is uploaded, not the files next to it. A model that
  expects companion files next to the given path will not find them in the
  worker;
* path-typed init parameters (kinds ``path``, ``csv_path`` and ``folder``,
  i.e. ``os.PathLike``, ``_types.CsvFilePath`` and ``_types.FolderPath``)
  whose value was not uploaded are rejected by the worker with an error that
  names them.

All arguments, including arrays, DataFrames and uploaded files, are encoded
by ``cellacdc/service/payload.py`` (see `Request format`_). ``LocalBackend``
always starts the worker with ``--defer-init`` and sends the init arguments
with ``POST /init``; arguments are not passed as JSON on the command line.

To send other data, load it on the client (see
``myutils.resolve_segm_model_init_data``) and put the array or DataFrame into
the init arguments; ``payload`` transfers it automatically.

6. Prepare and test the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the environment and check the parameters::

    acdc-setup-models MyModel

or, from Python:

.. code-block:: python

    from cellacdc import myutils
    from cellacdc.service.environment import get_or_create_model_env
    from cellacdc.service.prompts import CliHandler

    python = get_or_create_model_env('MyModel', prompt_func=CliHandler().ask)
    init_params, segment_params, url = myutils.getModelArgSpecFromEnv(
        'MyModel', python
    )

Run it like the GUI does:

.. code-block:: python

    from cellacdc.service.client import LocalBackend

    backend = LocalBackend(
        'MyModel', env_python=python, init_model_kwargs={'gpu': False}
    )
    try:
        model = backend.start(timeout=300)  # returns a ModelClient
        labels = model.segment(image, diameter=30)
    finally:
        backend.stop()

``with LocalBackend(...) as model:`` also works; it starts the worker,
returns the ``ModelClient`` and stops the worker at the end.

The integration test installs and runs models on a sample image. Specify 
``"MyModel"`` to test only that model, or remove ``-k "MyModel"`` to test all 
models. Specify the sample image with the ``IMAGE_PATH`` variable inside
``tests\test_all_model_environments.py`` file to set target image. Do not forget
to activate your ``acdc`` environment.
It is opt-in because it downloads a lot of data::

    python -X faulthandler -m pytest .\tests\test_all_model_environments.py -v -s  -rs --full-trace -k "MyModel"

If your model needs special test arguments (for example a model file that
matches 2D data), add them in ``test_model_segments_sample_tiff`` in
``tests/test_all_model_environments.py``.

Communicating from the server side to the client side
-----------------------------------------------------

The worker never interacts with the user: it has no GUI and may run on another
machine. Anything the model needs from the user is therefore handled in one of
the following ways.

1. Declare it, and the client asks before starting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the preferred way. The model's needs are declared in
``_env_config.py``, and the client asks the user *before* the worker starts:

* ``torch: True``: the client asks which PyTorch build to install.
* ``packages`` and ``setup_steps``: the client installs them, streaming the
  output to the log. With ``auto_install=False``, each setup step asks for
  confirmation, showing its ``description``.
* ``required_env``: the client asks for the value (masked if ``secret``). It
  is stored only in the environment of the running Cell-ACDC process, never
  on disk or in logs, and the worker inherits it. With ``unless_init_kwargs``
  the variable is not required if the user set one of those parameters
  (for example a custom model file instead of a download token).

2. Ask through the prompt interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Client-side code that prepares a model must not create Qt dialogs directly.
Instead, it receives a ``prompt_func`` and describes the question as a
``Prompt``:

.. code-block:: python

    from cellacdc.service import prompts

    def ensure_my_license(model_name, prompt_func=None):
        if prompt_func is None:
            raise RuntimeError(f'{model_name}: license must be accepted.')
        accepted = prompt_func(prompts.confirm(
            'License', f'Do you accept the license of {model_name}?',
            details='Full license text ...',
        ))
        if not accepted:
            raise RuntimeError(f'{model_name}: license not accepted.')

The same code then works everywhere:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Handler
     - Used by
   * - ``QtHandler(parent).ask``
     - The GUI. It can be called from worker threads: the dialog is shown in
       the GUI thread, and the calling thread waits for the answer. It uses
       the standard Cell-ACDC dialogs: ``widgets.myMessageBox`` for
       ``info``, ``confirm`` and ``choice`` with up to three options (one
       button per option), ``apps.QDialogCombobox`` for longer choices,
       ``apps.QLineEditDialog`` for ``text_input`` and
       ``apps.InstallPyTorchDialog`` for ``dialog='pytorch'``.
   * - ``CliHandler().ask``
     - The terminal, including ``acdc-setup-models``.
   * - ``AutoHandler().ask``
     - Automatic runs and tests (``acdc-setup-models --yes``): returns the
       default answer.

Answers are: ``choice`` → selected label or ``None`` (cancelled), ``confirm``
→ ``bool``, ``text_input`` → ``str`` or ``None``, ``info`` → ``None``. Use
``extra={'secret': True}`` for masked text input. ``dialog`` is an optional
hint for a specialised Qt dialog (for example ``dialog='pytorch'``); handlers
that do not know the hint use the generic dialog.

Prompts are plain, JSON-serializable data (``Prompt.to_dict()`` /
``Prompt.from_dict()``). If a remote worker ever needs to ask during setup,
the same objects can be sent over HTTP and answered by the client's handler.

To add a new kind of request, add a field to the model configuration (and
document it in the ``_env_config.py`` docstring). Then handle it in
``cellacdc/service/environment.py`` with ``prompt_func``, and call it from the
client before the worker starts (``ModelExecution.init_model`` for the
GUI, ``get_or_create_model_env`` for environment setup).
``ensure_required_env`` is a complete example.

3. Report errors back
~~~~~~~~~~~~~~~~~~~~~

If the worker cannot continue, raise a clear exception in the model code:

* During ``/segment``, ``/segment3DT`` or ``/init``, the worker returns
  HTTP 500 with the full traceback. ``ModelClient`` raises a ``RuntimeError``
  that contains it, and the GUI shows it.
* If the worker exits during startup (for example a failing import in
  ``__init__.py``) or does not answer in time, ``LocalBackend.start`` raises an
  error with the command, the Python executable and the last lines of the
  worker output. On timeout, it also checks whether the worker module can be
  imported at all.
* Qt background tasks re-raise errors in the GUI thread. Model-launching GUI
  entry points must use ``exception_handler`` so initialization failures are
  logged and displayed in the standard error dialog rather than escaping a Qt
  callback and potentially terminating the application.
* Missing packages in server mode raise a ``ModuleNotFoundError`` that points
  to ``_env_config.py``. Fix the configuration instead of installing the
  package from the model code.

4. Report capabilities
~~~~~~~~~~~~~~~~~~~~~~

``GET /info`` reports the model name, whether a GPU is available, and whether
the model supports ``segment3DT``, ``second_ch_img_to_stack`` and
``predictCcaState``.
``ModelClient`` reads it when it connects and exposes, for example,
``is_segment3DT_available``.

Diagnosing slow environment resolution
--------------------------------------

Launch Cell-ACDC in debug mode (``acdc --debug`` or ``acdc -d``) to log
elapsed wall-clock times for main-environment metadata, environment discovery,
source linking, runtime imports, the Cell-ACDC version check, individual
dependency checks, PyTorch options discovery (only when installation is
needed), model-spec inspection (worker preparation and ``/spec`` in
``LocalBackend.prepare``), and worker startup including process launch,
waiting for ``/health``, payload encoding, ``/init``, and the final ``/info``
connection::

    acdc --debug

Timing lines start with ``[model startup]`` and report seconds and
``completed`` or ``failed``. Environment/dependency timings are emitted through
the setup logger (normally the terminal); workflow totals use the workflow
logger. Nested timings overlap and must not be added to their parent totals.
Timings are disabled outside debug mode; no separate environment variable is
needed.
Debug mode is also passed to local workers, which report model-module import,
argument inspection, and model construction timings through their output.

Reusing an existing environment does not invoke a package solver when all
dependency checks pass. Existing environments are checked in a single Python
process: runtime, torch/torchvision (when required), base dependencies, and model
dependencies share Python's import cache. Version constraints are still checked,
and so is the Cell-ACDC build that the environment imports (see `Same
Cell-ACDC build in the client and the environment`_).
If this combined check fails or finds another Cell-ACDC build, its diagnostic
is logged and the individual checks/installers repair the environment as before
(``ensure_cellacdc_cli`` reinstalls Cell-ACDC). Model-specific setup steps
are checked even after a successful combined check. New environments use the
normal installation sequence.

Environment validation still uses a separate process from the model worker.
GUI parameter inspection and inference, however, use the same persistent worker.
Large libraries such as PyTorch and TensorFlow can still dominate preparation
even when inference is fast.
PyTorch wheel indexes and CUDA installation choices are only discovered if
the target environment cannot import PyTorch and torchvision.

Running a model on a remote server
----------------------------------

1. On the server, prepare the environment (``acdc-setup-models MyModel``) and
   set the ``required_env`` variables of the model in the server environment.
   The server must run a Cell-ACDC version with the same ``PROTOCOL_VERSION``
   as the client (see `Protocol versions`_). Otherwise
   ``RemoteBackend.get_client`` (and workflows with ``type = remote``) raise
   an error asking to install the same Cell-ACDC version on the server.
2. Start the worker with the model environment's Python::

       <model env python> -m cellacdc.service.worker --server --model MyModel --host 0.0.0.0 --port 8520

   Add ``--defer-init`` to start it without a model and send the init
   arguments (including data and uploaded files) with ``POST /init``.
3. On the client:

   .. code-block:: python

       from cellacdc.service.client import RemoteBackend

       backend = RemoteBackend('http://gpu-server:8520')
       model = backend.get_client(init_model_kwargs={'gpu': True})
       labels = model.segment(image)

   Omit ``init_model_kwargs`` if the worker was started with its init
   arguments. In workflows, set the ``backend`` section to ``type = remote``
   and ``url = http://gpu-server:8520``.

.. warning::

    The worker has no authentication. Only expose it on trusted networks, or
    put it behind a reverse proxy with authentication.
