============
Contributing
============

Set up a local environment
--------------------------

.. code-block:: bash

   git clone https://github.com/APPFL/appfl-log.git
   cd appfl-log
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev,docs]"

Run tests
---------

.. code-block:: bash

   pytest

Build docs
----------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

If the repository uses ``pre-commit``, run it before opening a pull request:

.. code-block:: bash

   pre-commit run --all-files
