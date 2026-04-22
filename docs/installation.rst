============
Installation
============

This page describes how to install HiveWatch from the repository checkout and
how to prepare the documentation toolchain.

Base installation
-----------------

For the core package:

.. code-block:: bash

   pip install -e .

Optional emitters
-----------------

Install only the integrations you need:

.. code-block:: bash

   pip install -e ".[wandb]"
   pip install -e ".[mlflow]"
   pip install -e ".[all]"

Developer and docs installation
-------------------------------

For local development and documentation builds:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev,docs]"

Build the docs locally
----------------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Then open ``docs/_build/html/index.html`` in your browser.
