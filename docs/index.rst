=========
HiveWatch
=========

HiveWatch is a lightweight observability toolkit for federated and distributed
machine learning workloads. It helps you capture round-level metrics, client
updates, communication costs, and geographic activity on a live map without
forcing you into a single training framework.

With HiveWatch you can:

* log client and round metrics from custom training loops,
* stream and replay runs locally with the built-in map dashboard,
* send metrics to Weights & Biases, MLflow, or both, and
* integrate observability into APPFL workflows with minimal glue code.

Technical Components
====================

HiveWatch is organized around a few clear technical components.

.. grid:: 3

   .. grid-item-card::

      Runtime API
      ^^^^^^^^^^^
      A small instrumentation API for logging round starts, client updates,
      round summaries, failures, and checkpoints.

   .. grid-item-card::

      SSE Emitter
      ^^^^^^^^^^^
      Streams local events, writes replayable run artifacts, and powers the
      built-in map dashboard workflow.

   .. grid-item-card::

      WandB Emitter
      ^^^^^^^^^^^^^
      Sends round metrics, per-client metrics, and alert-style events to
      Weights & Biases.

.. grid:: 3

   .. grid-item-card::

      MLflow Emitter
      ^^^^^^^^^^^^^^
      Logs tracking metrics, tags, and artifacts to local or remote MLflow
      servers.

   .. grid-item-card::

      Map Dashboard
      ^^^^^^^^^^^^^
      Visualizes client geography and run progress from saved ``.jsonl`` and
      ``.map.json`` artifacts.

   .. grid-item-card::

      APPFL Integration
      ^^^^^^^^^^^^^^^^^
      Fits into APPFL server-side orchestration so metrics can be emitted
      during real federated runs.

Main Topics
===========

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   emitters
   map
   appfl
   contributing
   changelog
