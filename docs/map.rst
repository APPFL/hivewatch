=============
Map Dashboard
=============

HiveWatch ships with a local dashboard flow based on saved run artifacts and a
small HTTP server.

Generate run artifacts
----------------------

Use ``SSEEmitter`` during training to create:

* ``runs/<run_id>.jsonl`` for the full event stream
* ``runs/<run_id>.map.json`` for map-friendly round and client metadata

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import SSEEmitter

   hw.init(emitters=[SSEEmitter(serve_map=False)])

Serve the dashboard
-------------------

.. code-block:: bash

   hivewatch map run --runs-dir runs --port 7070

Useful flags
------------

* ``--host`` changes the bind address.
* ``--map-path`` points at a custom HTML viewer.
* ``--poll-interval`` controls how often the runs directory is rescanned.

The bundled viewer in the ``examples/`` directory loads map metadata first and
falls back to the raw event history for older runs. That means the same viewer
works for both live monitoring and deferred replay.

Common map commands
-------------------

.. code-block:: bash

   # Watch a runs directory for live updates
   hivewatch map run --runs-dir runs --port 7070

   # Serve a custom viewer HTML file
   hivewatch map run --runs-dir runs --map-path /path/to/viewer.html

   # Bind the dashboard to a specific interface
   hivewatch map run --host 0.0.0.0 --runs-dir runs --port 7070

Deferred viewing
----------------

Because HiveWatch persists map metadata separately from the raw JSONL log, you
can train first and inspect later. This keeps a live workflow and a replay
workflow compatible with the same dashboard interface.
