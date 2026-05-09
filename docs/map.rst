=============
Map Dashboard
=============

HiveWatch ships with a local dashboard flow based on saved run artifacts and a
small HTTP server. The same viewer supports live monitoring and replay of
completed runs.

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

Without ``--run-id``, the server watches the run directory and publishes new
JSONL events as they arrive.

Useful flags
------------

* ``--host`` changes the bind address.
* ``--run-id`` opens one saved run in static replay mode.
* ``--map-path`` points at a custom HTML viewer.
* ``--poll-interval`` controls how often the runs directory is rescanned.

The bundled viewer loads ``.map.json`` metadata first and falls back to the raw
event history for older runs.

Common map commands
-------------------

.. code-block:: bash

   # Watch a runs directory for live updates
   hivewatch map run --runs-dir runs --port 7070

   # Open one completed run in static replay mode
   hivewatch map run --runs-dir runs --run-id run-abc123

   # Serve a custom viewer HTML file
   hivewatch map run --runs-dir runs --map-path /path/to/viewer.html

   # Bind the dashboard to a specific interface
   hivewatch map run --host 0.0.0.0 --runs-dir runs --port 7070

Deferred viewing
----------------

Because HiveWatch persists map metadata separately from the raw JSONL log, you
can train first and inspect later. This keeps a live workflow and a replay
workflow compatible with the same dashboard interface.

Client metadata in the map
--------------------------

The sidebar shows client fields from the run artifacts. Geo and system identity
fields such as ``lat``, ``lng``, ``city``, ``country``, ``ip``, and
``client_id`` are used internally or omitted from the card. Other scalar
metadata is displayed automatically.

Use a leading underscore for metadata that should be preserved but hidden from
the bundled viewer:

.. code-block:: python

   hw.log_client_update(
       client_id="client-1",
       round=round_num,
       current_local_steps=200,
       blocking=True,
       _debug_score=0.92,
   )

The map also includes a draggable event log. Drag it by the header to move it;
double-click the header to reset its saved position.
