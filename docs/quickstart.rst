===========
Quick Start
===========

This example shows the smallest useful HiveWatch setup: log a run with the
local SSE emitter, persist the data, and serve the dashboard separately.

Minimal example
---------------

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import SSEEmitter

   hw.init(
       algorithm="FedAvg",
       emitters=[SSEEmitter(port=7070, serve_map=False)],
   )

   for round_num in range(5):
       hw.round_start(round_num)

       hw.log_client_update(
           client_id="client-1",
           round=round_num,
           local_accuracy=0.70 + round_num * 0.03,
           local_loss=0.90 - round_num * 0.08,
           num_samples=500,
           bytes_sent=8192,
           lat=41.88,
           lng=-87.63,
           country="US",
       )

       hw.log_round(
           round=round_num,
           global_accuracy=0.74 + round_num * 0.03,
           global_loss=0.85 - round_num * 0.07,
       )

   hw.finish()

This creates two run artifacts under ``runs/``:

* ``<run_id>.jsonl`` with the full event stream.
* ``<run_id>.map.json`` with map-ready metadata for replay and deferred viewing.

Serve the dashboard
-------------------

.. code-block:: bash

   hivewatch map run --runs-dir runs --port 7070

Open ``http://localhost:7070`` to inspect the run.

Switch emitters
---------------

HiveWatch can attach more than one emitter to the same run:

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import MLflowEmitter, WandbEmitter

   hw.init(
       emitters=[
           WandbEmitter(project="my-fl-project"),
           MLflowEmitter(experiment="my-fl-project"),
       ],
   )
