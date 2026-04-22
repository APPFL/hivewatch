========
Emitters
========

HiveWatch keeps the runtime API small by pushing output behavior into emitter
classes. You create one or more emitters and pass them to ``hivewatch.init()``.

SSE Emitter
===========

Use ``SSEEmitter`` when you want a local dashboard, replayable run artifacts,
and the geographic map view without depending on an external service.

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import SSEEmitter

   hw.init(
       algorithm="FedAvg",
       emitters=[SSEEmitter(port=7070, serve_map=False)],
   )

This emitter writes:

* ``runs/<run_id>.jsonl`` with the full event stream
* ``runs/<run_id>.map.json`` with map-ready metadata

Weights & Biases Emitter
========================

Use ``WandbEmitter`` when you want hosted experiment tracking, round metrics,
per-client metrics, and alert-style events inside W&B.

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import WandbEmitter

   hw.init(
       algorithm="FedAvg",
       emitters=[WandbEmitter(project="my-fl-project")],
   )

If a W&B run already exists in the current process, HiveWatch adopts it
instead of creating a duplicate run.

MLflow Emitter
==============

Use ``MLflowEmitter`` when you want self-hosted experiment tracking and
artifact logging.

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import MLflowEmitter

   hw.init(
       algorithm="FedAvg",
       emitters=[MLflowEmitter(
           tracking_uri="http://localhost:5000",
           experiment="my-fl-project",
       )],
   )

Start a local MLflow server if needed:

.. code-block:: bash

   mlflow server --host 0.0.0.0 --port 5000

Combined workflow
=================

HiveWatch can log to all three emitters in the same run:

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import MLflowEmitter, SSEEmitter, WandbEmitter

   hw.init(
       emitters=[
           SSEEmitter(serve_map=False),
           WandbEmitter(project="my-fl-project"),
           MLflowEmitter(experiment="my-fl-project"),
       ],
   )
