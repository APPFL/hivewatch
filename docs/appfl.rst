=================
APPFL Integration
=================

HiveWatch is designed to work with APPFL without forcing invasive client-side
changes. A common pattern is to intercept server-side updates, emit client
metadata, and record the aggregated round summary.

Basic pattern
-------------

.. code-block:: python

   import hivewatch as hw
   from hivewatch.emitters import WandbEmitter

   hw.init(
       algorithm="FedAvg",
       emitters=[WandbEmitter(project="my-fl-project")],
   )

Then integrate logging inside the APPFL server flow:

.. code-block:: python

   hw.log_client_update(
       client_id=client_id,
       round=round_num,
       local_accuracy=val_accuracy,
       local_loss=val_loss,
       bytes_sent=bytes_sent,
   )

   hw.log_round(
       round=round_num,
       global_accuracy=global_accuracy,
       global_loss=global_loss,
   )

What to emit
------------

HiveWatch understands common FL metadata such as:

* ``local_accuracy`` and ``local_loss``
* ``num_samples``
* ``gradient_norm``
* ``bytes_sent`` and ``bytes_received``
* ``train_time_sec``
* ``cpu_pct``, ``ram_mb``, and GPU metrics
* ``lat``, ``lng``, ``city``, and ``country`` for map views

Unknown keys are preserved in the client payload, so you can attach
framework-specific metadata without breaking the core schema.
