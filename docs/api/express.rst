Express API
===========

``xarp.express`` provides asynchronous and blocking interfaces for interacting
with a connected XR client.

Type aliases
------------

.. autodata:: xarp.express.ElementBatch

.. autodata:: xarp.express.AssetKeyBatch

Asynchronous client
-------------------

.. autoclass:: xarp.express.AsyncXR
   :members:
   :member-order: bysource

Synchronous client
------------------

.. autoclass:: xarp.express.SyncXR
   :members:
   :member-order: bysource

Streaming iterator
------------------

.. autoclass:: xarp.express.AsyncGeneratorIterator
   :members:
   :member-order: bysource

Image serving
-------------

.. autofunction:: xarp.express.serve_pil_image_ephemeral
