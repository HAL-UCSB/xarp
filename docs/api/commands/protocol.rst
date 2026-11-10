Protocol Models
===============

Message and response modes
--------------------------

.. autoclass:: xarp.commands.MessageType
   :members:

.. autoclass:: xarp.commands.ResponseMode
   :members:

Incoming messages
-----------------

.. autoclass:: xarp.commands.Notification

.. autoclass:: xarp.commands.SingleResponse

.. autoclass:: xarp.commands.StreamResponse

.. autodata:: xarp.commands.IncomingMessage

.. autodata:: xarp.commands.IncomingMessageValidator

.. autodata:: xarp.commands.Response

Outgoing commands
-----------------

.. autoclass:: xarp.commands.Command
   :members: validate_response_value
   :member-order: bysource

.. autoclass:: xarp.commands.Bundle
   :members: model_dump, validate_response_value
   :member-order: bysource

.. autoclass:: xarp.commands.Cancel
