Gestures API
============

``xarp.gestures`` provides joint indices, finger chains, and stateless metrics
for tracked hands. Distances are measured in metres.

Joint indices
-------------

The indices address poses in a tracked hand tuple.

.. list-table::
   :header-rows: 1
   :widths: 32 12 32 12

   * - Constant
     - Index
     - Constant
     - Index
   * - ``PALM``
     - 0
     - ``WRIST``
     - 1
   * - ``THUMB_METACARPAL``
     - 2
     - ``THUMB_PROXIMAL``
     - 3
   * - ``THUMB_DISTAL``
     - 4
     - ``THUMB_TIP``
     - 5
   * - ``INDEX_METACARPAL``
     - 6
     - ``INDEX_PROXIMAL``
     - 7
   * - ``INDEX_INTERMEDIATE``
     - 8
     - ``INDEX_DISTAL``
     - 9
   * - ``INDEX_TIP``
     - 10
     - ``MIDDLE_METACARPAL``
     - 11
   * - ``MIDDLE_PROXIMAL``
     - 12
     - ``MIDDLE_INTERMEDIATE``
     - 13
   * - ``MIDDLE_DISTAL``
     - 14
     - ``MIDDLE_TIP``
     - 15
   * - ``RING_METACARPAL``
     - 16
     - ``RING_PROXIMAL``
     - 17
   * - ``RING_INTERMEDIATE``
     - 18
     - ``RING_DISTAL``
     - 19
   * - ``RING_TIP``
     - 20
     - ``PINKY_METACARPAL``
     - 21
   * - ``PINKY_PROXIMAL``
     - 22
     - ``PINKY_INTERMEDIATE``
     - 23
   * - ``PINKY_DISTAL``
     - 24
     - ``PINKY_TIP``
     - 25

Finger chains
-------------

``THUMB``, ``INDEX``, ``MIDDLE``, ``RING``, and ``PINKY`` contain the ordered
joint indices for each finger. ``FINGERS`` contains all five chains, while
``DIGITS`` contains the four non-thumb chains.

Finger metrics
--------------

.. autofunction:: xarp.gestures.finger_extension

.. autofunction:: xarp.gestures.finger_flexion

.. autofunction:: xarp.gestures.palm_normal

Pinch gestures
--------------

.. autofunction:: xarp.gestures.pinch

.. autofunction:: xarp.gestures.pinch_middle

.. autofunction:: xarp.gestures.pinch_ring

.. autofunction:: xarp.gestures.double_pinch

Pose gestures
-------------

.. autofunction:: xarp.gestures.fist

.. autofunction:: xarp.gestures.open_hand

.. autofunction:: xarp.gestures.point

.. autofunction:: xarp.gestures.victory

.. autofunction:: xarp.gestures.thumbs_up

.. autofunction:: xarp.gestures.flat_palm

.. autofunction:: xarp.gestures.coarse_grab

.. autofunction:: xarp.gestures.index_thumb_l
