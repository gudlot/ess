.. _release-notes:

Release Notes
=============

Since v0.4
----------

Features
~~~~~~~~


Breaking changes
~~~~~~~~~~~~~~~~

* Large refactor of the wave-frame multiplication submodule `#42 <https://github.com/scipp/ess/pull/42>`_:

  * It was moved out of the ``v20`` submodule and into its own ``wfm`` submodule.
  * The ``get_frames`` function now operates on a dataset that contains the data and the instrument geometry.
  * The mechanism for finding the frames has changed and is now using the full description in `Schmakat et al. (2020) <https://www.sciencedirect.com/science/article/pii/S0168900220308640>`_.
  * The plotting inside ``get_frames`` has been moved into its own function ``wfm.plot.time_distance_diagram()`` which then calls ``get_frames`` internally.
  * The stitching now automatically replaces the position of the source with the mid-point between the choppers.
  * The stitching can now either return a single data array where all frames have been rebinned onto a common axis (the old behaviour), or a dict containing the individual frames. To obtain the latter, use the ``merge_frames=False`` argument.
  * The plotting has been removed from the stitching. Users should manually plot the dict of frames instead, which is obtained by using ``merge_frames=False``.

Bugfixes
~~~~~~~~

Contributors
~~~~~~~~~~~~

Owen Arnold :sup:`b, c`\ ,
Simon Heybrock :sup:`a`\ ,
Matthew D. Jones :sup:`b, c`\ ,
Andrew McCluskey :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`\


Contributing Organizations
--------------------------
* :sup:`a`\  `European Spallation Source ERIC <https://europeanspallationsource.se/>`_, Sweden
* :sup:`b`\  `Science and Technology Facilities Council <https://www.ukri.org/councils/stfc/>`_, UK
* :sup:`c`\  `Tessella <https://www.tessella.com/>`_, UK
