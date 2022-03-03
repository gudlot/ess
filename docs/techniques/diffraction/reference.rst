.. currentmodule:: ess.diffraction

Reference
=========

Powder
------

Conversions
~~~~~~~~~~~

.. autosummary::
   :toctree: ../../generated

   powder.conversions.dspacing_from_diff_calibration
   powder.to_dspacing_with_calibration

Corrections
-----------

.. autosummary::
   :toctree: ../../generated

   merge_calibration
   normalize_by_monitor
   normalize_by_vanadium
   subtract_empty_instrument

Loading
-------

.. autosummary::
   :toctree: ../../generated

   load_and_preprocess_vanadium

Smoothing
-----------

.. autosummary::
   :toctree: ../../generated

   fft_smooth
