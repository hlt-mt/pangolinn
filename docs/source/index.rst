Welcome to *pangolinn* documentation
=====================================

**pangolinn** is a Python library for neural network developers
that contains test suites aimed at finding bugs (if any) in newly-created models.
Each test suite is dedicated to test a specific property of an architecture (model),
and it does not require pre-trained weights.

Check out the :doc:`installation` section for further information about how to install the project.

- **Github:** `https://github.com/hlt-mt/pangolinn.git <https://github.com/hlt-mt/pangolinn.git>`__
- **PyPi:** `https://pypi.org/project/pangolinn <https://pypi.org/project/pangolinn>`__

.. toctree::
   :hidden:

   installation

API Documentation
------------------

Here is the list of the modules currently part of the repository with
the corresponding documentation:

.. toctree::
   :maxdepth: 2

   sequence_to_sequence_api

Credits
________


If you use this library, please cite:

.. code-block::

  @inproceedings{Papi2023ReproducibilityIN,
    title={{Reproducibility is Nothing without Correctness: The Importance of Testing Code in NLP}},
    author={Sara Papi and Marco Gaido and Andrea Pilzer and Matteo Negri},
    year={2023}
  }
