Overview
========

Let’s face it, when dealing with segmentation of microscopy data we
often do not have time to check that **everything is correct**, because
it is a **tedious** and **very time consuming process**. Cell-ACDC comes
to the rescue! We combined the currently **best available neural network
models** (such as `Segment Anything Model
(SAM) <https://github.com/facebookresearch/segment-anything>`__,
`YeaZ <https://www.nature.com/articles/s41467-020-19557-4>`__,
`cellpose <https://www.nature.com/articles/s41592-020-01018-x>`__,
`StarDist <https://github.com/stardist/stardist>`__,
`YeastMate <https://github.com/hoerlteam/YeastMate>`__,
`omnipose <https://omnipose.readthedocs.io/>`__,
`delta <https://gitlab.com/dunloplab/delta>`__,
`DeepSea <https://doi.org/10.1016/j.crmeth.2023.100500>`__, etc.) and we
complemented them with a **fast and intuitive GUI**.

We developed and implemented several smart functionalities such as
**real-time continuous tracking**, **automatic propagation** of error
correction, and several tools to facilitate manual correction, from
simple yet useful **brush** and **eraser** to more complex flood fill
(magic wand) and Random Walker segmentation routines.

See below **how it compares** to other popular tools available (*Table 1
of
our* \ `publication <https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6>`__).

.. image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/figures/Table1.jpg
  :width: 700

Is it only about segmentation?
------------------------------

Of course not! Cell-ACDC automatically computes **several single-cell
numerical features** such as cell area and cell volume, plus the mean,
max, median, sum and quantiles of any additional fluorescent channel’s
signal. It even performs background correction, to compute the **protein
amount and concentration**.

You can load and analyse single **2D images**, **3D data** (3D z-stacks
or 2D images over time) and even **4D data** (3D z-stacks over time).

Finally, we provide Jupyter notebooks to **visualize** and interactively
**explore** the data produced.

**Do not hesitate to contact me** here on GitHub (by opening an issue)
or directly at my email padovaf@tcd.ie for any problem and/or feedback
on how to improve the user experience!

Bidirectional microscopy shift error correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is every second line in your files from your bidirectional microscopy
shifted? Look
`here <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/scripts/README.md>`__
for further information on how to correct your data.