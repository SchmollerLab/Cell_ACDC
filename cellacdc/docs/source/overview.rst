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

See `the table below <#comparison_table>`_ **how it compares** to other popular tools available (*Table 1
of
our* \ `publication <https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6>`__).

.. raw:: html

   <style>
   .rotated-header-cell {
      width: 40px;
      height: 100px;
      padding: 0;
      position: relative; /* Needed as anchor for absolute positioning */
      vertical-align: bottom;
      text-align: center;
   }

   .rotated-text {
      position: absolute;
      bottom: 50%; /* Align near bottom of the cell */
      left: 70%;
      transform: translateX(-50%) rotate(-75deg);
      transform-origin: bottom center;
      white-space: nowrap;
      line-height: 1;
      font-weight: normal;
      padding: 0;
      margin: 0;
      width: max-content;
      user-select: none;
   }

   /* Optional: fix first header cell vertical alignment */
   th:first-child {
      vertical-align: bottom;
      padding-bottom: 10px;
   }
   </style>

   <table border="1" cellspacing="0" cellpadding="6">
   <thead>
      <tr style="height: 100;">
         <th style="text-align: center; vertical-align: bottom;">Feature</th>
         <th class="rotated-header-cell"><div class="rotated-text">Cell-ACDC</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">YeaZ</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">Cellpose</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">YeastMate</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">DeepCell</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">PhyloCell</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">CellProfiler</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">ImageJ/Fiji</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">YeastSpotter</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">YeastNet</div></th>
         <th class="rotated-header-cell"><div class="rotated-text">MorphoLibJ</div></th>
      </tr>
   </thead>
   <tbody>
      <tr><td>Deep-learning segmentation</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td></tr>
      <tr><td>Traditional segmentation</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
      <tr><td>Tracking</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Manual corrections</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
      <tr><td>Automatic real-time tracking</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Automatic propagation of corrections</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Automatic mother-bud pairing</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Pedigree annotations</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Cell division annotations</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>Downstream analysis</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td></tr>
      <tr><td>3D z-stacks</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
      <tr><td>Multiple model organisms</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
      <tr><td>Bio-formats</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
      <tr><td>User manual</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
      <tr><td>Open source</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
      <tr><td>Does not require a licence</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
   </tbody>
   </table>

|

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