.. _gluoncv-model-zoo-action_recognition:

Action Recognition
==================

.. role:: greytag

Table of pre-trained models for video action recognition and their performance.

.. hint::

  Training commands work with this script:
  :download:`Download train_recognizer.py<../../scripts/action-recognition/train_recognizer.py>`

  A model can have differently trained parameters with different hashtags.
  Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

  - Download default pretrained weights: ``net = get_model('inceptionv3_ucf101', pretrained=True)``

  - Download weights given a hashtag: ``net = get_model('inceptionv3_ucf101', pretrained='0c453da8')``

  The test script :download:`Download test_recognizer.py<../../scripts/action-recognition/test_recognizer.py>` can be used for
  evaluating the models.

.. role:: tsntag

UCF101 Dataset
--------------

The following table lists pre-trained models trained on UCF101.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks" [2]_ . Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of UCF101 dataset, not the average of 3 splits.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

.. table::
    :widths: 45 10 10 10 25

    +---------------------------------------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        | Top-1     | Hashtag   | Train Command                                                                                                                                  | Train Log                                                                                                                              |
    +=============================================+===========+===========+================================================================================================================================================+========================================================================================================================================+
    | vgg16_ucf101 [2]_                           | 83.4      | d6dc1bba  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.sh>`_            | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.log>`_            |
    +---------------------------------------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | :tsntag:`vgg16_ucf101` [1]_                 | 81.5      | 05e319d4  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.log>`_                |
    +---------------------------------------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_ucf101 [2]_                     | 88.1      | 13ef5c3b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.log>`_      |
    +---------------------------------------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    | :tsntag:`inceptionv3_ucf101` [1]_           | 85.6      | 0c453da8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.log>`_          |
    +---------------------------------------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+

Kinetics400 Dataset
-------------------

The following table lists pre-trained models trained on Kinetics400.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks" [2]_ . Please check the reference paper for further information.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | inceptionv3_kinetics400 [2]_                |   ImageNet       |      3       |       1        | 72.5      | 8a4a6946  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+


.. [1] Limin Wang, Yuanjun Xiong, Zhe Wang, and Yu Qiao. \
       "Towards Good Practices for Very Deep Two-Stream ConvNets." \
       arXiv preprint arXiv:1507.02159 (2015).
.. [2] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang and Luc Van Gool. \
       "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition." \
       In European Conference on Computer Vision (ECCV). 2016.
