# DmsMsgRcg

![](https://img.shields.io/badge/python-3.6.2-brightgreen.svg)  ![](https://img.shields.io/badge/tensorflow-1.4.0-yellowgreen.svg?sanitize=true)

A photo OCR project aims to recognize and output DMS messages contained in sign structure images.

## Project Details
This project will provide an aided function to a well-established highway management software - Operations Task Manager 
(OTM) used in FDOT. The code is and will be implemented in Python with TensorFlow (and tf.keras in TF 1.4). The images
in this project are all of the same size (height * width = 480 * 640) and same format (in JPG). By design, the pipeline 
will include the following 4 phases or steps:

#### 1. Text area detection 
It is like object detection. This step try to locate where the text areas are. YOLO algorithm is slightly modified to 
adapt the need of this step. It is possible to know how many message areas are when feeding a sign structure image. 
There are also hint information available about whether the message areas contain Toll information (which are mainly 
digits) or status information (which mostly be 26 upper-case English letters). However, the image may contain other 
texts that can are same or similar words that should be discarded, for example, words - <em>EXPRESS LANES</em> or 
digits might appear in a text area we care or appear in the text area where we should ignore, with very similar fonts. 

You can find a sliding window version implementation for this step in the other branch in this repository. The training 
examples are generated both manually and by using the old text detector. A python script to create labels is included. 
It requires bounding boxes (with pure red, i.e. RGB = 255, 0, 0 and thickness = 1) be drawn on the images, which are then
save in PNG format.

The code is completed for this step.

#### 2. Text type classification 
This extra step tries to classify the cropped images from the first step into their corresponding types. This will rule 
out the need to recognize complicated Lane Status Messages.

Due to the new introduction of TensorFlow Keras, this step and the following two steps are under development.

#### 3. Character segmentation 
This will split the input image, which contains all the message characters, into individual character images.

#### 4. Character recognition
In this stage, we will output a single character for each input image.

Finally, all the information gets re-organized.

There are training and prediction associated with each stage. Theoretically, step 1 and 2 can be combined, however, 
that would require much more training examples, and the labelling process would be much more complicated.

Due to copyright issue, the training images cannot be exposed here.

## References:
1. https://github.com/experiencor/basic-yolo-keras
