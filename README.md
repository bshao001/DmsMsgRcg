# DmsMsgRcg

A photo OCR project aims to output DMS messages contained in sign structure images.

This project will provide an aided function to a well-established highway management software - Operations Task Manager 
(OTM) used in FDOT. The code will be implemented in Python with TensorFlow. By initial design, it will include the 
following 4 phases:

1. Text area detection. It is something like object detection. The purpose of this phase is to be able to correctly 
locate where the image areas are. It is possible to know how many message areas are when feeding a sign structure 
image. There are also hint information available about whether the message areas contain Toll information (which 
are mainly digits) or status information (which mostly be 26 upper-case English letters). However, the image may
contain other texts that can are same or similar words that should be discarded, for example, words - <em>EXPRESS 
LANES</em> or digits might appear in a text area we care or appear in the text area where we should ignore, with
very similar fonts.

2. Text type classification. This extra step tries to classify the cropped images from the first step into their 
corresponding types. This will rule out the need to recognize complicated Lane Status Messages.

3. Character segmentation. This will split the input image, which contains all the message characters, into individual 
character images.

4. Character recognition. In this stage, we will output a single character for each input image.

Finally, all the information gets re-organized.

There are training and prediction associated with each stage. The main idea is to use sliding windows for each stage, 
while the classifiers might be ConvNets, with possible model variations.

Due to copyright issue, the training images cannot be exposed here.
