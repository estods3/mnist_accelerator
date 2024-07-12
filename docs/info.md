<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works
This project is for a neural network accelerator ASIC for text recognition in MNIST handwritten digit dataset.
Due to constraints with fab (Tiny Tapeout), the following goals have been outlined.

Goals:

* classify at least 2 of the 10 digits (Ex. '3' and '5')

* recall of at least 0.8

* increase performance (speed) to classify image (benchmark: python-based neural net on a raspberry pi.)

Diagram:
<pre>
 _________________                                                _________                        _________
|                 |                                              |         |--Output: seg a ----->|         |
|  Raspberry Pi   |--Input: Pre-processed MNIST Image (7 pins)-->|   Tiny  |--Output: seg b ----->|  Seven  |
|                 |                                              | Tapeout |--Output: seg c------>|   Seg   |
|                 |                                              |   ASIC  |--Output: seg d ----->| Display |
|                 |                                              |         |--Output: seg e ----->|         |
|                 |<---------------------- Output: BCD (4 pins)--|         |--Output: seg f ----->|         |
|_________________|                                              |_________|--Output: seg g ----->|_________|
</pre>
The Raspberry Pi will hold all of the images from the test set. It will loop through each image, preprocess it (to reduce data size), and deliver it to the ASIC. the ASIC will then return the classification back. This process is repeated for all images in the test set.

The chip's logic is laid out into 4 main components:
<pre>
   _________________________________________________________________________
   |   _________         __________         __________         __________    |
   |   |       |         |        |         |        |         |        |    |
  \/   |  I/O  |         | Memory |         |   NN   |         | Output |    |
------>|       | ----->  |        | ----->  |        | ----->  |        | --->
       |       |         |        |         |        |         |        |
       |_______|         |________|         |________|         |________|
</pre>

## I/O
Wait to recieve image. Read 14x14 pre-processed image into memory by reading 7 pixels at a time, each pixel is 1 bit (black=0, white=1). This will take 14x14=196/7 = 28 clock cycles to read the image.

## Memory
stores recieved pixels until all 196 pixels of the image have been recieved. Memory will only be able to hold one image at a time.

## Neural Network
196 pixels form the input layer of the neural network. Return the classification of the image as a binary coded decimal

## Output
Display the decoded digit on 7 segment display. This section incorporates logic to decode a binary coded decimal (BCD) into 7 segment logic for external 7 segment display.
Additionally, the output layer should trigger an additional digital pin as a flag to signal the image has finished being processed and another image can be sent from the RaspberryPi.

## How to test

Change inputs to the desired numbers to AND together

## External hardware
A RaspberryPi is needed to preprocess the MNIST images and send them in the correct format to the tiny tapeout ASIC. 
A Seven Segment Display can be used to display the output classification from the deep learning accelerator.
