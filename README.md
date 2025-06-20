![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# MNIST Handwritten Digit Deep Learning Accelerator ASIC
A deep learning accelerator ASIC chip design to classify images from the MNIST handwritten image dataset.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png" /><br>Source: Wikipedia - MNIST database</p>

Design implementation for Tiny Tapeout.

Thanks to [Columbus IEEE Joint Chapter of the Solid-State Circuits and Circuits and Systems Societies](https://r2.ieee.org/columbus-ssccas/)!

Example:

Input --> |  Serialized to ASIC --> | Neural Network --> | Output
:----------:|:---------------:|:--------------:|:--------------
<img src="https://github.com/estods3/mnist_accelerator/blob/main/docs/real_image0.png" title="Example MNIST Image Reduced to 14x14 Black/White Pixels" alt="drawing" width="100"/><br>MNIST Image | Cycle  Input Pin: 0123456 0123456<br>----------------------------------<br>0-1   0000000 0000000<br>2-3   0000000 0000000<br>4-5    0000000 0111100<br>6-7    0000111 1111100<br>8-9    0000111 1100000<br>10-11   0000011 0000000<br>12-13   0000001 1000000<br>14-15   0000000 1110000<br>16-17   0000000 0110000<br>18-19   0000001 1110000<br>20-21   0000011 1100000<br>22-23   0011111 0000000<br>24-25   0011110 0000000<br>26-27   0000000 0000000<br> | <img src="https://github.com/estods3/mnist_accelerator/blob/main/docs/nndiagram.png" title="Example Neural Net Graph" alt="drawing" width="200"/> | BCD: 0101 = 5<br><br>7-Segment:5

## MNIST Dataset + Preprocessing
Input images from the [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database) are preprocessed by a raspberry pi and transmitted to the ASIC. The images in MNIST are 28x28 grayscale images. However, as part of the preprocessing step, these images are reduced to a 14x14 black/white image to reduce the amount of data needed to be transmitted to the ASIC and to reduce the complexity of the neural network. Since the images are 14x14, a 8-pin interface (ui_in) is used which transmits 7 pixels at a time for 28 clock cycles to transmit each image. The remaining bit, the most significant bit (MSB), is a active-low signal. pulled low to start transmitting a new image.

### Preprocessing Python Script
A preprocessing python script (utility.py) is provided to convert the standard MNIST images into the reduced dataformat used in this project. The script is used to train the network, test the network, convert the pytorch implementation into verilog, and generate cocotb unit-tests directly from the MNIST dataset.

## Design

### Neural Network
Based on [MNIST pytorch example](https://github.com/pytorch/examples/tree/main/mnist). 

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  #1,32,3,1
        self.conv2 = nn.Conv2d(16, 32, 3, 1) #32,64,3,1
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(800, 128) #9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

### Hardware Implementation
Implemented into Verilog as a main file: project.v with 3 supporting files for readimage.v, neuralnetwork.v, and decoder.v

Latest GDS Rendering:
![Latest GDS Render](https://camo.githubusercontent.com/228b13205764a96e707eca359e2bbcf6d30f91d01d457b0facd95521e1a55917/68747470733a2f2f6573746f6473332e6769746875622e696f2f6d6e6973745f616363656c657261746f722f6764735f72656e6465722e706e67)


## Results
Goal - show results of chip vs identical python-based neural network implementation.



## Tiny Tapeout
- [FAQ](https://tinytapeout.com/faq/)
- [Digital design lessons](https://tinytapeout.com/digital_design/)
- [Learn how semiconductors work](https://tinytapeout.com/siliwiz/)
- [Join the community](https://tinytapeout.com/discord)
- [Build your design locally](https://www.tinytapeout.com/guides/local-hardening/)

## References
@ARTICLE{leflow,
     author = {{Noronha}, D.~H. and {Salehpour}, B. and {Wilton}, S.~J.~E.},
     title = "{LeFlow: Enabling Flexible FPGA High-Level Synthesis of Tensorflow Deep Neural Networks}",
     journal = {ArXiv e-prints},
     archivePrefix = "arXiv",
     eprint = {1807.05317},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
     year = 2018,
     month = jul,
     adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180705317N}
} 
- [Example Verilog TT Seven Segment Display](https://github.com/TinyTapeout/tt05-verilog-demo/blob/main/src/tt_um_seven_segment_seconds.v)
- [Example Verilog reading an image](https://www.edaboard.com/threads/reading-image-file-in-verilog.268155/)
- [Example PyTorch MNIST Neural Network](https://github.com/pytorch/examples/blob/main/mnist/main.py)
- [MNIST Database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
