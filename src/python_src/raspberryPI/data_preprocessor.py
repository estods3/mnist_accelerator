from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR

# Image Preprocessor
# desc - function to preprocess MNIST images for use with ASIC or Python(pytorch) NN
#
def preprocessor():
    image_threshold = 0
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Resize(size=(14,14), antialias=True),             # Preprocessing Step: Reduce Image size
    lambda x: x>image_threshold,                 # Preprocessing Step: Convert grayscale to Binary Image (Source: https://discuss.pytorch.org/t/binarize-image-on-training/12576/3)
    lambda x: x.float() #,
    #transforms.Normalize(mean=[-0.5],std=[0.5])
    ])
    return transform

# Test Case Writer Function
# desc - called for each row (individual test) of the dataframe.
#
def test_writer(row):
    with open("randomtests.py", "a") as testcasefile:
        testcasefile.write("@cocotb.test()\n")
        testcasefile.write("async def test_batch" + str(row["batch"])+"_sample"+str(row["sample"])+"(dut):\n")
        testcasefile.write("    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py\n")
        testcasefile.write("    # Test: Batch="+ str(row["batch"])+", Sample="+str(row["sample"])+"\n")
        testcasefile.write("    # Author: estods3\n")
        testcasefile.write("    # Input: described in 'input_image'\n")
        testcasefile.write("    # Expected Result: BCD = " + str(row["label"]) + "\n")
        testcasefile.write("    # --------------------------------------------" + "\n")
        testcasefile.write("    input_image = [LogicArray(\"" + row["data vector"][0:14] + "\"), \\" + "\n")
        for ind in range(14, len(row["data vector"]), 14):
            testcasefile.write("                   LogicArray(\"" + row["data vector"][ind:ind+14] + "\"), \\" + "\n")
        testcasefile.write("    ]\n")
        testcasefile.write("\n")
        testcasefile.write("    classification_result = " + str(row["label"]) + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # PERFORM TEST" + "\n")
        testcasefile.write("    # ------------" + "\n")
        testcasefile.write("    # Set the clock period to 10 us (100 KHz)" + "\n")
        testcasefile.write("    clock = Clock(dut.clk, 10, units='us')" + "\n")
        testcasefile.write("    cocotb.start_soon(clock.start())" + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # Initial Conditions" + "\n")
        testcasefile.write("    dut.ena.value = 1" + "\n")
        testcasefile.write("    dut.ui_in.value = 128" + "\n")
        testcasefile.write("    dut.uio_in.value = 0" + "\n")
        testcasefile.write("    dut.rst_n.value = 0" + "\n")
        testcasefile.write("    await ClockCycles(dut.clk, 10)" + "\n")
        testcasefile.write("    dut.rst_n.value = 1" + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # Enter 'READ' Mode" + "\n")
        testcasefile.write("    dut.ui_in.value = 0                       # Negative Edge (start transmission)" + "\n")
        testcasefile.write("    await ClockCycles(dut.clk, 2)" + "\n")
        testcasefile.write("    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted" + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # Transmit Input Image (Serial Transmission)" + "\n")
        testcasefile.write("    dut._log.info('Transmitting Image...')" + "\n")
        testcasefile.write("    for row in input_image:" + "\n")
        testcasefile.write("        dut.ui_in.value = 128 + row[13:7].integer" + "\n")
        testcasefile.write("        await ClockCycles(dut.clk, 1)" + "\n")
        testcasefile.write("        dut.ui_in.value = 128 + row[6:0].integer" + "\n")
        testcasefile.write("        await ClockCycles(dut.clk, 1)" + "\n")
        testcasefile.write("    dut._log.info('Transmitting Image...Done')" + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # Wait for Additional Clock Cycle(s) Before Evaluating" + "\n")
        testcasefile.write("    await ClockCycles(dut.clk, 10)" + "\n")
        testcasefile.write("\n")
        testcasefile.write("    # Evaluate Results" + "\n")
        testcasefile.write("    # ----------------" + "\n")
        testcasefile.write("    dut._log.info('Evaluating...')" + "\n")
        testcasefile.write("    #print(dut.uo_out.value)" + "\n")
        testcasefile.write("    #print(dut.uio_out.value)" + "\n")
        testcasefile.write("    #print(dut.uio_oe.value)" + "\n")
        testcasefile.write("    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1" + "\n")
        testcasefile.write("    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'" + "\n")
        testcasefile.write("    assert int(dut.uio_out.value) == classification_result" + "\n")
        testcasefile.write("    if('1.8.1' in cocotb.__version__):" + "\n")
        testcasefile.write("        # Flip Endian-ness in cocotb v1.8.1" + "\n")
        testcasefile.write("        assert int(dut.uo_out.value[1:7]) == segments[classification_result]" + "\n")
        testcasefile.write("    else:" + "\n")
        testcasefile.write("        assert int(dut.uo_out.value[6:0]) == segments[classification_result]" + "\n")
        testcasefile.write("    dut._log.info('Evaluating...Done')" + "\n")
        testcasefile.write("\n")
        testcasefile.write("\n")

# Generate cocotb compatible Test Cases
# desc - generate a set of cocotb test cases from the dataframe
#
def generate_cocotb_tests(df, filename):
    print("Generating cocotb unittests")
    df.apply(test_writer, axis=1)
    print("Generating cocotb unittests...Done")