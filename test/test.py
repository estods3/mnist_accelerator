# SPDX-FileCopyrightText: 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.types import LogicArray
from cocotb.binary import BinaryValue
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles

# Helper Functions
# ----------------
#def transmit_image(dut, input_image):
#    dut._log.info("Transmitting Image...")
#    dut.ui_in.value = 0 # Negative Edge (start transmission)
#    await ClockCycles(dut.clk, 1)
#    for i in range(0, 28):
#        dut.ui_in.value = 128 #[1,0,0,0,0,0,0,0]
#        await ClockCycles(dut.clk, 1)
#    dut._log.info("Transmitting Image...Done")

# Seven Segments Lookup Table
# ---------------------------
# Integer representation of the 7 segments for digits 0-9
segments = [ 63, 6, 91, 79, 102, 109, 125, 7, 127, 111 ]

@cocotb.test()
async def test_blank_image(dut):
    # Test 1: Blank Image
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image.
    # Expected Result: BCD = 0, Seven Segment = 63
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000")]

    classification_result = 0

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    print(dut.uo_out.value)
    print(dut.uio_out.value)
    print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_1_checksum(dut):
    # Test 2: Blank Image with 1 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a '1' in LSB
    # Expected Result: BCD = 1, Seven Segment = 6
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000001")]

    classification_result = 1

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    print(dut.uo_out.value)
    print(dut.uio_out.value)
    print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_2_checksum(dut):
    # Test 3: Blank Image with 2 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 2 checksum
    # Expected Result: BCD = 2, Seven Segment = 91
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000010")]

    classification_result = 2

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        #print(dut.ui_in.value)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
        #print(dut.ui_in.value)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    print(dut.uo_out.value)
    print(dut.uio_out.value)
    print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_3_checksum(dut):
    # Test 4: Blank Image with 3 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 3 checksum
    # Expected Result: BCD = 3, Seven Segment = 79
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000011")]

    classification_result = 3

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_4_checksum(dut):
    # Test 5: Blank Image with 4 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 4 checksum
    # Expected Result: BCD = 4, Seven Segment = 102
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000100")]

    classification_result = 4

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_5_checksum(dut):
    # Test 6: Blank Image with 5 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 5 checksum
    # Expected Result: BCD = 5, Seven Segment = 109
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000101")]

    classification_result = 5

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_6_checksum(dut):
    # Test 7: Blank Image with 6 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 6 checksum
    # Expected Result: BCD = 6, Seven Segment = 125
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000110")]

    classification_result = 6

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_7_checksum(dut):
    # Test 8: Blank Image with 7 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 7 checksum
    # Expected Result: BCD = 7, Seven Segment = 7
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000111")]

    classification_result = 7

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_8_checksum(dut):
    # Test 9: Blank Image with 8 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 8 checksum
    # Expected Result: BCD = 8, Seven Segment = 127
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000001000")]

    classification_result = 8

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_blank_image_with_9_checksum(dut):
    # Test 10: Blank Image with 9 checksum
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image with a 9 checksum
    # Expected Result: BCD = 9, Seven Segment = 111
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000001001")]

    classification_result = 9

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter "READ" Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if("1.8.1" in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info("Evaluating...Done")


####################################################################
#                                                                  #
#                     AUTOGENERATED TEST CASES                     #
#                                                                  #
####################################################################

@cocotb.test()
async def test_batch1_sample21(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=21
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 4
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000011000"), \
                   LogicArray("00011000111000"), \
                   LogicArray("00011100111000"), \
                   LogicArray("00011101111000"), \
                   LogicArray("00011111111000"), \
                   LogicArray("00011111110000"), \
                   LogicArray("00011111111000"), \
                   LogicArray("00000001111000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000001100000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 4

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch0_sample11(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=0, Sample=11
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 6
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000110000000"), \
                   LogicArray("00001110000000"), \
                   LogicArray("00001100000000"), \
                   LogicArray("00001100011000"), \
                   LogicArray("00011101111100"), \
                   LogicArray("00011011111100"), \
                   LogicArray("00011111001100"), \
                   LogicArray("00011111001100"), \
                   LogicArray("00011111111000"), \
                   LogicArray("00001111110000"), \
                   LogicArray("00000111100000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 6

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch1_sample23(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=23
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 3
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00011111000000"), \
                   LogicArray("00111111000000"), \
                   LogicArray("00000111000000"), \
                   LogicArray("00001111000000"), \
                   LogicArray("00011111100000"), \
                   LogicArray("00011001111100"), \
                   LogicArray("00000000111000"), \
                   LogicArray("00000001111000"), \
                   LogicArray("00000011110000"), \
                   LogicArray("00000111110000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 3

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch1_sample13(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=13
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 2
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000111000000"), \
                   LogicArray("00011111110000"), \
                   LogicArray("00111000110000"), \
                   LogicArray("00110000110000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000111100000"), \
                   LogicArray("00000111111110"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 2

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch1_sample63(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=63
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 5
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000011100"), \
                   LogicArray("00000011111100"), \
                   LogicArray("00001111110000"), \
                   LogicArray("00011111000000"), \
                   LogicArray("00011100000000"), \
                   LogicArray("00001111000000"), \
                   LogicArray("00000111100000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00001001110000"), \
                   LogicArray("00001111100000"), \
                   LogicArray("00000111000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 5

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch1_sample10(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=10
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 1
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 1

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch0_sample5(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=0, Sample=5
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 1
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000110000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000001100000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000111000000"), \
                   LogicArray("00000111000000"), \
                   LogicArray("00000110000000"), \
                   LogicArray("00000010000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 1

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch1_sample37(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=1, Sample=37
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 0
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000001110000"), \
                   LogicArray("00000111111000"), \
                   LogicArray("00001111111100"), \
                   LogicArray("00001110001100"), \
                   LogicArray("00011100001100"), \
                   LogicArray("00011000001100"), \
                   LogicArray("00011000001100"), \
                   LogicArray("00011000011100"), \
                   LogicArray("00001100111000"), \
                   LogicArray("00001111110000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 0

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch0_sample12(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=0, Sample=12
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 9
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000011110000"), \
                   LogicArray("00000111111000"), \
                   LogicArray("00001110111000"), \
                   LogicArray("00001100110000"), \
                   LogicArray("00001111110000"), \
                   LogicArray("00001111110000"), \
                   LogicArray("00000001100000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011100000"), \
                   LogicArray("00000011000000"), \
                   LogicArray("00000011000000"), \
    ]

    classification_result = 9

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


@cocotb.test()
async def test_batch0_sample18(dut):
    # THIS TEST WAS AUTOGENERATED USING data_preprocessor.py
    # Test: Batch=0, Sample=18
    # Author: estods3
    # Input: described in 'input_image'
    # Expected Result: BCD = 3
    # --------------------------------------------
    input_image = [LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00011100000000"), \
                   LogicArray("00111111000000"), \
                   LogicArray("00110111100000"), \
                   LogicArray("00111111100000"), \
                   LogicArray("00111111111000"), \
                   LogicArray("00001111111100"), \
                   LogicArray("00001100011110"), \
                   LogicArray("00001110001110"), \
                   LogicArray("00000111111110"), \
                   LogicArray("00000011111100"), \
                   LogicArray("00000000000000"), \
                   LogicArray("00000000000000"), \
    ]

    classification_result = 3

    # PERFORM TEST
    # ------------
    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    # Initial Conditions
    dut.ena.value = 1
    dut.ui_in.value = 128
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Enter 'READ' Mode
    dut.ui_in.value = 0                       # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uo_out[7].value) == 0      # Confirm Outputs Invalid (flag = 0) before Image is Transmitted

    # Transmit Input Image (Serial Transmission)
    dut._log.info('Transmitting Image...')
    for row in input_image:
        dut.ui_in.value = 128 + row[13:7].integer
        await ClockCycles(dut.clk, 1)
        dut.ui_in.value = 128 + row[6:0].integer
        await ClockCycles(dut.clk, 1)
    dut._log.info('Transmitting Image...Done')

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    # ----------------
    dut._log.info('Evaluating...')
    #print(dut.uo_out.value)
    #print(dut.uio_out.value)
    #print(dut.uio_oe.value)
    assert int(dut.uo_out[7].value) == 1  #Test Classification Flag set to 1
    assert int(dut.uio_oe.value) == 0xFF  #Test All Bidirectional I/O Output Enable set to '1'
    assert int(dut.uio_out.value) == classification_result
    if('1.8.1' in cocotb.__version__):
        # Flip Endian-ness in cocotb v1.8.1
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
    else:
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
    dut._log.info('Evaluating...Done')


