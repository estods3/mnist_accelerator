# SPDX-FileCopyrightText: 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.types import LogicArray
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles

# Helper Functions
# ----------------
#def transmit_image():
# TODO

# Seven Segments Lookup Table
# ---------------------------
# Integer representation of the 7 segments for digits 0-9
segments = [ 63, 6, 91, 79, 102, 109, 125, 7, 127, 111 ]

@cocotb.test()
async def test_blank_image(dut):
    # Test 1: Blank Image
    # Author: estods3
    # Input: Blank (all 0s) 14x14 image.
    # Expected Result: BSD = 0, Seven Segment = 63
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

    # Transmit Input Image (Serial Transmission)
    dut._log.info("Transmitting Image...")
    dut.ui_in.value = 0 # Negative Edge (start transmission)
    await ClockCycles(dut.clk, 1)
    for i in range(0, 28):
        dut.ui_in.value = 128 #[1,0,0,0,0,0,0,0]
        await ClockCycles(dut.clk, 1)
    dut._log.info("Transmitting Image...Done")

    # Wait for Additional Clock Cycle(s) Before Evaluating
    await ClockCycles(dut.clk, 10)

    # Evaluate Results
    dut._log.info("Evaluating...")
    print(dut.uo_out.value)  # Seven Segment Values ==> '0'=63
    print(dut.uio_out.value) # BSD Value ==> 00000010
    print(dut.uio_oe.value)
    if("1.8.1" in cocotb.__version__):
        assert int(dut.uo_out[7].value) == 1
        # Flip Endian-ness
        assert int(dut.uo_out.value[1:7]) == segments[classification_result]
        assert int(dut.uio_out.value) == classification_result
        assert int(dut.uio_oe.value) == 0xFF
    else:
        assert int(dut.uo_out[7].value) == 1
        assert int(dut.uo_out.value[6:0]) == segments[classification_result]
        assert int(dut.uio_out.value) == classification_result
        assert int(dut.uio_oe.value) == 0xFF
    dut._log.info("Evaluating...Done")

@cocotb.test()
async def test_7seg_0(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Expected Result
    Classification_Result = 0

    # Test Digit: 0
    dut._log.info("Testing Digit: 0")
    print(dut.uo_out.value)  # Seven Segment Values ==> '0'=63
    print(dut.uio_out.value) # BSD Value ==> 00000010
    print(dut.uio_oe.value)
    #if("1.8.1" in cocotb.__version__):
    #    assert int(dut.uo_out[7].value) == 1
    #    # Flip Endian-ness
    #    assert int(dut.uo_out.value[1:7]) == segments[Classification_Result]
    #    assert int(dut.uio_out.value) == Classification_Result
    #    assert int(dut.uio_oe.value) == 0xFF
    #else:
    #    assert int(dut.uo_out[7].value) == 1
    #    assert int(dut.uo_out.value[6:0]) == segments[Classification_Result]
    #    assert int(dut.uio_out.value) == Classification_Result
    #    assert int(dut.uio_oe.value) == 0xFF

@cocotb.test()
async def test_7seg_1(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Expected Result
    Classification_Result = 1

    # Test Digit: 1
    dut._log.info("Testing Digit: 1")
    print(dut.uo_out.value)  # Seven Segment Values ==> '0'=6
    print(dut.uio_out.value) # BSD Value ==> 00000010
    print(dut.uio_oe.value)
    if("1.8.1" in cocotb.__version__):
        assert int(dut.uo_out[7].value) == 1
        # Flip Endian-ness
        assert int(dut.uo_out.value[1:7]) == segments[Classification_Result]
        assert int(dut.uio_out.value) == Classification_Result
        assert int(dut.uio_oe.value) == 0xFF
    else:
        assert int(dut.uo_out[7].value) == 1
        assert int(dut.uo_out.value[6:0]) == segments[Classification_Result]
        assert int(dut.uio_out.value) == Classification_Result
        assert int(dut.uio_oe.value) == 0xFF
