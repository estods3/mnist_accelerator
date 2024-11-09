# SPDX-FileCopyrightText: 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    dut._log.info("Test project behavior")

    # Set the input values you want to test
    dut.ui_in.value = 20
    dut.uio_in.value = 30

    # Wait for one clock cycle to see the output values
    await ClockCycles(dut.clk, 1)

    # The following assersion is just an example of how to check the output values.
    # Change it to match the actual expected output of your module:
    #assert dut.uo_out.value == 20

    # Keep testing the module by changing the input values, waiting for
    # one or more clock cycles, and asserting the expected output values.


segments = [ 63, 6, 91, 79, 102, 109, 125, 7, 127, 111 ]

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
