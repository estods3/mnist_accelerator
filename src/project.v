/*
 * Copyright (c) 2024 Your Name
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_estods3_nnaccelerator (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    wire [6:0] input_image_row;
    wire [6:0] seven_seg_display_out;

    assign ui_in[6:0] = input_image_row;
    assign uo_out[6:0] = seven_seg_display_out;
    assign uo_out[7] = 1'b0;

    reg [195:0] image_array; //14x14=196 memory
    integer i=0; 

    
    reg [3:0] digit_classification_bcd = 4'b0000;

    always@(posedge clk)  
    
        if(i<189) begin
            image_array[i] <= input_image_row; //Read image 7 bits at a time into memory. 
            i <= i + 7;
        end

   
    // instantiate segment display
    seg7 seg7(.counter(digit_classification_bcd), .segments(seven_seg_display_out));
    
    // All output pins must be assigned. If not used, assign to 0.
    assign uio_in = 0;
    assign uio_out[3:0] = digit_classification_bcd;
    assign uio_out[7:4] = 0;
    assign uio_oe = 1;

    // List all unused inputs to prevent warnings
    wire _unused = &{ena, clk, rst_n, 1'b0};

endmodule
