/*
 * Copyright (c) 2024 Evan Stoddart
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_estods3_nnaccelerator (
    input  wire [7:0] ui_in,                               // Dedicated inputs
    output wire [7:0] uo_out,                              // Dedicated outputs
    input  wire [7:0] uio_in,                              // IOs: Input path
    output wire [7:0] uio_out,                             // IOs: Output path
    output wire [7:0] uio_oe,                              // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,                                 // always 1 when the design is powered, so you can ignore it
    input  wire       clk,                                 // clock
    input  wire       rst_n                                // reset_n - low to reset
);

    // IO Variables
    // ------------
    reg [6:0] input_image_row;
    reg [6:0] seven_seg_display_out;
    reg reset_flag;
    reg complete_flag;
    reg classification_complete_flag;
    reg [3:0] digit_classification_bcd;
    reg [3:0] digit_classification_nn_bcd;

    // INTERNAL Variables
    // ------------------
    reg [195:0] image_array;                               // 14x14=196 memory
    reg image_ready;                                       // flag will be set when image has been read into image_array     

    // Assign IO
    // ---------
    assign input_image_row = ui_in[6:0];                   // First 7 ui_in bits used to read image (row by row)
    assign reset_flag = ui_in[7];                          // Last 1 ui_in bit used as a reset_flag
    assign uo_out[6:0] = seven_seg_display_out;            // First 7 uo_out bits used for 7-seg Display
    assign uo_out[7] = complete_flag;                      // Last 1 uo_out bit used as a complete flag
    // use bidirectionals as outputs
    assign uio_oe = 8'b11111111;                           // All uio set as outputs
    assign uio_out[3:0] = digit_classification_bcd;        // First 4 uio bits set as BCD Output
    assign uio_out[7:4] = 0;                               // Last 4 uio bits not used, set to 0.
    wire _unused = &{ena, uio_in, rst_n, 1'b0};            /// List all unused inputs to prevent warnings

    // Input Image
    // -----------
    // Read Image 7 bits at a time for 28 clock cycles to read full image
    // Another Image will be sent when reset_flag is low
    ImageReader ImageReader(.clk(clk), .reset_n(reset_flag), .data_in(input_image_row), .image_data(image_array), .image_ready(image_ready));
    

    // classifier
    // ----------
    // Perform Inferencing (Forward Pass)
    digit_classifier classifier(.clk(clk), .rst_n(rst_n & image_ready), .start(image_ready), .image_in(image_array), .digit_out(digit_classification_nn_bcd), .valid_out(classification_complete_flag));


    // Output
    // ------
    // instantiate segment display
    // output classification as BCD to output pins
    // raise flag to signal to raspberry pi to send another image
    seg7 seg7(.counter(digit_classification_bcd), .segments(seven_seg_display_out));


    // Process Image
    // -------------
    always @(posedge clk) begin
        if(image_ready) begin    
            // Output
            // ------
            if(image_array <= 196'd9) begin
                // TESTING MODE - If the sum of the image is a digit (0-9)
                // This is to test if the BCD to seven segment decoder works
                digit_classification_bcd <= image_array;
                complete_flag <= 1'b1;
            end else begin
                wait(classification_complete_flag);
                if (classification_complete_flag) begin                
                    // NORMAL MODE - if NN produces a valid output, set it to the digit_classification_bcd
                    digit_classification_bcd <= digit_classification_nn_bcd;
                    complete_flag <= classification_complete_flag;
                    //image_ready <= 0; //TODO - cant drive image_ready, do we need to reset it here?
                end
            end
        end else begin
            complete_flag <= 1'b0;
        end
    end

endmodule
