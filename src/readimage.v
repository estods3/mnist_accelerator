/*
 * Copyright (c) 2024 Evan Stoddart
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module ImageReader(
    input wire clk,                // Clock input
    input wire reset_n,            // Reset input (active low)
    input wire [6:0] data_in,      // 7-bit bus input for image data
    output reg [195:0] image_data, // Output vector register to store the image data
    output reg image_ready         // Output flag to signal the image is ready (active high)
);

    // Internal variables
    reg [7:0] row_data [0:13];  // Array to store each row of 14 bits

    // Registers for address and control
    reg [3:0] row_index;   // Index for rows (0 to 13)
    reg [2:0] col_index;   // Index for columns (0 to 13)

    // State machine states
    parameter IDLE = 2'b00;
    parameter READ_DATA = 2'b01;
    reg [1:0] state;       // State machine state register

    always @(posedge clk or negedge reset_n) begin
        if (~reset_n) begin
            // Reset conditions
            state <= READ_DATA;
            row_index <= 0;
            col_index <= 0;
            image_data <= 0;
            image_ready <= 0;
            //row_data <= 0;
        end else begin
            // State machine transitions
            case(state)
                IDLE: begin
                end

                READ_DATA: begin
                    // Read data from input bus and store in row_data
                    row_data[row_index] <= data_in;

                    // Increment column index
                    col_index <= col_index + 1;

                    // Check if all columns of current row are read
                    if (col_index == 13) begin
                        // Move to next row
                        row_index <= row_index + 1;
                        col_index <= 0;
                    end

                    // Check if all rows are read
                    if (row_index == 13) begin
                        // Move to IDLE state or another state as needed
                        state <= IDLE;
                        // Combine row_data into a 196-bit vector
                        // Assuming you want to concatenate rows into a single vector
                        for (int i = 0; i < 14; i = i + 1) begin
                            image_data <= {image_data, row_data[i]};
                        end
                        image_ready <= 1'b1;  // flag that image has been read
                    end
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
