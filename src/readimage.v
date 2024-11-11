/*
 * Copyright (c) 2024 Evan Stoddart
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module ImageReader(
    input wire clk,                           // Clock input
    input wire reset_n,                       // Reset input (active low)
    input wire [6:0] data_in,                 // 7-bit bus input for image data
    output reg [195:0] image_data,            // Output vector register to store the image data
    output reg image_ready                    // Output flag to signal the image is ready (active high)
);

    // Registers for address and control
    reg [4:0] rows_read;                      // Index to keep track of the number of rows that have been read since last reset_n

    // State machine states
    parameter IDLE = 2'b00;                   // When Finished Reading Data, Transmit image_ready flag and then IDLE. Remain in IDLE until next negative edge of reset_n
    parameter READ_DATA = 2'b01;              // Read Data State - once reset_n triggers start of data, continue reading until buffer full, or new negative edge is recieved.
    reg [1:0] state;                          // State machine state register

    always @(posedge clk or negedge reset_n) begin
        if (~reset_n) begin
            //$display("RESET!!");
            // Reset conditions
            state <= READ_DATA;
            rows_read <= 0;
            image_data <= 0;
            image_ready <= 0;
        end else begin
            // State machine transitions
            case(state)
                IDLE: begin
                end

                READ_DATA: begin
                    // Read data from input bus and store in image_data
                    if (rows_read < 28) begin
                        image_data <= (image_data << 7) | data_in;
                        rows_read <= rows_read + 1;
                        //$display("--------");
                        //$display(data_in);
                        //$display(image_data);
                        //$display(rows_read);
                    end

                    // Check if all rows are read
                    if (rows_read == 28) begin
                        state <= IDLE;
                        image_ready <= 1'b1;  // flag that image has been read
                    end
                end
            endcase
        end
    end

endmodule
