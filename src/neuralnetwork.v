/*
 * Copyright (c) 2024 Evan Stoddart
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

//`timescale 1ns / 1ps

module digit_classifier (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [195:0] image_in,  // 14x14 binary image (196 bits)
    output reg [3:0] digit_out,   // 4 bits to represent digits 0-9
    output reg valid_out          // Indicates when output is valid
);

    // Parameters
    localparam INPUT_SIZE = 196;
    localparam HIDDEN1_SIZE = 128;
    localparam HIDDEN2_SIZE = 64;
    localparam OUTPUT_SIZE = 10;

    // States
    localparam IDLE = 2'b00;
    localparam LAYER1 = 2'b01;
    localparam LAYER2 = 2'b10;
    localparam LAYER3 = 2'b11;
    
    // State and counters
    reg [1:0] state;
    reg [7:0] neuron_counter;  // Count which neuron we're processing
    
    // Intermediate results for each layer
    reg signed [15:0] layer1_outputs [0:HIDDEN1_SIZE-1];  // 16-bit fixed point for first hidden layer
    reg signed [15:0] layer2_outputs [0:HIDDEN2_SIZE-1];  // 16-bit fixed point for second hidden layer
    reg signed [15:0] layer3_outputs [0:OUTPUT_SIZE-1];   // 16-bit fixed point for output layer
    
    // Hardcoded weights and biases (simplified with patterns for brevity)
    // In a real implementation, these would be exact trained values
    // Using fixed point: 1 sign bit, 3 integer bits, 12 fractional bits
    
    // Temporary variables for calculations
    reg signed [31:0] acc;  // Accumulator for MAC operations
    integer i, j;           // Loop iterators
    integer max_val_idx;    // Index of maximum value
    reg signed [15:0] max_val; // Maximum value
    
    // ReLU function - implemented as a simple comparator
    function signed [15:0] relu;
        input signed [15:0] x;
        begin
            relu = (x > 0) ? x : 0;
        end
    endfunction
    
    // Function to get weight for layer 1 (input to hidden1)
    // Using a deterministic pattern for weights - would be replaced with actual trained values
    function signed [15:0] get_weight_layer1;
        input integer input_idx;
        input integer neuron_idx;
        reg signed [15:0] weight;
        begin
            if ((input_idx + neuron_idx) % 3 == 0)
                weight = 16'h0400; // +1.0 in our fixed-point format
            else if ((input_idx + neuron_idx) % 3 == 1)
                weight = 16'hFC00; // -1.0 in our fixed-point format
            else
                weight = 16'h0200; // +0.5 in our fixed-point format
                
            get_weight_layer1 = weight;
        end
    endfunction
    
    // Function to get weight for layer 2 (hidden1 to hidden2)
    function signed [15:0] get_weight_layer2;
        input integer input_idx;
        input integer neuron_idx;
        reg signed [15:0] weight;
        begin
            if ((input_idx * neuron_idx) % 4 == 0)
                weight = 16'h0300; // +0.75 in our fixed-point format
            else if ((input_idx * neuron_idx) % 4 == 1)
                weight = 16'hFD00; // -0.75 in our fixed-point format
            else if ((input_idx * neuron_idx) % 4 == 2)
                weight = 16'h0200; // +0.5 in our fixed-point format
            else
                weight = 16'hFE00; // -0.5 in our fixed-point format
                
            get_weight_layer2 = weight;
        end
    endfunction
    
    // Function to get weight for layer 3 (hidden2 to output)
    function signed [15:0] get_weight_layer3;
        input integer input_idx;
        input integer neuron_idx;
        reg signed [15:0] weight;
        begin
            if ((input_idx + neuron_idx) % 5 == 0)
                weight = 16'h0500; // +1.25 in our fixed-point format
            else if ((input_idx + neuron_idx) % 5 == 1)
                weight = 16'hFB00; // -1.25 in our fixed-point format
            else if ((input_idx + neuron_idx) % 5 == 2)
                weight = 16'h0300; // +0.75 in our fixed-point format
            else if ((input_idx + neuron_idx) % 5 == 3)
                weight = 16'hFD00; // -0.75 in our fixed-point format
            else
                weight = 16'h0100; // +0.25 in our fixed-point format
                
            get_weight_layer3 = weight;
        end
    endfunction
    
    // Function to get bias for layer 1
    function signed [15:0] get_bias_layer1;
        input integer neuron_idx;
        reg signed [15:0] bias;
        begin
            if (neuron_idx % 4 == 0)
                bias = 16'h0080; // +0.125 in our fixed-point format
            else if (neuron_idx % 4 == 1)
                bias = 16'hFF80; // -0.125 in our fixed-point format
            else if (neuron_idx % 4 == 2)
                bias = 16'h0040; // +0.0625 in our fixed-point format
            else
                bias = 16'hFFC0; // -0.0625 in our fixed-point format
                
            get_bias_layer1 = bias;
        end
    endfunction
    
    // Function to get bias for layer 2
    function signed [15:0] get_bias_layer2;
        input integer neuron_idx;
        reg signed [15:0] bias;
        begin
            if (neuron_idx % 3 == 0)
                bias = 16'h00C0; // +0.1875 in our fixed-point format
            else if (neuron_idx % 3 == 1)
                bias = 16'hFF40; // -0.1875 in our fixed-point format
            else
                bias = 16'h0100; // +0.25 in our fixed-point format
                
            get_bias_layer2 = bias;
        end
    endfunction
    
    // Function to get bias for layer 3
    function signed [15:0] get_bias_layer3;
        input integer neuron_idx;
        reg signed [15:0] bias;
        begin
            case (neuron_idx)
                0: bias = 16'h0100; // +0.25
                1: bias = 16'hFF00; // -0.25
                2: bias = 16'h0080; // +0.125
                3: bias = 16'hFF80; // -0.125
                4: bias = 16'h0040; // +0.0625
                5: bias = 16'hFFC0; // -0.0625
                6: bias = 16'h0020; // +0.03125
                7: bias = 16'hFFE0; // -0.03125
                8: bias = 16'h0010; // +0.015625
                9: bias = 16'hFFF0; // -0.015625
                default: bias = 16'h0000;
            endcase
            
            get_bias_layer3 = bias;
        end
    endfunction
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            neuron_counter <= 0;
            valid_out <= 0;
            digit_out <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LAYER1;
                        neuron_counter <= 0;
                        valid_out <= 0;
                    end
                end
                
                LAYER1: begin
                    // Process first hidden layer
                    if (neuron_counter < HIDDEN1_SIZE) begin
                        // Calculate dot product for current neuron
                        acc = 0;
                        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
                            if (image_in[i]) begin  // Binary image, so just add weight if pixel is 1
                                acc = acc + get_weight_layer1(i, neuron_counter);
                            end
                        end
                        // Add bias and apply ReLU
                        layer1_outputs[neuron_counter] = relu(acc[27:12] + get_bias_layer1(neuron_counter));
                        neuron_counter <= neuron_counter + 1;
                    end else begin
                        state <= LAYER2;
                        neuron_counter <= 0;
                    end
                end
                
                LAYER2: begin
                    // Process second hidden layer
                    if (neuron_counter < HIDDEN2_SIZE) begin
                        // Calculate dot product for current neuron
                        acc = 0;
                        for (i = 0; i < HIDDEN1_SIZE; i = i + 1) begin
                            acc = acc + (layer1_outputs[i] * get_weight_layer2(i, neuron_counter));
                        end
                        // Add bias and apply ReLU
                        layer2_outputs[neuron_counter] = relu(acc[27:12] + get_bias_layer2(neuron_counter));
                        neuron_counter <= neuron_counter + 1;
                    end else begin
                        state <= LAYER3;
                        neuron_counter <= 0;
                    end
                end
                
                LAYER3: begin
                    // Process output layer
                    if (neuron_counter < OUTPUT_SIZE) begin
                        // Calculate dot product for current neuron
                        acc = 0;
                        for (i = 0; i < HIDDEN2_SIZE; i = i + 1) begin
                            acc = acc + (layer2_outputs[i] * get_weight_layer3(i, neuron_counter));
                        end
                        // Add bias (no activation function in output layer)
                        layer3_outputs[neuron_counter] = acc[27:12] + get_bias_layer3(neuron_counter);
                        neuron_counter <= neuron_counter + 1;
                    end else begin
                        // Find the maximum value (argmax)
                        max_val = layer3_outputs[0];
                        max_val_idx = 0;
                        for (i = 1; i < OUTPUT_SIZE; i = i + 1) begin
                            if (layer3_outputs[i] > max_val) begin
                                max_val = layer3_outputs[i];
                                max_val_idx = i;
                            end
                        end
                        
                        digit_out <= max_val_idx;  // Output the predicted digit
                        valid_out <= 1;           // Set output valid flag
                        state <= IDLE;            // Return to idle state
                    end
                end
            endcase
        end
    end
    
    // Initialize control signals
    initial begin
        state = IDLE;
        neuron_counter = 0;
        valid_out = 0;
        digit_out = 0;
    end
    
endmodule