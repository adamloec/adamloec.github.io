// Neuron module, handles the addition and multiplication for the neural network layers.

module neuron #(parameter file = "", parameter num_weights = 0, parameter num_outputs = 0, parameter count = 0)(in, bias, result, final);

    input [(num_weights*32)-1:0] in;
    input [31:0] bias;
    output [31:0] result;
    output [31:0] final;

    wire [(num_weights*32)-1:0] weights;
    wire [31:0] results [(num_weights)-1:0];
    wire [63:0] product;
    wire [31:0] summation;
    wire [31:0] temp_sum;
    wire carryout;
    reg carry_in;

    // Multiplying the weights, and adding the biases, against the respective input data
    fileread #(.file(file), .size(num_weights), .count(count)) weight (.out(weights));
    genvar i, j;
    generate
        for (i = 0; i < num_weights; i = i + 1) begin: multiply
            cla_multiplier mul (.multicand(in[(31 + (32*i))-1:32*i]), 
                .multiplier(weights[(31 + (32*i))-1:32*i]),
                .product(product), 
                .final(results[i]));
        end
    endgenerate

    // Adding each product of the weights and inputs
    generate
        for (j = 0; j < num_weights; j = j + 1) begin: add
            assign temp_sum = summation;
            cla_adder add1 (
     				.sum(summation[31:0]),
         	        .carry_out(carry_out),
         	        .in1(results[j]),
         	        .in2(temp_sum[31:0]),
         	        .carry_in(carry_in));
        end
    endgenerate

    // Adding the bias to the end of the summation of the product of weights and inputs
    cla_adder add2 (
     				.sum(result[31:0]),
         	        .carry_out(carry_out),
         	        .in1(temp_sum[31:0]),
         	        .in2(bias[31:0]),
         	        .carry_in(carry_in));
    
    // Activation function for the neuron
    relu resulttt (.in(result[31:0]), .out(final[31:0]));
endmodule