module neuron_tb();
    
    wire [(2*32)-1:0] image;
    wire [31:0] bias;
    wire [31:0] result;

    fileread #(.file("test_input.txt"), .size(2), .count(1)) in (.out(image));
    fileread #(.file("test_biases.txt"), .size(1), .count(1)) bi (.out(bias));

    neuron #(.file("test_weights.txt"), .num_weights(2), .num_outputs(1), .count(1)) n
        (.in(image),
        .bias(bias),
        .result(result));
endmodule