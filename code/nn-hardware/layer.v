module layer #(parameter file = "", parameter num_in = 0, parameter num_out = 0)(outputs);
    output [31:0] outputs [(num_out)-1:0];
    wire [(num_in*32)-1:0] image;
    wire [(num_out*32)-1:0] bias;
    genvar i;

    fileread #(.file("test_input.txt"), .size(num_in), .count(1)) in (.out(image));
    fileread #(.file("test_biases.txt"), .size(num_out), .count(1)) bi (.out(bias));

    generate
        for (i = 0; i < num_out; i = i + 1) begin: layer
            neuron #(.file(file), .num_weights(num_in), .num_outputs(1), .count(i)) n
                (.in(image),
                .bias(bias),
                .result(outputs[i]),
                .final(outputs[i]));
        end
    endgenerate
endmodule