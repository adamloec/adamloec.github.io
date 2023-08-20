module mlpnet #(parameter input_file = "")(guess);
    
    wire [31:0] outputs1 [(200)-1:0];
    output [31:0] guess [(10)-1:0];

    // Hidden layer, 784, 200
    layer #(.file("hidden_weights.txt"), .num_in(784), .num_out(200)) hidden_layer (outputs1);
    
    // Output layer, 200, 10
    layer #(.file("output_weights.txt"), .num_in(200), .num_out(10)) output_layer (guess);
endmodule