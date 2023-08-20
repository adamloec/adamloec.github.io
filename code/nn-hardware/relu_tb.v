module relu_tb();
	reg [31:0] in;
	wire [31:0] out;

	relu rel(
		.in(in[31:0]),
		.out(out[31:0]));

	initial begin
		in = 32'h3f800000;
		#10;
	end
endmodule
