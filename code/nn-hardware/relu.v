module relu (in, out);
	input [31:0] in;

	output [31:0] out;
	
	// If the last bit is zero, then the result is positive and 
	// the output is the input.
	// Else the result is negative and the output is zero.
	assign out = (in[31] == 0)? in : 0;
endmodule