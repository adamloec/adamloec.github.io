
module cla_adder_tb();
 	reg carry_in;
 	reg [31:0] in1;
 	reg [31:0] in2;

	wire carry_out;
 	wire [31:0] sum;

 	cla_adder cla1(
         	.sum(sum[31:0]),
         	.carry_out(carry_out),
         	.in1(in1[31:0]),
         	.in2(in2[31:0]),
         	.carry_in(carry_in));

endmodule