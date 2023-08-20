module cla_multiplier_tb();
 	reg [31:0] multicand;
 	reg [31:0] multiplier;

 	wire [63:0] product;
	wire [31:0] final;
 	integer i;
 	cla_multiplier mul1(
      		.product (product[63:0]),
      		.multicand (multicand[31:0]),
      		.multiplier (multiplier[31:0]),
		.final (final[31:0]));

 	initial begin
  		for (i = 0; i < 30; i = i + 1) begin: W
   			multicand = multicand + 1; 
   			multiplier = multiplier + 1;
  		end
  		multicand = 32'h1000_0010;
  		multiplier = 32'h0000_0010;
 	end
endmodule