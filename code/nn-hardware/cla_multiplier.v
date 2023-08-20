module cla_multiplier (multicand, multiplier, product, final);

 	input [31:0] multicand;
 	input [31:0] multiplier;
 	output [63:0] product;
	output [31:0] final;

 	wire [31:0] multicand_tmp [31:0];
 	wire [31:0] product_tmp [31:0];
 	wire [31:0] carry_tmp;
	
 	genvar j, i;

 	generate 
  		for(j = 0; j < 32; j = j + 1) begin: for_loop_j
   			assign multicand_tmp[j] =  multicand & {32{multiplier[j]}};
  		end
 
  		assign product_tmp[0] = multicand_tmp[0];
  		assign carry_tmp[0] = 1'b0;
  		assign product[0] = product_tmp[0][0];

  		for(i = 1; i < 32; i = i + 1) begin: for_loop_i
 
   			cla_adder add1 (
     				.sum(product_tmp[i]),
     				.carry_out(carry_tmp[i]),
     				.carry_in(1'b0),
     				.in1(multicand_tmp[i]),
     				.in2({carry_tmp[i-1],product_tmp[i-1][31-:31]}));
   			assign product[i] = product_tmp[i][0];
  		end
 		assign product[63:32] = {carry_tmp[31],product_tmp[31][31-:31]};
 	endgenerate
	

	wire [63:0] temp = product;
	assign final[31:0] = temp[31:0];
endmodule