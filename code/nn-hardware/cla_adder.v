module cla_adder (in1, in2, carry_in, sum, carry_out);

 	input [31:0] in1;
 	input [31:0] in2;
 	input carry_in;
 	output [31:0] sum;
 	output carry_out;

 	wire [31:0] gen;
 	wire [31:0] pro;
 	wire [32:0] carry_tmp;

 	genvar j, i;
 	generate

  		assign carry_tmp[0] = carry_in;
 
  		for(j = 0; j < 32; j = j + 1) begin: carry_generator
   			assign gen[j] = in1[j] & in2[j];
   			assign pro[j] = in1[j] | in2[j];
   			assign carry_tmp[j+1] = gen[j] | pro[j] & carry_tmp[j];
  		end

  		assign carry_out = carry_tmp[32];
 
  		for(i = 0; i < 32; i = i+1) begin: sum_without_carry
   			assign sum[i] = in1[i] ^ in2[i] ^ carry_tmp[i];
  		end 
 
 	endgenerate 
endmodule