module fileread #(parameter file = "test_weights.txt", parameter size = 2, parameter count = 1)(out);
 
	output reg [(size*count*32)-1:0] out;
	reg [31:0] temp;
	reg [(size*count*32)-1:0] A;
	integer wb_file;
	integer i;
 
	initial begin
  		wb_file = $fopen(file, "r");
		while(! $feof(wb_file)) begin
			for (i = 0; i < size*count; i = i + 1) begin
				$fscanf(wb_file, "%h\n", temp);
				A[32*(i) +: 32] = temp[31:0];
			end
		end

		assign out = A[(size*count*32)-1:0];
	end
endmodule