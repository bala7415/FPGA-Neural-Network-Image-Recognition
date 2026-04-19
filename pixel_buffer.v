module pixel_buffer (
    input  wire       clk,
    input  wire       rst,
 
    // UART feed
    input  wire [7:0] uart_data,
    input  wire       uart_valid,
 
    // Handshake to NN core
    output reg        image_ready,   // 1-clock pulse when 400 bytes received
 
    // Read port for NN core
    input  wire [8:0] rd_addr,       // 0..399
    output wire [7:0] rd_data
);
 
// -----------------------------------------------------------
// 400-byte pixel RAM  (inferred as distributed RAM / BRAM)
// -----------------------------------------------------------
reg [7:0] mem [0:399];
integer i;
initial begin
    for (i = 0; i < 400; i = i+1)
        mem[i] = 8'h00;
end
 
// Read port is asynchronous for simplicity (NN reads freely)
assign rd_data = mem[rd_addr];
 
// -----------------------------------------------------------
// Write state machine
// -----------------------------------------------------------
reg [8:0] wr_ptr = 0;   // 0..399, then wraps
 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        wr_ptr      <= 0;
        image_ready <= 0;
    end else begin
        image_ready <= 0;   // default - only high for one cycle
 
        if (uart_valid) begin
            mem[wr_ptr] <= uart_data;
 
            if (wr_ptr == 399) begin
                wr_ptr      <= 0;       // wrap: ready for next image
                image_ready <= 1;       // tell NN to start
            end else begin
                wr_ptr <= wr_ptr + 1;
            end
        end
    end
end
 
endmodule