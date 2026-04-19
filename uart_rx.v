// ============================================================
// uart_rx.v  -  8N1 UART Receiver
// Unchanged from user's working version.
// ============================================================
module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data_out,
    output reg        data_valid
);
 
localparam BAUD_TICK = CLK_FREQ / BAUD_RATE;
localparam HALF_TICK = BAUD_TICK / 2;
 
// 2-FF synchroniser
reg rx_sync1, rx_sync2;
always @(posedge clk) begin
    rx_sync1 <= rx;
    rx_sync2 <= rx_sync1;
end
wire rx_clean = rx_sync2;
 
localparam IDLE  = 2'd0;
localparam START = 2'd1;
localparam DATA  = 2'd2;
localparam STOP  = 2'd3;
 
reg [1:0]  state    = IDLE;
reg [15:0] baud_cnt = 0;
reg [2:0]  bit_cnt  = 0;
reg [7:0]  shift_reg = 0;
 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state      <= IDLE;
        baud_cnt   <= 0;
        bit_cnt    <= 0;
        data_out   <= 0;
        data_valid <= 0;
    end else begin
        data_valid <= 0;
 
        case (state)
            IDLE: begin
                baud_cnt <= 0;
                bit_cnt  <= 0;
                if (rx_clean == 0)
                    state <= START;
            end
 
            START: begin
                if (baud_cnt == HALF_TICK) begin
                    baud_cnt <= 0;
                    state    <= (rx_clean == 0) ? DATA : IDLE;
                end else
                    baud_cnt <= baud_cnt + 1;
            end
 
            DATA: begin
                if (baud_cnt == BAUD_TICK-1) begin
                    baud_cnt  <= 0;
                    shift_reg <= {rx_clean, shift_reg[7:1]};
                    if (bit_cnt == 7) begin
                        bit_cnt <= 0;
                        state   <= STOP;
                    end else
                        bit_cnt <= bit_cnt + 1;
                end else
                    baud_cnt <= baud_cnt + 1;
            end
 
            STOP: begin
                if (baud_cnt == BAUD_TICK-1) begin
                    baud_cnt <= 0;
                    if (rx_clean == 1) begin
                        data_out   <= shift_reg;
                        data_valid <= 1;
                    end
                    state <= IDLE;
                end else
                    baud_cnt <= baud_cnt + 1;
            end
        endcase
    end
end
 
endmodule