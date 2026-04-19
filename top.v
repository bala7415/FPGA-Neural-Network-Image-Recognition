// ============================================================
// top.v  -  Digit Recognizer Top Module
// Nexys4 rev-B  |  Artix-7 XC7A100T
//
// Data flow:
//   PC --UART--> pixel_buf (400 bytes, 20x20)
//              --> nn_core (3-layer NN)
//              --> seven_seg_ctrl
//
// Buttons: BTNU = next image, BTND = previous image
// Seven-seg: left-4 = "CP?-A"  right-4 = "---<digit>"
// ============================================================
module top (
    input  wire        clk,          // E3  100 MHz
    input  wire        rst_btn,      // C12 CPU_RESET (active-low on Nexys4)
    input  wire        rx,           // C4  UART RX from FTDI
 
    // Buttons
    input  wire        btnU,         // F15 next image
    input  wire        btnD,         // V10 previous image
 
    // 7-segment display
    output wire [6:0]  seg,          // CA..CG
    output wire        dp,
    output wire [7:0]  an,           // AN0..AN7
 
    // Debug LEDs (optional - shows inference running)
    output wire [15:0] led
);
 
// -------------------------------------------------------
// Reset: CPU_RESET is active-LOW on Nexys4
// -------------------------------------------------------
wire rst = ~rst_btn;
 
// -------------------------------------------------------
// UART RX
// -------------------------------------------------------
wire [7:0] uart_data;
wire       uart_valid;
 
uart_rx #(
    .CLK_FREQ (100_000_000),
        .BAUD_RATE(115200)
) u_uart (
    .clk       (clk),
    .rst       (rst),
    .rx        (rx),
    .data_out  (uart_data),
    .data_valid(uart_valid)
);
 
// -------------------------------------------------------
// Pixel buffer controller
//   Waits for 400 bytes then signals "image_ready"
//   New data automatically overwrites old buffer -
//   no reprogramming needed.
// -------------------------------------------------------
wire        image_ready;
wire [7:0]  pixel_out;
wire [8:0]  pixel_addr;   // 0..399
 
pixel_buffer u_pixbuf (
    .clk         (clk),
    .rst         (rst),
    .uart_data   (uart_data),
    .uart_valid  (uart_valid),
    .image_ready (image_ready),
    .rd_addr     (pixel_addr),
    .rd_data     (pixel_out)
);
 
// -------------------------------------------------------
// Image index (next/prev buttons with debounce)
// -------------------------------------------------------
wire btn_next, btn_prev;
 
debounce u_db_next (.clk(clk),.rst(rst),.btn_in(btnU),.btn_out(btn_next));
debounce u_db_prev (.clk(clk),.rst(rst),.btn_in(btnD),.btn_out(btn_prev));
 
reg [7:0] img_index = 0;   // 0..255 (supports up to 256 images)
 
always @(posedge clk or posedge rst) begin
    if (rst) img_index <= 0;
    else if (btn_next) img_index <= img_index + 1;
    else if (btn_prev) img_index <= img_index - 1;
end
 
// -------------------------------------------------------
// Neural Network Core (3-layer)
// -------------------------------------------------------
wire [3:0] nn_result;   // predicted digit 0-9
wire       nn_done;
wire       nn_start;
 
assign nn_start = image_ready;
 
nn_core u_nn (
    .clk        (clk),
    .rst        (rst),
    .start      (nn_start),
    .pixel_addr (pixel_addr),
    .pixel_data (pixel_out),
    .result     (nn_result),
    .done       (nn_done)
);
 
// -------------------------------------------------------
// Latch result & index when inference finishes
// -------------------------------------------------------
reg [3:0] display_digit = 0;
reg [7:0] display_index = 0;
 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        display_digit <= 0;
        display_index <= 0;
    end else if (nn_done) begin
        display_digit <= nn_result;
        display_index <= img_index;
    end
end
 
// -------------------------------------------------------
// Seven-segment display controller
// Left  4 digits: "CP  -A" where  = image number (0-99)
//   digit7='C' digit6='P' digit5=tens digit4=units '-' removed
//   Actually: digit7=C digit6=P digit5=index_tens digit4='-'
//             then right side digit3=index_units digit2='-' digit1='-' digit0=digit
//
// Layout (AN7..AN0 = leftmost..rightmost):
//   AN7: 'C'
//   AN6: 'P'
//   AN5: tens of image index
//   AN4: units of image index  (e.g. index=3 ? '0','3')
//   AN3: '-'
//   AN2: 'A'
//   AN1: '-'
//   AN0: predicted digit
// -------------------------------------------------------
wire [3:0] idx_tens  = display_index / 10;
wire [3:0] idx_units = display_index % 10;
 
seven_seg_ctrl u_ssd (
    .clk          (clk),
    .rst          (rst),
    .digit7       (4'hC),       // 'C' (custom encoding)
    .digit6       (4'hB),       // 'P' (custom encoding)
    .digit5       (idx_tens),
    .digit4       (idx_units),
    .digit3       (4'hA),       // '-'
    .digit2       (4'hE),       // 'A' (custom)
    .digit1       (4'hA),       // '-'
    .digit0       (display_digit),
    .dp_en        (8'b0),
    .seg          (seg),
    .dp           (dp),
    .an           (an)
);
 
// -------------------------------------------------------
// LED debug
// -------------------------------------------------------
assign led[15]   = nn_done;
assign led[14]   = image_ready;
assign led[13]   = uart_valid;
assign led[3:0]  = nn_result;
assign led[12:4] = 9'b0;
 
endmodule