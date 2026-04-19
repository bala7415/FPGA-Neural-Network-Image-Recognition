// ============================================================
// seven_seg_ctrl.v  -  8-digit multiplexed 7-segment driver
//
// Nexys4 has two 4-digit common-anode displays wired as one
// 8-digit display. AN0..AN7 are ACTIVE-LOW enables.
// Segment signals CA..CG (seg[0..6]) are ACTIVE-LOW.
//
// This module:
//   1. Divides 100 MHz clock to ~1 KHz refresh (each digit gets
//      125 µs on-time ? full refresh every 1 ms, no flicker).
//   2. Cycles through digits 0-7 (AN0=rightmost).
//   3. Drives the correct 7-segment pattern for each digit.
//
// Digit encoding (4-bit input ? seg pattern):
//   0-9  : numeric digits
//   0xA  : '-'  (minus/dash)
//   0xB  : 'P'
//   0xC  : 'C'
//   0xD  : 'd'
//   0xE  : 'A' (looks like A on 7-seg)
//   0xF  : ' ' (blank)
// ============================================================
module seven_seg_ctrl (
    input  wire       clk,
    input  wire       rst,
 
    // 4-bit encoded digit values, digit7=leftmost, digit0=rightmost
    input  wire [3:0] digit7,
    input  wire [3:0] digit6,
    input  wire [3:0] digit5,
    input  wire [3:0] digit4,
    input  wire [3:0] digit3,
    input  wire [3:0] digit2,
    input  wire [3:0] digit1,
    input  wire [3:0] digit0,
 
    input  wire [7:0] dp_en,   // decimal point enable (bit7=left)
 
    output reg  [6:0] seg,     // seg[6]=CA seg[0]=CG (active low)
    output reg        dp,      // decimal point (active low)
    output reg  [7:0] an       // AN7..AN0 (active low)
);
 
// -----------------------------------------------------------
// Clock divider: 100 MHz ? ~1 KHz (period = 100_000 clocks)
// Each of 8 digits gets 1/8 of the period ? 12_500 clocks
// -----------------------------------------------------------
localparam DIV_COUNT = 12_500;   // ~12.5 µs per digit slot
 
reg [13:0] div_cnt  = 0;
reg [2:0]  digit_sel = 0;   // 0..7
 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        div_cnt   <= 0;
        digit_sel <= 0;
    end else begin
        if (div_cnt == DIV_COUNT - 1) begin
            div_cnt   <= 0;
            digit_sel <= digit_sel + 1;  // wraps 7?0
        end else
            div_cnt <= div_cnt + 1;
    end
end
 
// -----------------------------------------------------------
// Digit mux: select current digit value & AN
// -----------------------------------------------------------
reg [3:0] cur_digit;
reg       cur_dp;
 
always @(*) begin
    case (digit_sel)
        3'd7: begin cur_digit = digit7; cur_dp = dp_en[7]; end
        3'd6: begin cur_digit = digit6; cur_dp = dp_en[6]; end
        3'd5: begin cur_digit = digit5; cur_dp = dp_en[5]; end
        3'd4: begin cur_digit = digit4; cur_dp = dp_en[4]; end
        3'd3: begin cur_digit = digit3; cur_dp = dp_en[3]; end
        3'd2: begin cur_digit = digit2; cur_dp = dp_en[2]; end
        3'd1: begin cur_digit = digit1; cur_dp = dp_en[1]; end
        3'd0: begin cur_digit = digit0; cur_dp = dp_en[0]; end
        default: begin cur_digit = 4'hF; cur_dp = 1'b0; end
    endcase
end
 
// Active-low AN decoder
always @(*) begin
    an = 8'b1111_1111;          // all off by default
    an[digit_sel] = 1'b0;       // enable current digit (active low)
end
 
// -----------------------------------------------------------
// 7-segment decoder
// Segment mapping: seg = {CA, CB, CC, CD, CE, CF, CG}
//                        bit6 bit5 bit4 bit3 bit2 bit1 bit0
// Active LOW: 0=segment ON, 1=segment OFF
//
//    _
//   |_|   A=top  B=top-right  C=bot-right  D=bottom
//   |_|   E=bot-left  F=top-left  G=middle
//
//  seg[6]=A  seg[5]=B  seg[4]=C  seg[3]=D
//  seg[2]=E  seg[1]=F  seg[0]=G
// -----------------------------------------------------------
always @(*) begin
    case (cur_digit)
        //           ABCDEFG  (0=ON, 1=OFF)
        4'h0: seg = 7'b000_0001;   // 0
        4'h1: seg = 7'b100_1111;   // 1
        4'h2: seg = 7'b001_0010;   // 2  (was 0010010)
        4'h3: seg = 7'b000_0110;   // 3
        4'h4: seg = 7'b100_1100;   // 4
        4'h5: seg = 7'b010_0100;   // 5
        4'h6: seg = 7'b010_0000;   // 6
        4'h7: seg = 7'b000_1111;   // 7
        4'h8: seg = 7'b000_0000;   // 8
        4'h9: seg = 7'b000_0100;   // 9
        4'hA: seg = 7'b111_1110;   // '-'  (only G segment)
        4'hB: seg = 7'b000_1100;   // 'P'  (A B E F G lit)
        4'hC: seg = 7'b011_0001;   // 'C'  (A D E F)
        4'hD: seg = 7'b100_0010;   // 'd'  (B C D E G)
        4'hE: seg = 7'b000_1000;   // 'A'  (A B C E F G)
        4'hF: seg = 7'b111_1111;   // ' '  (blank)
        default: seg = 7'b111_1111;
    endcase
end
 
// Decimal point (active low)
always @(*) begin
    dp = ~cur_dp;
end
 
endmodule