// ============================================================
// nn_core.v  (v3  -  H1=8, H2=8, matches train_fixed_v2.py)
//
// CHANGE from previous version:
//   H1_SIZE 16 ? 8
//   H2_SIZE 16 ? 8
//   Accumulator width increased to handle full-precision sums
// ============================================================
module nn_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
 
    output reg  [8:0]  pixel_addr,
    input  wire [7:0]  pixel_data,
 
    output reg  [3:0]  result,
    output reg         done
);
 
// ?? Sizes  (MUST match train_fixed_v2.py H1_SIZE / H2_SIZE) ??
localparam INPUT_SIZE = 400;
localparam H1_SIZE    = 8;     // ? changed from 16
localparam H2_SIZE    = 8;     // ? changed from 16
localparam OUT_SIZE   = 10;
 
// ?? Weight ROMs ???????????????????????????????????????????????
// Paste hex files in the SAME folder as this .v file AND in:
//   <project>.runs/synth_1/
//   <project>.runs/impl_1/
// OR use full absolute paths (forward slashes on Windows too).
//
// w1: H1 × INPUT  = 8×400  = 3200 bytes
// w2: H2 × H1    = 8×8    =   64 bytes
// w3: OUT × H2   = 10×8   =   80 bytes
 
reg signed [7:0] w1 [0 : H1_SIZE*INPUT_SIZE-1];
reg signed [7:0] b1 [0 : H1_SIZE-1];
reg signed [7:0] w2 [0 : H2_SIZE*H1_SIZE-1];
reg signed [7:0] b2 [0 : H2_SIZE-1];
reg signed [7:0] w3 [0 : OUT_SIZE*H2_SIZE-1];
reg signed [7:0] b3 [0 : OUT_SIZE-1];
 
initial begin
    // ?? EDIT THESE PATHS TO YOUR FULL ABSOLUTE PATH ??????????
    // Example (Windows, forward slashes):
    // $readmemh("E:/FPGA_Encytion_Final/Nureal Network/Nureal Network.srcs/sources_1/new/w1.hex", w1);
    $readmemh("w1.hex", w1);
    $readmemh("b1.hex", b1);
    $readmemh("w2.hex", w2);
    $readmemh("b2.hex", b2);
    $readmemh("w3.hex", w3);
    $readmemh("b3.hex", b3);
end
 
// ?? Activation storage ????????????????????????????????????????
// 20-bit: enough for 400 × 127 = 50800 max sum in L1
reg signed [19:0] a1     [0 : H1_SIZE-1];
reg signed [19:0] a2     [0 : H2_SIZE-1];
reg signed [19:0] a3_out [0 : OUT_SIZE-1];
 
// ?? FSM states ????????????????????????????????????????????????
localparam ST_IDLE     = 4'd0;
localparam ST_L1_INIT  = 4'd1;
localparam ST_L1_ACC   = 4'd2;
localparam ST_L1_RELU  = 4'd3;
localparam ST_L2_INIT  = 4'd4;
localparam ST_L2_ACC   = 4'd5;
localparam ST_L2_RELU  = 4'd6;
localparam ST_L3_INIT  = 4'd7;
localparam ST_L3_ACC   = 4'd8;
localparam ST_L3_STORE = 4'd9;
localparam ST_ARGMAX   = 4'd10;
localparam ST_DONE     = 4'd11;
 
reg [3:0]  state = ST_IDLE;
reg [3:0]  n_idx = 0;    // current neuron (max OUT_SIZE=10 ? 4 bits)
reg [9:0]  i_idx = 0;    // current input  (max INPUT_SIZE=400)
reg signed [19:0] accum = 0;
 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state      <= ST_IDLE;
        done       <= 0;
        result     <= 0;
        pixel_addr <= 0;
        n_idx      <= 0;
        i_idx      <= 0;
        accum      <= 0;
    end else begin
        done <= 0;
 
        case (state)
 
        // ?? Wait for image_ready ??????????????????????????????
        ST_IDLE: begin
            if (start) begin
                n_idx <= 0;
                state <= ST_L1_INIT;
            end
        end
 
        // ?????????? LAYER 1  (400 inputs ? 8 neurons) ????????
        ST_L1_INIT: begin
            if (n_idx == H1_SIZE[3:0]) begin
                n_idx <= 0;
                state <= ST_L2_INIT;
            end else begin
                // Bias: Q4.4 ? shift left 4 to match pixel sum scale
                accum      <= {{12{b1[n_idx][7]}}, b1[n_idx], 4'b0};
                i_idx      <= 0;
                pixel_addr <= 0;
                state      <= ST_L1_ACC;
            end
        end
 
        ST_L1_ACC: begin
            if (i_idx > 0) begin
                // Pixel: 0x00=background, 0xFF=foreground ? treat as binary 0/1
                if (pixel_data != 8'h00)
                    accum <= accum + {{12{w1[n_idx*INPUT_SIZE + (i_idx-1)][7]}},
                                       w1[n_idx*INPUT_SIZE + (i_idx-1)]};
            end
            if (i_idx == INPUT_SIZE) begin
                state <= ST_L1_RELU;
            end else begin
                pixel_addr <= i_idx[8:0];
                i_idx      <= i_idx + 1;
            end
        end
 
        ST_L1_RELU: begin
            a1[n_idx] <= accum[19] ? 20'd0 : accum;
            n_idx     <= n_idx + 1;
            state     <= ST_L1_INIT;
        end
 
        // ?????????? LAYER 2  (8 inputs ? 8 neurons) ??????????
        ST_L2_INIT: begin
            if (n_idx == H2_SIZE[3:0]) begin
                n_idx <= 0;
                state <= ST_L3_INIT;
            end else begin
                accum <= {{12{b2[n_idx][7]}}, b2[n_idx], 4'b0};
                i_idx <= 0;
                state <= ST_L2_ACC;
            end
        end
 
        ST_L2_ACC: begin
            if (i_idx > 0)
                accum <= accum + ($signed(w2[n_idx*H1_SIZE + (i_idx-1)])
                                  * $signed(a1[i_idx-1][14:4]));
            // a1 >>> 4 to compensate for Q4.4 scale
            if (i_idx == H2_SIZE[9:0])
                state <= ST_L2_RELU;
            else
                i_idx <= i_idx + 1;
        end
 
        ST_L2_RELU: begin
            a2[n_idx] <= accum[19] ? 20'd0 : accum;
            n_idx     <= n_idx + 1;
            state     <= ST_L2_INIT;
        end
 
        // ?????????? LAYER 3  (8 inputs ? 10 outputs) ?????????
        ST_L3_INIT: begin
            if (n_idx == OUT_SIZE[3:0]) begin
                n_idx <= 0;
                state <= ST_ARGMAX;
            end else begin
                accum <= {{12{b3[n_idx][7]}}, b3[n_idx], 4'b0};
                i_idx <= 0;
                state <= ST_L3_ACC;
            end
        end
 
        ST_L3_ACC: begin
            if (i_idx > 0)
                accum <= accum + ($signed(w3[n_idx*H2_SIZE + (i_idx-1)])
                                  * $signed(a2[i_idx-1][14:4]));
            if (i_idx == H2_SIZE[9:0])
                state <= ST_L3_STORE;
            else
                i_idx <= i_idx + 1;
        end
 
        ST_L3_STORE: begin
            a3_out[n_idx] <= accum;
            n_idx         <= n_idx + 1;
            state         <= ST_L3_INIT;
        end
 
        // ?????????? ARGMAX ????????????????????????????????????
        ST_ARGMAX: begin
            begin : argmax_blk
                integer k;
                reg signed [19:0] mx;
                reg [3:0]         mi;
                mx = a3_out[0];
                mi = 4'd0;
                for (k = 1; k < OUT_SIZE; k = k+1) begin
                    if ($signed(a3_out[k]) > $signed(mx)) begin
                        mx = a3_out[k];
                        mi = k[3:0];
                    end
                end
                result <= mi;
            end
            state <= ST_DONE;
        end
 
        ST_DONE: begin
            done  <= 1;
            state <= ST_IDLE;
        end
 
        default: state <= ST_IDLE;
        endcase
    end
end
 
endmodule