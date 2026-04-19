#!/usr/bin/env python3
"""
send_image.py  –  PC-side UART sender
======================================
Sends a 20x20 image as 400 bytes over UART to the FPGA.
 
Usage:
    python send_image.py P0_a.png --port COM3
    python send_image.py P0_a.png --port /dev/ttyUSB0
 
The FPGA accumulates exactly 400 bytes, then automatically
triggers the Neural Network inference. No reprogramming needed
– just run this script again for the next image.
"""

import serial, sys, time, argparse
from PIL import Image
import numpy as np

def image_to_bytes(img_path: str) -> bytes:
    """
    Load image, resize to 20x20, threshold, return 400 bytes.
    Pixel > 127  → 0xFF  (foreground)
    Pixel <= 127 → 0x00  (background)
    """
    img = Image.open(img_path).convert('L').resize((20, 20))
    pixels = np.array(img).flatten()
    return bytes([0xFF if p > 127 else 0x00 for p in pixels])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to PNG image file')
    parser.add_argument('--port', default='COM3', help='Serial port')
    parser.add_argument('--baud', default=115200, type=int)
    args = parser.parse_args()

    data = image_to_bytes(args.image)
    print(f"Image: {args.image}")
    print(f"Pixels: {len(data)} bytes  (20x20)")
    print(f"Sending on {args.port} @ {args.baud} baud...")

    # Print pixel grid for visual check
    grid = np.frombuffer(data, dtype=np.uint8).reshape(20, 20)
    print("\nPixel preview (# = foreground):")
    for row in grid:
        print(''.join('#' if p else '.' for p in row))

    with serial.Serial(args.port, args.baud, timeout=2) as ser:
        time.sleep(0.1)  # let port settle
        n = ser.write(data)
        print(f"\nSent {n} bytes. FPGA will run NN now.")

if __name__ == '__main__':
    main()