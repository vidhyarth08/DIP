"""Lossless adaptive integer arithmetic coding of bytes (standard library only).

Usage: python arithmetic.py encode input.png output.ari
       python arithmetic.py decode output.ari restored.png
Uses 32-bit intervals and renormalization, not floating-point fractions.
"""
import argparse
from bisect import bisect_right
from pathlib import Path
import struct
import zlib

FULL = 1 << 32
HALF = FULL >> 1
QUARTER = HALF >> 1
HEADER = struct.Struct(">4sQQI")  # magic, original bytes, payload bits, CRC32


class Bits:
    def __init__(self, data=b"", length=0):
        self.data = bytearray(data)
        self.length = length
        self.position = 0

    def write(self, bit):
        if self.length % 8 == 0:
            self.data.append(0)
        self.data[-1] |= bit << (7 - self.length % 8)
        self.length += 1

    def read(self):
        # Arithmetic termination implicitly pads the encoded fraction with zeros.
        index = self.position
        self.position += 1
        return (self.data[index // 8] >> (7 - index % 8)) & 1 if index < self.length else 0


class Model:
    """All 256 bytes start equally likely; encoder and decoder update together."""
    def __init__(self):
        self.frequencies = [1] * 256
        self.cumulative = list(range(257))

    def update(self, symbol):
        self.frequencies[symbol] += 1
        if self.cumulative[-1] + 1 >= 16384:
            self.frequencies = [(count + 1) // 2 for count in self.frequencies]
            self.cumulative = [0]
            for count in self.frequencies:
                self.cumulative.append(self.cumulative[-1] + count)
        else:
            for index in range(symbol + 1, 257):
                self.cumulative[index] += 1


def encode(data):
    model = Model()
    bits = Bits()
    low, high, pending = 0, FULL - 1, 0

    def emit(bit):
        nonlocal pending
        bits.write(bit)
        for _ in range(pending):
            bits.write(1 - bit)
        pending = 0

    for symbol in data:
        cumulative = model.cumulative
        width = high - low + 1
        high = low + width * cumulative[symbol + 1] // cumulative[-1] - 1
        low += width * cumulative[symbol] // cumulative[-1]
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
        model.update(symbol)
    pending += 1
    emit(0 if low < QUARTER else 1)
    return HEADER.pack(b"ARI1", len(data), bits.length, zlib.crc32(data)) + bytes(bits.data)


def decode(archive):
    if len(archive) < HEADER.size:
        raise ValueError("Truncated arithmetic header")
    magic, size, length, checksum = HEADER.unpack_from(archive)
    payload = archive[HEADER.size:]
    if magic != b"ARI1" or length < 2 or len(payload) != (length + 7) // 8:
        raise ValueError("Invalid or truncated arithmetic archive")
    if length % 8 and payload[-1] & ((1 << (8 - length % 8)) - 1):
        raise ValueError("Invalid padding")
    bits = Bits(payload, length)
    model = Model()
    low, high, value = 0, FULL - 1, 0
    for _ in range(32):
        value = (value << 1) | bits.read()
    result = bytearray()
    for _ in range(size):
        cumulative = model.cumulative
        width = high - low + 1
        scaled = ((value - low + 1) * cumulative[-1] - 1) // width
        symbol = bisect_right(cumulative, scaled) - 1
        if not 0 <= symbol < 256:
            raise ValueError("Invalid arithmetic interval")
        result.append(symbol)
        high = low + width * cumulative[symbol + 1] // cumulative[-1] - 1
        low += width * cumulative[symbol] // cumulative[-1]
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
            value = value * 2 + bits.read()
        model.update(symbol)
    if zlib.crc32(result) != checksum:
        raise ValueError("Arithmetic integrity check failed")
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    try:
        data = args.input.read_bytes()
        result = encode(data) if args.mode == "encode" else decode(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(result)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{len(data):,} bytes -> {len(result):,} bytes: {args.output}")


if __name__ == "__main__":
    main()