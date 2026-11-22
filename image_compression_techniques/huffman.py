"""Lossless byte-wise Huffman coding, with a self-contained binary archive.

Usage: python huffman.py encode input.png output.huf
       python huffman.py decode output.huf restored.png
Only Python's standard library is needed. No built-in compressor is used.
"""
import argparse
from collections import Counter
import heapq
from pathlib import Path
import struct
import zlib

HEADER = struct.Struct(">4sQIH")  # magic, original size, CRC32, symbol count
ENTRY = struct.Struct(">BQ")     # byte and frequency


def build_tree(frequencies):
    """Stable tie-breaking lets the decoder rebuild exactly the same tree."""
    heap = [(count, symbol, symbol) for symbol, count in sorted(frequencies.items())]
    heapq.heapify(heap)
    serial = 256
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, (left[0] + right[0], serial, (left[2], right[2])))
        serial += 1
    return heap[0][2] if heap else None


def encode(data):
    """Return packed Huffman bits plus the frequency table needed to decode."""
    frequencies = Counter(data)
    root = build_tree(frequencies)
    codes = {}

    def visit(node, value=0, width=0):
        if isinstance(node, int):
            codes[node] = (value, max(width, 1))
        else:
            visit(node[0], value << 1, width + 1)
            visit(node[1], (value << 1) | 1, width + 1)

    if root is not None:
        visit(root)
    result = bytearray(HEADER.pack(b"HUF1", len(data), zlib.crc32(data), len(frequencies)))
    for symbol, count in sorted(frequencies.items()):
        result.extend(ENTRY.pack(symbol, count))
    value = width = 0
    for symbol in data:
        code, bits = codes[symbol]
        value = (value << bits) | code
        width += bits
        while width >= 8:
            width -= 8
            result.append((value >> width) & 255)
        value &= (1 << width) - 1
    if width:
        result.append(value << (8 - width))
    return bytes(result)


def decode(archive):
    """Restore the exact original bytes; reject malformed or corrupt archives."""
    if len(archive) < HEADER.size:
        raise ValueError("Truncated Huffman header")
    magic, size, checksum, count = HEADER.unpack_from(archive)
    if magic != b"HUF1" or count > 256:
        raise ValueError("Invalid Huffman header")
    offset = HEADER.size
    frequencies = {}
    for _ in range(count):
        if offset + ENTRY.size > len(archive):
            raise ValueError("Truncated frequency table")
        symbol, frequency = ENTRY.unpack_from(archive, offset)
        offset += ENTRY.size
        if frequency == 0 or symbol in frequencies:
            raise ValueError("Invalid frequency table")
        frequencies[symbol] = frequency
    if sum(frequencies.values()) != size:
        raise ValueError("Frequency total does not match original size")
    root = build_tree(frequencies)
    output = bytearray()
    node = root
    used = 0
    for byte in archive[offset:]:
        for shift in range(7, -1, -1):
            if len(output) == size:
                break
            bit = (byte >> shift) & 1
            used += 1
            if isinstance(root, int):
                if bit:
                    raise ValueError("Invalid single-symbol code")
                output.append(root)
            else:
                node = node[bit]
                if isinstance(node, int):
                    output.append(node)
                    node = root
        if len(output) == size:
            break
    payload = archive[offset:]
    if len(output) != size or len(payload) != (used + 7) // 8:
        raise ValueError("Truncated or trailing Huffman data")
    if used % 8 and payload[-1] & ((1 << (8 - used % 8)) - 1):
        raise ValueError("Invalid padding")
    if Counter(output) != frequencies or zlib.crc32(output) != checksum:
        raise ValueError("Huffman integrity check failed")
    return bytes(output)


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