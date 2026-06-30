#!/usr/bin/env python3
import argparse
import struct
import zlib


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def read_chunks(path):
    with open(path, "rb") as f:
        if f.read(8) != PNG_SIG:
            raise ValueError("Not a PNG file")
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]
            ctype = f.read(4)
            data = f.read(length)
            f.read(4)  # crc
            yield ctype, data


def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def unfilter_scanlines(raw, width, height, bpp):
    stride = width * bpp
    out = bytearray(height * stride)
    prev = bytearray(stride)
    i = 0
    for y in range(height):
        ftype = raw[i]
        i += 1
        line = raw[i:i + stride]
        i += stride
        recon = bytearray(stride)
        if ftype == 0:
            recon[:] = line
        elif ftype == 1:
            for x in range(stride):
                left = recon[x - bpp] if x >= bpp else 0
                recon[x] = (line[x] + left) & 0xFF
        elif ftype == 2:
            for x in range(stride):
                recon[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:
            for x in range(stride):
                left = recon[x - bpp] if x >= bpp else 0
                up = prev[x]
                recon[x] = (line[x] + ((left + up) // 2)) & 0xFF
        elif ftype == 4:
            for x in range(stride):
                left = recon[x - bpp] if x >= bpp else 0
                up = prev[x]
                up_left = prev[x - bpp] if x >= bpp else 0
                recon[x] = (line[x] + paeth(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported filter type: {ftype}")
        out[y * stride:(y + 1) * stride] = recon
        prev = recon
    return out


def decode_png(path):
    width = height = None
    bit_depth = color_type = None
    palette = None
    trns = None
    idat = bytearray()

    for ctype, data in read_chunks(path):
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", data)
        elif ctype == b"PLTE":
            palette = data
        elif ctype == b"tRNS":
            trns = data
        elif ctype == b"IDAT":
            idat.extend(data)
        elif ctype == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("Missing IHDR")
    if bit_depth != 8:
        raise ValueError("Only 8-bit PNG supported")

    if color_type == 3:
        if palette is None:
            raise ValueError("Indexed PNG missing PLTE")
        bpp = 1
    elif color_type == 2:
        bpp = 3
    elif color_type == 6:
        bpp = 4
    else:
        raise ValueError(f"Unsupported color type: {color_type}")

    raw = zlib.decompress(bytes(idat))
    pixels = unfilter_scanlines(raw, width, height, bpp)

    rgb = bytearray(width * height * 3)
    if color_type == 3:
        for i in range(width * height):
            idx = pixels[i]
            base = idx * 3
            r = palette[base]
            g = palette[base + 1]
            b = palette[base + 2]
            a = 255
            if trns and idx < len(trns):
                a = trns[idx]
            if a < 255:
                # composite on white
                r = (r * a + 255 * (255 - a)) // 255
                g = (g * a + 255 * (255 - a)) // 255
                b = (b * a + 255 * (255 - a)) // 255
            rgb[i * 3:i * 3 + 3] = bytes([r, g, b])
    elif color_type == 2:
        rgb[:] = pixels
    elif color_type == 6:
        for i in range(width * height):
            r, g, b, a = pixels[i * 4:i * 4 + 4]
            if a < 255:
                r = (r * a + 255 * (255 - a)) // 255
                g = (g * a + 255 * (255 - a)) // 255
                b = (b * a + 255 * (255 - a)) // 255
            rgb[i * 3:i * 3 + 3] = bytes([r, g, b])

    return width, height, rgb


def sharpen_rgb(width, height, rgb, strength=1.0):
    out = bytearray(len(rgb))
    w = width
    h = height

    def idx(x, y):
        return (y * w + x) * 3

    for y in range(h):
        for x in range(w):
            for c in range(3):
                center = rgb[idx(x, y) + c]
                up = rgb[idx(x, y - 1) + c] if y > 0 else center
                down = rgb[idx(x, y + 1) + c] if y < h - 1 else center
                left = rgb[idx(x - 1, y) + c] if x > 0 else center
                right = rgb[idx(x + 1, y) + c] if x < w - 1 else center

                val = (5 * center - up - down - left - right)
                val = int(center + strength * (val - center))
                if val < 0:
                    val = 0
                if val > 255:
                    val = 255
                out[idx(x, y) + c] = val
    return out


def write_png(path, width, height, rgb):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(rgb[start:start + stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=6)

    png = PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def parse_args():
    p = argparse.ArgumentParser(description="Sharpen a PNG image (no external deps).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strength", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    width, height, rgb = decode_png(args.input)
    sharpened = sharpen_rgb(width, height, rgb, strength=args.strength)
    write_png(args.output, width, height, sharpened)


if __name__ == "__main__":
    main()
