import os

def split_jsonl(path, chunk_size=30 * 1024 * 1024, out_dir=None):
    """
    Split a .jsonl file sequentially into parts of at most `chunk_size` bytes,
    always breaking on line boundaries so each part is itself valid JSONL.
    Parts: <stem>.part001.jsonl, <stem>.part002.jsonl, ...
    Returns the list of part paths in order.
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(path))[0]
    parts = []
    part_num = 1
    buf = []            # list of raw line-bytes for the current part
    buf_size = 0        # bytes currently buffered

    def flush():
        nonlocal buf, buf_size, part_num
        if not buf:
            return
        out_path = os.path.join(out_dir, f"{stem}.part{part_num:03d}.jsonl")
        with open(out_path, "wb") as dst:
            dst.writelines(buf)
        parts.append(out_path)
        print(f"wrote {out_path}  ({buf_size:,} bytes, ends part {part_num})")
        part_num += 1
        buf, buf_size = [], 0

    with open(path, "rb") as src:
        for line in src:                       # line includes its trailing b"\n"
            line_len = len(line)

            # If a single line is bigger than the cap, it can't be made to fit.
            if line_len > chunk_size:
                flush()                        # close whatever's open first
                print(f"  WARNING: one line is {line_len:,} bytes "
                      f"(> chunk_size {chunk_size:,}); it gets its own oversized part.")

            # Would adding this line overflow the current part? If so, flush first.
            if buf and buf_size + line_len > chunk_size:
                flush()

            buf.append(line)
            buf_size += line_len

    flush()   # write the final part

    print(f"\nDone: {len(parts)} parts, {os.path.getsize(path):,} bytes total")
    return parts

if __name__ == "__main__":
    split_jsonl("crab-train.jsonl")