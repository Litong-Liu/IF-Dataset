import os, re, glob, hashlib

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def join_jsonl(stem, out_path, in_dir="."):
    """
    Reassemble <stem>.part001.jsonl, .part002.jsonl, ... into `out_path`.
    `stem` is the original filename without its .jsonl extension.
    """
    pattern = os.path.join(in_dir, f"{stem}.part*.jsonl")
    parts = glob.glob(pattern)
    if not parts:
        raise FileNotFoundError(f"No parts found matching {pattern}")

    # Sort by the numeric part index, not lexically, to be safe.
    def part_index(p):
        m = re.search(r"\.part(\d+)\.jsonl$", p)
        return int(m.group(1)) if m else -1
    parts.sort(key=part_index)

    with open(out_path, "wb") as dst:
        for part in parts:
            with open(part, "rb") as src:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(block)
            print(f"appended {part}")

    print(f"\nRecovered {out_path}  ({os.path.getsize(out_path):,} bytes)")
    return out_path

if __name__ == "__main__":
    join_jsonl("yourfile", "yourfile_restored.jsonl")

    # Optional: confirm the recovery is bit-for-bit identical -> prints True
    print("identical:", _sha256("yourfile.jsonl") == _sha256("yourfile_restored.jsonl"))