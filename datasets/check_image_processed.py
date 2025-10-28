from pathlib import Path
import json
import re

# Directories
src_dir = Path("/root/autodl-tmp/data/SWiG_jsons")
proc_dir = Path("/root/autodl-tmp/data/Blip2_features")

def collect_image_stems_from_folder(folder: Path) -> set:
    return {
        path.stem
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }

def collect_image_stems_from_jsons(folder: Path) -> set:
    image_stems: set = set()
    json_files = [p for p in folder.rglob("*.json") if p.is_file()]
    candidate_keys = {
        "image", "img", "image_id", "image_name", "image_filename", "filename", "file_name"
    }
    jpg_pattern = re.compile(r"(^|[\\/\\\\])([^\\/\\\\]+)\.(?:jpe?g)$", re.IGNORECASE)

    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Fallback: regex scan the raw text for jpg-like substrings
            try:
                text = jf.read_text(encoding="utf-8", errors="ignore")
                for m in re.finditer(r"([^\\/\\\\]+)\.(?:jpe?g)", text, re.IGNORECASE):
                    image_stems.add(Path(m.group(1)).stem)
            except Exception:
                pass
            continue

        def consider_string(s: str):
            s_lower = s.lower()
            if s_lower.endswith((".jpg", ".jpeg")):
                image_stems.add(Path(s).stem)

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # If the KEY itself looks like an image filename, capture it
                    if isinstance(k, str) and k.lower().endswith((".jpg", ".jpeg")):
                        image_stems.add(Path(k).stem)
                    # Prefer known keys for values
                    if isinstance(k, str) and k in candidate_keys:
                        if isinstance(v, str):
                            consider_string(v)
                    # Generic scan
                    if isinstance(v, str):
                        consider_string(v)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
            elif isinstance(obj, str):
                consider_string(obj)

        try:
            walk(data)
        except Exception:
            # As a last resort, regex over the dumped string
            try:
                text = json.dumps(data)
                for m in re.finditer(r"([^\\/\\\\]+)\.(?:jpe?g)", text, re.IGNORECASE):
                    image_stems.add(Path(m.group(1)).stem)
            except Exception:
                pass

    return image_stems

# 1) Try to collect .jpgs directly in the folder
image_stems = collect_image_stems_from_folder(src_dir)

# 2) If none found, attempt to extract from JSON files
if not image_stems:
    image_stems = collect_image_stems_from_jsons(src_dir)

# Gather processed stems from target.
# If you know the exact extensions, restrict them below to avoid noise.
processed_exts = {".npy", ".npz", ".pt", ".pth", ".pkl", ".json", ".txt"}
processed_stems = {
    p.stem
    for p in proc_dir.rglob("*")
    if p.is_file() and (not processed_exts or p.suffix.lower() in processed_exts)
}

# Find which images are missing in processed outputs
missing_stems = sorted(image_stems - processed_stems)

# Persist results (printing may be suppressed in some environments)
missing_out_path = Path("/root/OV_GSR/missing_images.txt")
missing_out_path.write_text("\n".join(stem + ".jpg" for stem in missing_stems), encoding="utf-8")

summary = (
    f"Source dir: {src_dir}\n"
    f"Processed dir: {proc_dir}\n"
    f"Images detected (unique stems): {len(image_stems)}\n"
    f"Processed files considered (unique stems): {len(processed_stems)}\n"
    f"Missing images: {len(missing_stems)}\n"
    f"Missing list written to: {missing_out_path}\n"
)

summary_path = Path("/root/OV_GSR/check_image_processed_summary.txt")
summary_path.write_text(summary, encoding="utf-8")

print(summary)