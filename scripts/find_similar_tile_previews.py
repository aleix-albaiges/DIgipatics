from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalize(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def feature(path: Path, size: int = 128) -> dict[str, np.ndarray]:
    rgb = read_rgb(path, size)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    tissue = (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 25) & (hsv[:, :, 2] < 245)
    if int(tissue.sum()) < 100:
        tissue = np.ones(gray.shape, dtype=bool)
    mask = tissue.astype(np.uint8)

    lab_hist = cv2.calcHist([lab], [0, 1, 2], mask, [7, 7, 7], [0, 256, 0, 256, 0, 256])
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], mask, [10, 5, 5], [0, 180, 0, 256, 0, 256])

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    grad_hist, _ = np.histogram(
        angle[tissue],
        bins=12,
        range=(0, 2 * math.pi),
        weights=mag[tissue],
    )

    tissue_lab = lab[tissue].astype(np.float32)
    tissue_gray = gray[tissue].astype(np.float32)
    stats = np.array(
        [
            tissue.mean(),
            *tissue_lab.mean(axis=0),
            *tissue_lab.std(axis=0),
            tissue_gray.mean(),
            tissue_gray.std(),
            float(np.mean(mag[tissue])),
        ],
        dtype=np.float32,
    )

    return {
        "lab_hist": normalize(lab_hist),
        "hsv_hist": normalize(hsv_hist),
        "grad_hist": normalize(grad_hist),
        "stats": stats,
    }


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(max(0.0, 1.0 - float(np.sqrt(a * b).sum())))


def score(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    stats_scale = np.array([1, 255, 255, 255, 128, 128, 128, 255, 128, 128], dtype=np.float32)
    return (
        0.55 * bhattacharyya(candidate["lab_hist"], reference["lab_hist"])
        + 0.25 * bhattacharyya(candidate["hsv_hist"], reference["hsv_hist"])
        + 0.25 * bhattacharyya(candidate["grad_hist"], reference["grad_hist"])
        + 0.25 * float(np.mean(np.abs((candidate["stats"] - reference["stats"]) / stats_scale)))
    )


def make_contact_sheet(refs: list[Path], top_rows: list[tuple[float, float, Path, str]], output: Path) -> None:
    def tile(path: Path, label: str, size: int = 160) -> Image.Image:
        with Image.open(path) as img:
            im = img.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size + 36), "white")
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        ImageDraw.Draw(canvas).text((4, size + 4), label[:28], fill=(0, 0, 0))
        return canvas

    cols = 5
    cell_w, cell_h = 160, 196
    items: list[tuple[Path, str]] = [(p, f"REF {i:02d}") for i, p in enumerate(refs[:10], 1)]
    items.extend((p, f"{i:02d} {best:.3f}") for i, (best, _avg, p, _ref) in enumerate(top_rows[:10], 1))
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, (path, label) in enumerate(items):
        row, col = divmod(idx, cols)
        sheet.paste(tile(path, label, cell_w), (col * cell_w, row * cell_h))
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=Path, default=Path("ICS/tile preview"))
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--out", type=Path, default=Path("outputs/similar_tile_search"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    refs = image_paths(args.refs)
    candidates = image_paths(args.images)
    if not refs:
        raise SystemExit(f"No reference images found in {args.refs}")
    if not candidates:
        raise SystemExit(f"No candidate images found in {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"References: {len(refs)}", flush=True)
    print(f"Candidates: {len(candidates)}", flush=True)

    ref_features = [(path, feature(path)) for path in refs]
    rows: list[tuple[float, float, Path, str]] = []
    for index, path in enumerate(candidates, 1):
        candidate_feature = feature(path)
        candidate_scores = [(score(candidate_feature, ref_feature), ref_path.name) for ref_path, ref_feature in ref_features]
        best_score, best_ref = min(candidate_scores, key=lambda item: item[0])
        avg_score = float(np.mean([value for value, _name in candidate_scores]))
        rows.append((best_score, avg_score, path, best_ref))
        if index % 1000 == 0:
            print(f"Processed {index}/{len(candidates)}", flush=True)

    rows.sort(key=lambda item: item[0])

    csv_path = args.out / "similar_tiles_top50.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "best_score", "avg_score", "image", "best_reference"])
        for rank, (best_score, avg_score, path, best_ref) in enumerate(rows[:50], 1):
            writer.writerow([rank, f"{best_score:.6f}", f"{avg_score:.6f}", path.as_posix(), best_ref])

    top_dir = args.out / "top10"
    top_dir.mkdir(exist_ok=True)
    for old_file in top_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()
    for rank, (_best_score, _avg_score, path, _best_ref) in enumerate(rows[: args.top], 1):
        shutil.copy2(path, top_dir / f"{rank:02d}_{path.name}")

    sheet_path = args.out / "similar_tiles_contact_sheet.jpg"
    make_contact_sheet(refs, rows[: args.top], sheet_path)

    print("Top 10:", flush=True)
    for rank, (best_score, avg_score, path, best_ref) in enumerate(rows[: args.top], 1):
        print(f"{rank:02d}\t{best_score:.6f}\t{avg_score:.6f}\t{path.as_posix()}\tbest_ref={best_ref}", flush=True)
    print(f"CSV: {csv_path.as_posix()}", flush=True)
    print(f"Copied top10: {top_dir.as_posix()}", flush=True)
    print(f"Contact sheet: {sheet_path.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
