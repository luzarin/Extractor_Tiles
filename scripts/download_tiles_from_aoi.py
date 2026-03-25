from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import geopandas as gpd
import mercantile
import numpy as np
import rasterio
import requests
from requests import RequestException
from tqdm import tqdm
from rasterio.transform import from_bounds
from shapely.geometry import box


APPLE_TILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class RateLimiter:
    def __init__(self, max_rps: float):
        self.max_rps = float(max_rps)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.max_rps <= 0:
            return
        interval = 1.0 / self.max_rps
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + interval
                    return
                sleep_for = self._next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)


def _parse_access_key_and_version(raw_key: str) -> Tuple[str, Optional[int]]:
    key = (raw_key or "").strip()
    tile_version: Optional[int] = None

    # If user pasted full URL/query, extract both v and accessKey.
    if "accessKey=" in key and ("http" in key or "?" in key or "&" in key):
        if key.startswith("http://") or key.startswith("https://"):
            parsed = urllib.parse.urlparse(key)
            q = urllib.parse.parse_qs(parsed.query)
        else:
            q = urllib.parse.parse_qs(key.lstrip("?&"))
        if "v" in q and q["v"]:
            try:
                tile_version = int(q["v"][0])
            except ValueError:
                tile_version = None
        if "accessKey" in q and q["accessKey"]:
            key = q["accessKey"][0]

    if key.startswith("&accessKey="):
        key = key[len("&accessKey=") :]
    if "accessKey=" in key:
        key = key.split("accessKey=", 1)[1].split("&", 1)[0]
    key = key.strip()
    if not key:
        return key, tile_version
    if "%" in key:
        return key, tile_version
    return urllib.parse.quote(key, safe=""), tile_version


def _preflight_apple_tile_key(
    zoom: int,
    x_tile: int,
    y_tile: int,
    encoded_access_key: str,
    tile_version: int,
) -> None:
    test_url = (
        "https://sat-cdn1.apple-mapkit.com/tile"
        f"?style=7&size=1&scale=1&z={zoom}&x={x_tile}&y={y_tile}&v={tile_version}&accessKey={encoded_access_key}"
    )
    try:
        resp = requests.get(test_url, headers=APPLE_TILE_HEADERS, timeout=20)
    except RequestException as exc:
        raise RuntimeError(
            "Cannot connect to Apple tile CDN (sat-cdn1.apple-mapkit.com:443). "
            "Network/firewall/proxy is blocking outbound access."
        ) from exc
    if resp.status_code != 200:
        if resp.status_code == 410:
            raise RuntimeError(
                f"Apple tile preflight returned HTTP 410 (Gone). Access key expired/invalid "
                f"or tile version v={tile_version} is not valid for this session."
            )
        raise RuntimeError(f"Apple tile preflight failed (HTTP {resp.status_code}).")

    body_l = resp.content[:256].lower()
    if b"access denied" in body_l or b"forbidden" in body_l:
        raise RuntimeError("Apple tile preflight denied. Access key is invalid or expired.")

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        snippet = resp.text[:180].replace("\n", " ").replace("\r", " ")
        raise RuntimeError(f"Apple tile preflight returned non-image payload: {snippet}")


def load_aoi(
    aoi_file: str,
    aoi_layer: str = "",
    aoi_where: str = "",
) -> gpd.GeoDataFrame:
    read_kwargs = {}
    if aoi_layer:
        read_kwargs["layer"] = aoi_layer
    if aoi_where:
        read_kwargs["where"] = aoi_where

    try:
        gdf = gpd.read_file(aoi_file, **read_kwargs)
    except TypeError:
        read_kwargs.pop("where", None)
        gdf = gpd.read_file(aoi_file, **read_kwargs)

    if gdf.empty:
        raise ValueError(f"AOI file has no features: {aoi_file}")
    if gdf.crs is None:
        raise ValueError(f"AOI file has no CRS: {aoi_file}")

    gdf = gdf.to_crs(4326)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"AOI file has no valid geometry after cleanup: {aoi_file}")
    return gdf


def find_tiles(aoi_4326: gpd.GeoDataFrame, zoom: int) -> List[mercantile.Tile]:
    if hasattr(aoi_4326.geometry, "union_all"):
        aoi_geom = aoi_4326.geometry.union_all()
    else:
        aoi_geom = aoi_4326.unary_union
    minx, miny, maxx, maxy = aoi_geom.bounds

    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, [zoom]))
    filtered = []
    for tile in tiles:
        b = mercantile.bounds(tile)
        tile_poly = box(b.west, b.south, b.east, b.north)
        if aoi_geom.intersects(tile_poly):
            filtered.append(tile)
    filtered.sort(key=lambda t: (t.y, t.x))
    return filtered


def decode_tile_image(raw_bytes: bytes, tile_size: int) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    if img_bgr.shape[0] != tile_size or img_bgr.shape[1] != tile_size:
        img_bgr = cv2.resize(img_bgr, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def build_mosaic_from_tiles(
    tiles: List[mercantile.Tile],
    tile_url_template: str,
    tile_size: int,
    timeout_s: float,
    user_agent: str,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    if not tiles:
        raise ValueError("No tiles to download")

    x_min = min(t.x for t in tiles)
    x_max = max(t.x for t in tiles)
    y_min = min(t.y for t in tiles)
    y_max = max(t.y for t in tiles)

    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)

    session = requests.Session()
    headers = {"User-Agent": user_agent}
    total = len(tiles)
    downloaded = 0

    for idx, tile in enumerate(tiles, start=1):
        url = tile_url_template.format(z=tile.z, x=tile.x, y=tile.y)
        row = tile.y - y_min
        col = tile.x - x_min
        y1 = row * tile_size
        y2 = y1 + tile_size
        x1 = col * tile_size
        x2 = x1 + tile_size

        try:
            resp = session.get(url, timeout=timeout_s, headers=headers)
            resp.raise_for_status()
            img = decode_tile_image(resp.content, tile_size)
            if img is None:
                print(f"[WARN] Could not decode tile {tile.z}/{tile.x}/{tile.y}")
                continue
            mosaic[y1:y2, x1:x2, :] = img
            valid[y1:y2, x1:x2] = 255
            downloaded += 1
        except Exception as exc:
            print(f"[WARN] Tile download failed {tile.z}/{tile.x}/{tile.y}: {exc}")

        if idx % 50 == 0 or idx == total:
            print(f"[INFO] Downloaded {downloaded}/{total} processed tiles ({total} total)")

    if downloaded == 0:
        raise RuntimeError("No tile could be downloaded. Check tile URL template and connectivity.")
    return mosaic, valid, x_min, x_max, y_min, y_max


def _write_geotiff(
    output_tif: Path,
    mosaic_rgb: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs: str,
) -> None:
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    height, width = mosaic_rgb.shape[:2]
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(mosaic_rgb[:, :, 0], 1)
        dst.write(mosaic_rgb[:, :, 1], 2)
        dst.write(mosaic_rgb[:, :, 2], 3)
        dst.write(valid_mask.astype(np.uint8), 4)


def write_geotiff_from_xyz_grid(
    output_tif: Path,
    mosaic_rgb: np.ndarray,
    valid_mask: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    zoom: int,
) -> None:
    ul = mercantile.xy_bounds(mercantile.Tile(x=x_min, y=y_min, z=zoom))
    lr = mercantile.xy_bounds(mercantile.Tile(x=x_max, y=y_max, z=zoom))
    transform = from_bounds(ul.left, lr.bottom, lr.right, ul.top, mosaic_rgb.shape[1], mosaic_rgb.shape[0])
    _write_geotiff(output_tif, mosaic_rgb, valid_mask, transform, "EPSG:3857")


def write_geotiff_from_latlon_bounds(
    output_tif: Path,
    mosaic_rgb: np.ndarray,
    valid_mask: np.ndarray,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> None:
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, mosaic_rgb.shape[1], mosaic_rgb.shape[0])
    _write_geotiff(output_tif, mosaic_rgb, valid_mask, transform, "EPSG:4326")


def run_xyz_provider(
    aoi: gpd.GeoDataFrame,
    zoom: int,
    tile_url_template: str,
    output_tif: Path,
    tile_size: int,
    timeout_s: float,
    user_agent: str,
) -> None:
    tiles = find_tiles(aoi, zoom)
    print(f"[INFO] AOI selected {len(tiles)} tile(s) at z={zoom}")
    if not tiles:
        raise RuntimeError("AOI does not intersect any tile at this zoom level.")

    mosaic, valid, x_min, x_max, y_min, y_max = build_mosaic_from_tiles(
        tiles=tiles,
        tile_url_template=tile_url_template,
        tile_size=tile_size,
        timeout_s=timeout_s,
        user_agent=user_agent,
    )
    write_geotiff_from_xyz_grid(
        output_tif=output_tif,
        mosaic_rgb=mosaic,
        valid_mask=valid,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        zoom=zoom,
    )


def _load_jimutmap_api_class():
    spec = importlib.util.find_spec("jimutmap")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "jimutmap package not found. Install with: "
            "pip install --no-deps --no-build-isolation git+https://github.com/Jimut123/jimutmap.git"
        )

    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    module_path = pkg_dir / "jimutmap_1.py"
    if not module_path.exists():
        raise RuntimeError(f"jimutmap_1.py not found at {module_path}")

    module_name = "_jimutmap_runtime_jimutmap_1"
    module = sys.modules.get(module_name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        sys.modules[module_name] = module
        # Compatibility alias for legacy absolute imports used by jimutmap modules.
        sys.modules.setdefault("jimutmap_1", module)

    if not hasattr(module, "api"):
        raise RuntimeError("jimutmap_1 module has no 'api' class")
    return module.api


def _instantiate_jimutmap_api(
    api_class,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    threads: int,
    container_dir: Path,
    access_key: str,
):
    normalized_access_key, _ = _parse_access_key_and_version(access_key)
    kwargs = {
        "min_lat_deg": min_lat,
        "max_lat_deg": max_lat,
        "min_lon_deg": min_lon,
        "max_lon_deg": max_lon,
        "zoom": zoom,
        "verbose": True,
        "threads_": threads,
        "container_dir": str(container_dir),
    }
    if normalized_access_key:
        kwargs["ac_key"] = normalized_access_key
    try:
        return api_class(**kwargs)
    except TypeError:
        legacy_key = normalized_access_key
        return api_class(
            legacy_key,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            zoom=zoom,
            verbose=True,
            threads_=threads,
            container_dir=str(container_dir),
        )


def _stitch_jimutmap_sat_tiles(container_dir: Path, tile_size: int = 256):
    pattern = re.compile(r"(\d+)_(\d+)\.jpg$", re.IGNORECASE)
    entries = []
    for p in container_dir.glob("*.jpg"):
        m = pattern.search(p.name)
        if not m:
            continue
        x_tile = int(m.group(1))
        y_tile = int(m.group(2))
        entries.append((x_tile, y_tile, p))

    if not entries:
        raise RuntimeError(f"No satellite .jpg tiles found in {container_dir}")

    x_min = min(e[0] for e in entries)
    x_max = max(e[0] for e in entries)
    y_min = min(e[1] for e in entries)
    y_max = max(e[1] for e in entries)

    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)

    for x_tile, y_tile, p in entries:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if img_bgr.shape[0] != tile_size or img_bgr.shape[1] != tile_size:
            img_bgr = cv2.resize(img_bgr, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tile_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        row = y_tile - y_min
        col = x_tile - x_min
        y1 = row * tile_size
        y2 = y1 + tile_size
        x1 = col * tile_size
        x2 = x1 + tile_size
        mosaic[y1:y2, x1:x2, :] = tile_rgb
        valid[y1:y2, x1:x2] = 255

    return mosaic, valid, x_min, x_max, y_min, y_max


def _jimutmap_download_without_multiprocessing(
    download_obj,
    tile_list: List[tuple[int, int]],
    get_masks: bool,
    threads: int,
    tile_version: int,
) -> int:
    if not tile_list:
        raise RuntimeError("No tiles computed for AOI.")

    worker_count = max(1, int(threads))
    worker_count = min(worker_count, 64)
    print(f"[INFO] jimutmap tile count (AOI intersect): {len(tile_list)} (workers={worker_count})")

    def _task(coords):
        download_obj.get_img(coords, vNumber=int(tile_version), getMask=bool(get_masks))
        return 1

    completed = 0
    if worker_count == 1:
        for coords in tqdm(tile_list, desc="Downloading Tiles"):
            _task(coords)
            completed += 1
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_task, coords) for coords in tile_list]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading Tiles"):
                completed += 1
    return completed


def run_jimutmap_apple_provider(
    aoi: gpd.GeoDataFrame,
    zoom: int,
    output_tif: Path,
    container_dir: Path,
    threads: int,
    access_key: str,
    get_masks: bool,
    max_tiles: int,
    tile_version: int,
) -> None:
    try:
        api_class = _load_jimutmap_api_class()
    except Exception as exc:
        raise RuntimeError(
            "jimutmap is not installed or failed to import. "
            "Install with: pip install --no-deps --no-build-isolation git+https://github.com/Jimut123/jimutmap.git"
        ) from exc

    if hasattr(aoi.geometry, "union_all"):
        aoi_geom = aoi.geometry.union_all()
    else:
        aoi_geom = aoi.unary_union
    min_lon, min_lat, max_lon, max_lat = aoi_geom.bounds
    print(f"[INFO] AOI bounds (EPSG:4326): {min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")

    container_dir.mkdir(parents=True, exist_ok=True)
    for old in container_dir.iterdir():
        if old.is_file() and old.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                old.unlink()
            except OSError:
                pass

    access_key, version_from_key = _parse_access_key_and_version(access_key)
    if version_from_key is not None:
        tile_version = version_from_key
    if not access_key:
        raise RuntimeError(
            "No Apple access key provided. Use --jimutmap-access-key or set JIMUTMAP_ACCESS_KEY in environment."
        )
    if int(tile_version) <= 0:
        raise RuntimeError("Invalid tile version. Use --jimutmap-v with a positive integer.")
    print(f"[INFO] Apple tile version (v): {tile_version}")

    download_obj = _instantiate_jimutmap_api(
        api_class=api_class,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        zoom=zoom,
        threads=threads,
        container_dir=container_dir,
        access_key=access_key,
    )
    if access_key:
        if hasattr(download_obj, "ac_key"):
            download_obj.ac_key = access_key
        elif hasattr(download_obj, "access_key"):
            setattr(download_obj, "access_key", access_key)

    merc_tiles = find_tiles(aoi, zoom)
    tile_coords = [(t.x, t.y) for t in merc_tiles]
    if not tile_coords:
        raise RuntimeError("AOI has no intersecting tiles at this zoom.")
    if max_tiles > 0 and len(tile_coords) > max_tiles:
        raise RuntimeError(
            f"AOI at zoom={zoom} requires {len(tile_coords)} tiles, exceeding max_tiles={max_tiles}. "
            "Lower zoom (e.g., 15-16), split AOI, or raise --max-tiles."
        )
    # Fail fast if key is invalid/expired instead of running thousands of silent failed downloads.
    _preflight_apple_tile_key(
        zoom=zoom,
        x_tile=tile_coords[0][0],
        y_tile=tile_coords[0][1],
        encoded_access_key=access_key,
        tile_version=int(tile_version),
    )
    print("[INFO] Downloading Apple tiles using jimutmap...")
    downloaded = _jimutmap_download_without_multiprocessing(
        download_obj=download_obj,
        tile_list=tile_coords,
        get_masks=bool(get_masks),
        threads=int(threads),
        tile_version=int(tile_version),
    )
    jpg_count = len(list(container_dir.glob("*.jpg")))
    print(f"[INFO] jimutmap download tasks completed: {downloaded}, jpg files present: {jpg_count}")
    if jpg_count == 0:
        raise RuntimeError(
            "No JPEG tiles were downloaded. Access key is likely invalid/expired or Apple blocked requests."
        )

    print("[INFO] Stitching downloaded jimutmap tiles...")
    mosaic_rgb, valid, x_min, x_max, y_min, y_max = _stitch_jimutmap_sat_tiles(container_dir)
    write_geotiff_from_xyz_grid(
        output_tif=output_tif,
        mosaic_rgb=mosaic_rgb,
        valid_mask=valid,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        zoom=zoom,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tiles for an AOI and build a GeoTIFF mosaic.")
    parser.add_argument("--aoi-file", required=True, help="AOI vector file path (SHP/GPKG/GeoJSON).")
    parser.add_argument("--aoi-layer", default="", help="Optional AOI layer name.")
    parser.add_argument("--aoi-where", default="", help="Optional OGR WHERE filter for AOI selection.")
    parser.add_argument("--zoom", type=int, required=True, help="Tile zoom level.")
    parser.add_argument(
        "--provider",
        choices=["jimutmap_apple", "xyz_template"],
        default="jimutmap_apple",
        help="Tile provider backend.",
    )
    parser.add_argument(
        "--tile-url-template",
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="Only for provider=xyz_template. URL template with {z}, {x}, {y}.",
    )
    parser.add_argument("--output-tif", required=True, help="Output GeoTIFF path.")
    parser.add_argument("--tile-size", type=int, default=256, help="Only for provider=xyz_template.")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout per tile request.")
    parser.add_argument(
        "--user-agent",
        default="Fields-SAM-AOI-Downloader/1.0",
        help="HTTP User-Agent for provider=xyz_template.",
    )
    parser.add_argument(
        "--jimutmap-container-dir",
        default="",
        help="Optional temp directory for jimutmap downloaded tiles.",
    )
    parser.add_argument("--jimutmap-threads", type=int, default=20, help="jimutmap download threads.")
    parser.add_argument(
        "--jimutmap-v",
        type=int,
        default=10221,
        help="Apple tile version parameter 'v'. If omitted, defaults to 10221. If access key input includes v=..., that value is used.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=50000,
        help="Safety limit for AOI tile count. Set 0 to disable.",
    )
    parser.add_argument(
        "--jimutmap-access-key",
        default=os.environ.get("JIMUTMAP_ACCESS_KEY", ""),
        help="Optional Apple access key used by jimutmap (fallback to env JIMUTMAP_ACCESS_KEY).",
    )
    parser.add_argument(
        "--jimutmap-get-masks",
        action="store_true",
        help="If set, requests road masks where supported by jimutmap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_tif = Path(args.output_tif).resolve()
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    aoi = load_aoi(args.aoi_file, args.aoi_layer, args.aoi_where)

    if args.provider == "xyz_template":
        run_xyz_provider(
            aoi=aoi,
            zoom=args.zoom,
            tile_url_template=args.tile_url_template,
            output_tif=output_tif,
            tile_size=args.tile_size,
            timeout_s=args.timeout_s,
            user_agent=args.user_agent,
        )
    else:
        if args.jimutmap_container_dir:
            container_dir = Path(args.jimutmap_container_dir).resolve()
        else:
            container_dir = output_tif.parent / f"{output_tif.stem}_jimutmap_tiles"
        run_jimutmap_apple_provider(
            aoi=aoi,
            zoom=args.zoom,
            output_tif=output_tif,
            container_dir=container_dir,
            threads=args.jimutmap_threads,
            access_key=args.jimutmap_access_key,
            get_masks=args.jimutmap_get_masks,
            max_tiles=args.max_tiles,
            tile_version=args.jimutmap_v,
        )

    print(f"[INFO] Wrote tile mosaic: {output_tif}")


if __name__ == "__main__":
    main()
