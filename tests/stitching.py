import numpy as np


def tile_bounds(rank, n_tile, nx, ny, overlap):
    ix = rank % n_tile
    iy = rank // n_tile

    # Core (owned) region: same logic as box_mpi.py
    ix_min = ix * nx // n_tile
    ix_max = (ix + 1) * nx // n_tile
    iy_min = iy * ny // n_tile
    iy_max = (iy + 1) * ny // n_tile

    # Loaded region with overlap (clamped)
    lx_min = max(0, ix_min - overlap)
    lx_max = min(nx, ix_max + overlap)
    ly_min = max(0, iy_min - overlap)
    ly_max = min(ny, iy_max + overlap)

    return (ix_min, ix_max, iy_min, iy_max), (lx_min, lx_max, ly_min, ly_max)


def run_test(nx=100, ny=100, n_tile=2, overlap=8):
    comm_size = n_tile * n_tile

    # "Truth" global image: unique value per pixel
    truth = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)

    stitched = np.zeros_like(truth)
    ownership = np.zeros_like(truth, dtype=np.int32)

    for rank in range(comm_size):
        (ix_min, ix_max, iy_min, iy_max), (lx_min, lx_max, ly_min, ly_max) = tile_bounds(
            rank, n_tile, nx, ny, overlap
        )

        # Mimic per-rank local projection:
        # only loaded region contains valid values
        local = np.zeros_like(truth)
        local[ly_min:ly_max, lx_min:lx_max] = truth[ly_min:ly_max, lx_min:lx_max]

        # Inject garbage outside loaded region (to ensure mask protects stitching)
        outside_loaded = np.ones_like(truth, dtype=bool)
        outside_loaded[ly_min:ly_max, lx_min:lx_max] = False
        local[outside_loaded] = -1e9 - rank

        # Core mask (owned region)
        mask = np.zeros_like(truth, dtype=bool)
        mask[iy_min:iy_max, ix_min:ix_max] = True

        # Apply ownership mask (the critical step)
        local[~mask] = 0.0

        stitched += local
        ownership += mask.astype(np.int32)

        print(
            f"rank={rank} core=({iy_min}:{iy_max},{ix_min}:{ix_max}) "
            f"load=({ly_min}:{ly_max},{lx_min}:{lx_max})"
        )

    # Checks
    own_min, own_max = ownership.min(), ownership.max()
    print(f"ownership min/max = {own_min}/{own_max} (expected 1/1)")
    if own_min != 1 or own_max != 1:
        raise RuntimeError("Ownership mask has gaps or overlaps.")

    max_abs_err = np.max(np.abs(stitched - truth))
    print(f"max |stitched - truth| = {max_abs_err:.3e}")
    if not np.array_equal(stitched, truth):
        raise RuntimeError("Stitching failed: stitched map != truth map.")

    print("PASS: tiling + overlap + masking stitches back to exact original grid.")


if __name__ == "__main__":
    run_test(nx=100, ny=100, n_tile=2, overlap=8)