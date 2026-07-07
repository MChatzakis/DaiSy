import numpy as np
import os
import faiss
from gt_utils import formatFile_db, formatFile_query, find_dataset_pairs, parse_filename_for_config


def save_range_gt(out_dir, prefix, lims, I, D):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}.npz")
    np.savez_compressed(path, lims=lims, I=I, D=D)
    n_results = len(I)
    n_queries = len(lims) - 1
    avg = n_results / max(1, n_queries)
    print(f"  Saved {n_results} results ({avg:.1f} avg/query) -> {path}")


def generate_range_gt(dim, db_file, query_file, num_db, num_queries, db_name, r_values):
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)

    index = faiss.IndexFlatL2(dim)
    index.add(db)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    out_dir = os.path.join(project_root, "tests", "groundtruth", "Range")

    for r in r_values:
        print(f"  r={r:.4f}: running FAISS range_search ...")
        lims, D_flat, I_flat = index.range_search(queries, r)

        # sort each query's results by distance
        for qi in range(num_queries):
            sl = slice(int(lims[qi]), int(lims[qi + 1]))
            if lims[qi + 1] > lims[qi]:
                order = np.argsort(D_flat[sl])
                D_flat[sl] = D_flat[sl][order]
                I_flat[sl] = I_flat[sl][order]

        r_tag = f"{r:.4f}".replace('.', 'p')
        prefix = f"rangeGT_{db_name}_len{dim}_size{num_db}_q{num_queries}_r{r_tag}"
        save_range_gt(out_dir,
                      prefix,
                      lims.astype(np.int64),
                      I_flat.astype(np.int64),
                      D_flat.astype(np.float32))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    data_folder = os.path.join(project_root, 'data')

    dataset_pairs = find_dataset_pairs(data_folder)
    if not dataset_pairs:
        print(f"No dataset pairs found in {data_folder}")
        raise SystemExit(1)

    for db_path, query_path in dataset_pairs:
        db_name, dim, num_db = parse_filename_for_config(db_path)
        _, _, num_queries = parse_filename_for_config(query_path)

        # pick radii that yield roughly 5%, 20%, 50% result sets
        # for random N(0,1) data in d dims, avg pairwise L2sq ~ 2*d
        avg_l2sq = 2.0 * dim
        r_values = sorted({round(p * avg_l2sq, 4) for p in [0.05, 0.20, 0.50]})

        print(f"\nDataset: {db_name}, dim={dim}, n_db={num_db}, n_queries={num_queries}")
        print(f"Radii: {r_values}")
        generate_range_gt(dim, db_path, query_path, num_db, num_queries, db_name, r_values)

    print("\nDone.")
