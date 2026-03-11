import numpy as np
import pandas as pd

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between points on the earth
    (specified in decimal degrees). Vectorized for NumPy arrays.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def load_and_prepare_data(cfg):
    """
    Loads all necessary data files and performs preprocessing.

    Args:
        cfg: The configuration module.

    Returns:
        A dictionary containing all loaded and processed data.
    """
    print("Loading and preparing data...")

    # Read population data
    population_df = pd.read_csv(cfg.filename_population)
    
    if cfg.dataset_name == "NE-England":
        population_df = population_df.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon',
            'patch_id': 'id'
        })
        population_df['name'] = 'patch_' + population_df['id'].astype(str)
        if 'group' not in population_df.columns:
            population_df['group'] = '1'
            
    if hasattr(cfg, 'population_factor') and cfg.population_factor != 1.0:
        population_df['population'] = (population_df['population'] * cfg.population_factor).round().astype(int)
        print(f"Applied population scaling factor: {cfg.population_factor}")

    population_df.reset_index(drop=True, inplace=True)
    population_df['group'] = population_df['group'].astype(str)
    n_patches = len(population_df)

    # Compute pairwise distance matrix
    coords = population_df.sort_values('id')[['lon', 'lat']].to_numpy()
    lons = coords[:, 0]
    lats = coords[:, 1]
    lons1, lats1 = lons[:, np.newaxis], lats[:, np.newaxis]
    lons2, lats2 = lons[np.newaxis, :], lats[np.newaxis, :]
    distance_matrix_km = haversine_np(lons1, lats1, lons2, lats2)
    print(f"Computed distance matrix between all {n_patches} districts.")

    # Read and process flow matrix
    if cfg.dataset_name == "NE-England":
        flow_matrix_df = pd.read_csv(cfg.filename_flow_matrix, index_col=0)
    else:
        flow_matrix_df = pd.read_csv(cfg.filename_flow_matrix)
        if flow_matrix_df.shape[1] == n_patches + 1:
            print(f"  -> Detected extra column in flow matrix (likely index). Dropping first column.")
            flow_matrix_df = flow_matrix_df.iloc[:, 1:]
        
    if hasattr(cfg, 'population_factor') and cfg.population_factor != 1.0:
        flow_matrix_df = (flow_matrix_df * cfg.population_factor).round().astype(int)
        print(f"Applied flow matrix scaling factor: {cfg.population_factor}")

    flow_matrix_df.reset_index(drop=True, inplace=True)
    if cfg.dataset_name != "NE-England":
        flow_matrix_df = flow_matrix_df.set_axis(population_df['name'].to_list(), axis=1)
    M_ij_np = flow_matrix_df.to_numpy()

    # Prepare mobility matrix D_ij
    sum_Mij = M_ij_np.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        D_ij_np = np.nan_to_num(M_ij_np / sum_Mij)
    np.fill_diagonal(D_ij_np, 1.0 - (D_ij_np.sum(axis=1) - np.diag(D_ij_np)))

    # Vectorized calculation of flow_ij
    N_i = population_df['population'].to_numpy()
    N_ij_np = (D_ij_np * N_i[:, np.newaxis]).round().astype(int)

    patch_indices = np.arange(n_patches)
    patch_i, patch_j = np.meshgrid(patch_indices, patch_indices, indexing='ij')

    flow_ij_df = pd.DataFrame({
        'patch_i': patch_i.ravel(),
        'patch_j': patch_j.ravel(),
        'N_ij': N_ij_np.ravel()
    })
    flow_ij_df['patch_i'] = flow_ij_df['patch_i'].astype(int)
    flow_ij_df['patch_j'] = flow_ij_df['patch_j'].astype(int)
    flow_ij_df['N_ij'] = flow_ij_df['N_ij'].astype(int)

    data = {
        "population_df": population_df,
        "n_patches": n_patches,
        "distance_matrix_km": distance_matrix_km,
        "flow_ij_df": flow_ij_df,
        "N_ij_initial": N_ij_np
    }
    print("Data loading complete.")
    return data