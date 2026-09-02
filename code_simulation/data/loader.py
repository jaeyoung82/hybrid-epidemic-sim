"""Centralized data loading with lazy loading and caching."""
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import numpy as np
import pandas as pd


class DataLoader:
    """
    Centralized data loader for epidemic simulation.
    
    Handles lazy loading of population and mobility data with caching
    for efficient repeated access.
    """
    
    def __init__(self, dataset_name: str = "Madrid", base_path: Optional[Path] = None):
        self._dataset_name = dataset_name
        self._base_path = base_path or Path(".")
        self._mobility_dir = self._base_path / f"data_mobility_{dataset_name}"
        self._cache: Dict[str, Any] = {}
    
    @property
    def dataset_name(self) -> str:
        return self._dataset_name
    
    @staticmethod
    def _haversine_np(lon1, lat1, lon2, lat2):
        """Calculate the great-circle distance between points on the earth."""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    
    @cached_property
    def population_df(self) -> pd.DataFrame:
        """Load and process population data."""
        pop_file = self._mobility_dir / f"population_{self._dataset_name}.csv"
        df = pd.read_csv(pop_file)
        
        if self._dataset_name == "NE-England":
            df = df.rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'patch_id': 'id'
            })
            df['name'] = 'patch_' + df['id'].astype(str)
            if 'group' not in df.columns:
                df['group'] = '1'
        
        df.reset_index(drop=True, inplace=True)
        df['group'] = df['group'].astype(str)
        return df
    
    @cached_property
    def n_patches(self) -> int:
        """Number of districts/patches in the dataset."""
        return len(self.population_df)
    
    @cached_property
    def distance_matrix_km(self) -> np.ndarray:
        """Distance matrix between all patch centroids (km)."""
        coords = self.population_df.sort_values('id')[['lon', 'lat']].to_numpy()
        lons = coords[:, 0]
        lats = coords[:, 1]
        lons1, lats1 = lons[:, np.newaxis], lats[:, np.newaxis]
        lons2, lats2 = lons[np.newaxis, :], lats[np.newaxis, :]
        return self._haversine_np(lons1, lats1, lons2, lats2)
    
    @cached_property
    def mobility_matrix(self) -> np.ndarray:
        """Raw mobility/flow matrix M_ij."""
        flow_file = self._mobility_dir / f"mobility_matrix_{self._dataset_name}.csv"
        
        if self._dataset_name == "NE-England":
            return pd.read_csv(flow_file, index_col=0).to_numpy()
        else:
            df = pd.read_csv(flow_file)
            if df.shape[1] == self.n_patches + 1:
                df = df.iloc[:, 1:]
            return df.to_numpy()
    
    @cached_property
    def D_ij_matrix(self) -> np.ndarray:
        """Proportion matrix derived from mobility."""
        M = self.mobility_matrix
        sum_Mij = M.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_ij = np.nan_to_num(M / sum_Mij)
        np.fill_diagonal(D_ij, 1.0 - (D_ij.sum(axis=1) - np.diag(D_ij)))
        return D_ij
    
    @cached_property
    def N_ij_initial(self) -> np.ndarray:
        """Population flow matrix N_ij."""
        return (self.D_ij_matrix * self.population_df['population'].to_numpy()[:, np.newaxis]).round().astype(int)
    
    @cached_property
    def flow_ij_df(self) -> pd.DataFrame:
        """Flow matrix as DataFrame with patch indices."""
        n_patches = self.n_patches
        patch_indices = np.arange(n_patches)
        patch_i, patch_j = np.meshgrid(patch_indices, patch_indices, indexing='ij')
        
        return pd.DataFrame({
            'patch_i': patch_i.ravel(),
            'patch_j': patch_j.ravel(),
            'N_ij': self.N_ij_initial.ravel()
        })
    
    def load_contact_data(self, contact_file: str) -> pd.DataFrame:
        """Load contact network data for an event."""
        path = self._base_path / contact_file
        return pd.read_csv(path)
    
    def load_contact_data_cached(self, contact_file: str) -> pd.DataFrame:
        """Load contact data with caching."""
        if contact_file not in self._cache:
            self._cache[contact_file] = self.load_contact_data(contact_file)
        return self._cache[contact_file]
    
    def prepare_all(self) -> Dict[str, Any]:
        """Load all required data and return as dictionary."""
        return {
            "population_df": self.population_df,
            "n_patches": self.n_patches,
            "distance_matrix_km": self.distance_matrix_km,
            "flow_ij_df": self.flow_ij_df,
            "N_ij_initial": self.N_ij_initial,
        }


def preprocess_contact_file(cfg: Any) -> Tuple[str, int, Dict[str, str]]:
    """
    Preprocesses the contact file to convert string IDs to integers.
    Returns the path to the new file, the actual number of attendees, and the header map.
    """
    original_file = cfg.large_venue_contact_file
    print(f"Preprocessing contact file '{original_file}' to convert string IDs to integers...")
    
    df = pd.read_csv(original_file)
    
    # Determine column names
    header_map = cfg.event_contact_header_map or {}
    
    def get_col(df, keys, default_candidates):
        if isinstance(keys, str): keys = [keys]
        for k in keys:
            cfg_name = header_map.get(k)
            if cfg_name and cfg_name in df.columns:
                return cfg_name
        for c in default_candidates:
            if c in df.columns:
                print(f"  -> Warning: Configured column for '{keys}' not found. Using '{c}' instead.")
                return c
        return None

    col_i = get_col(df, "agent_i", ["agent_i", "ped1_id", "id_1", "u", "p1"])
    col_j = get_col(df, "agent_j", ["agent_j", "ped2_id", "id_2", "v", "p2"])
    col_dur = get_col(df, ["duration_frames", "tc_s"], ["total_duration_frames", "tc_s", "duration", "frames"])
    
    if not col_i: raise KeyError(f"Column for agent_i not found in {original_file}. Available: {list(df.columns)}")
    if not col_j: raise KeyError(f"Column for agent_j not found in {original_file}. Available: {list(df.columns)}")
    if not col_dur: raise KeyError(f"Column for duration not found in {original_file}. Available: {list(df.columns)}")
    
    # Update config to ensure downstream simulation uses the correct columns
    if cfg.event_contact_header_map is None: cfg.event_contact_header_map = {}
    cfg.event_contact_header_map["agent_i"] = col_i
    cfg.event_contact_header_map["agent_j"] = col_j
    cfg.event_contact_header_map["duration_frames"] = col_dur

    # Create mapping for string IDs to integers
    s_i = df[col_i].astype(str).str.strip()
    s_j = df[col_j].astype(str).str.strip()
    
    all_agents = pd.concat([s_i, s_j]).unique()
    all_agents = all_agents[all_agents != 'nan']
    all_agents.sort()
    
    n_unique_agents = len(all_agents)
    id_map = {agent_id: idx for idx, agent_id in enumerate(all_agents)}
    
    mapping_filename = f"{Path(original_file).stem}_id_mapping.csv"
    pd.DataFrame(list(id_map.items()), columns=['original_id', 'new_id']).to_csv(mapping_filename, index=False)
    print(f"  -> Saved ID mapping to '{mapping_filename}'")

    df[col_i] = s_i.map(id_map)
    df[col_j] = s_j.map(id_map)
    df.dropna(subset=[col_i, col_j], inplace=True)
    df[col_i] = df[col_i].astype(int)
    df[col_j] = df[col_j].astype(int)
    
    temp_filename = f"{Path(original_file).stem}_int_mapped.csv"
    df.to_csv(temp_filename, index=False)
    print(f"  -> Saved preprocessed file to '{temp_filename}'")

    header_map = {"agent_i": col_i, "agent_j": col_j}
    return temp_filename, n_unique_agents, header_map


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


def load_and_prepare_data(cfg_or_dataset: Any) -> Dict[str, Any]:
    """
    Backward-compatible function to load data.
    
    Accepts either a config module-like object or dataset name string.
    """
    if isinstance(cfg_or_dataset, str):
        dataset_name = cfg_or_dataset
        population_factor = 1.0
    else:
        dataset_name = getattr(cfg_or_dataset, 'dataset_name', 'Madrid')
        population_factor = getattr(cfg_or_dataset, 'population_factor', 1.0)
    
    loader = DataLoader(dataset_name)
    
    data = loader.prepare_all()
    
    if population_factor != 1.0:
        data["population_df"]['population'] = (
            data["population_df"]['population'] * population_factor
        ).round().astype(int)
        data["N_ij_initial"] = (
            data["N_ij_initial"] * population_factor
        ).round().astype(int)
    
    return data