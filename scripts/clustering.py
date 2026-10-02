# -*- coding: utf-8 -*-
"""
K-means based automated clustering and sorting of mixed SEM particle datasets.

Author: Karishma Begum
Year: 2026
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class DatasetSorter:
    def __init__(self):
        """
        Initializes the in-memory sorting coordinator.
        Tracks all metadata inside a clean data matrix.
        """
        self.summary_df = None

    def process_live_grouping(self, n_clusters=3, target_min=0.0, target_max=1000.0):
        """
        Dynamically filters micrographs by size window, clusters the remaining valid 
        images using K-Means, and maps cluster IDs sequentially from smallest to largest.
        """
        if self.summary_df is None or self.summary_df.empty:
            return pd.DataFrame()

        working_df = self.summary_df.copy()

        # Separate valid (QC-passed) files from invalid/blank ones
        qc_pass_mask = working_df["particle_count"] > 0

        # Isolate files within the requested size range
        range_mask = qc_pass_mask & (working_df["avg_size_nm"] >= target_min) & (working_df["avg_size_nm"] <= target_max)

        # Subset used for clustering
        cluster_data = working_df[range_mask]

        if not cluster_data.empty:
            X = cluster_data[["avg_size_nm"]].values

            # Cap cluster count to the number of available samples
            actual_clusters = min(n_clusters, len(cluster_data))

            # Run K-Means on the filtered data
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto')
            raw_labels = kmeans.fit_predict(X)

            # Reorder cluster IDs so Group 0 is always the smallest average size
            id_means = {i: X[raw_labels == i].mean() for i in range(actual_clusters)}
            sorted_ids = sorted(id_means, key=id_means.get)
            label_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

            # Map sequential IDs back into the working dataframe
            final_labels = [label_mapping[label] for label in raw_labels]
            working_df.loc[range_mask, "cluster_id"] = final_labels

            # Build display labels from per-cluster mean size
            cluster_means = working_df[range_mask].groupby("cluster_id")["avg_size_nm"].mean().to_dict()
            working_df.loc[range_mask, "group_category"] = working_df[range_mask].apply(
                lambda r: f"Batch Group {int(r['cluster_id'])} (~{int(cluster_means[r['cluster_id']])}nm)", axis=1
            )
        else:
            # No samples match the requested size window
            working_df["cluster_id"] = -1
            working_df["group_category"] = "No Group Assigned"

        # Assign pipeline status per group
        working_df.loc[range_mask, "status"] = "READY FOR SAM ANALYSIS"

        # QC-passed but outside the requested size window
        hold_mask = qc_pass_mask & ~range_mask
        working_df.loc[hold_mask, "cluster_id"] = -1
        working_df.loc[hold_mask, "group_category"] = "Hold (Outside Size Window)"
        working_df.loc[hold_mask, "status"] = "HOLD IN BACKGROUND"

        # Excluded due to failed QC (see validator.py)
        working_df.loc[~qc_pass_mask, "cluster_id"] = -1
        working_df.loc[~qc_pass_mask, "group_category"] = "Excluded (Low Quality/Sparse)"
        working_df.loc[~qc_pass_mask, "status"] = "EXCLUDED"

        return working_df.sort_values(by="avg_size_nm").reset_index(drop=True)