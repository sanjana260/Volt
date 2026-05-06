"""
Preprocessing script for Articulate3D dataset.

Converts Articulate3D annotations (parts.json, artic.json) to per-point labels.

Usage:
    python tools/preprocess_articulate3d.py \
        --articulate_root /path/to/articulate3d/data \
        --scannetpp_root /path/to/scannetpp/data \
        --output_root /path/to/output/labels

Author: Sanjana Mohan
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import open3d as o3d


# ===== ADDED: Load mesh and part annotations =====
def load_scene_annotations(scene_id, articulate_root, scannetpp_root):
    """Load mesh and annotations for a scene.

    Args:
        scene_id: scene identifier
        articulate_root: root directory of Articulate3D dataset
        scannetpp_root: root directory of ScanNet++ data

    Returns:
        mesh: open3d mesh
        parts_dict: part information from parts.json
        artic_dict: articulation information from artic.json
    """
    # Load mesh
    mesh_path = Path(scannetpp_root) / scene_id / "mesh_aligned_0.05.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # Load parts annotations
    parts_path = Path(articulate_root) / scene_id / "parts.json"
    if parts_path.exists():
        with open(parts_path) as f:
            parts_dict = json.load(f)
    else:
        parts_dict = {}

    # Load articulation annotations
    artic_path = Path(articulate_root) / scene_id / "artic.json"
    if artic_path.exists():
        with open(artic_path) as f:
            artic_dict = json.load(f)
    else:
        artic_dict = {}

    return mesh, parts_dict, artic_dict


# ===== ADDED: Extract per-vertex articulation labels =====
def compute_per_vertex_labels(mesh, parts_dict, artic_dict):
    """Compute per-vertex articulation labels.

    Strategy: For each vertex, vote over adjacent faces to assign labels.

    Args:
        mesh: open3d mesh object
        parts_dict: dict mapping part_id -> {'triIndices': [...]}
        artic_dict: dict mapping part_id -> articulation info

    Returns:
        movable_labels: (N,) array, values in {0=fixed, 1=rotation, 2=translation}
        interactable_labels: (N,) binary array
    """
    num_vertices = len(mesh.vertices)
    movable_labels = np.zeros(num_vertices, dtype=np.int64)
    interactable_labels = np.zeros(num_vertices, dtype=np.int64)

    # Create a map from vertex to adjacent faces
    triangles = np.asarray(mesh.triangles)
    vertex_to_faces = [[] for _ in range(num_vertices)]
    for face_idx, face in enumerate(triangles):
        for v_idx in face:
            vertex_to_faces[v_idx].append(face_idx)

    # Map triangle indices to part IDs
    triangle_to_part = {}
    for part_id, part_info in parts_dict.items():
        tri_indices = part_info.get("triIndices", [])
        for tri_idx in tri_indices:
            triangle_to_part[tri_idx] = part_id

    # Assign labels to vertices
    for vertex_idx in range(num_vertices):
        adjacent_faces = vertex_to_faces[vertex_idx]

        # Collect part IDs from adjacent faces
        part_ids = []
        for face_idx in adjacent_faces:
            if face_idx in triangle_to_part:
                part_ids.append(triangle_to_part[face_idx])

        if not part_ids:
            # No adjacent faces with part labels
            continue

        # Most common part ID
        part_id_counter = {}
        for p_id in part_ids:
            part_id_counter[p_id] = part_id_counter.get(p_id, 0) + 1
        most_common_part = max(part_id_counter, key=part_id_counter.get)

        # Check if part is movable (appears in artic.json)
        if most_common_part in artic_dict:
            artic_info = artic_dict[most_common_part]
            # Determine motion type
            if "motion_type" in artic_info:
                motion_type = artic_info["motion_type"]
                if "rotation" in motion_type.lower():
                    movable_labels[vertex_idx] = 1
                elif "translation" in motion_type.lower():
                    movable_labels[vertex_idx] = 2
            else:
                # Default to rotation if motion type not specified
                movable_labels[vertex_idx] = 1

        # Check if part is interactable (has handle/knob tag)
        # This depends on your hierarchy encoding scheme
        # Example: parts with certain IDs are handles
        part_name = most_common_part
        if "handle" in part_name.lower() or "knob" in part_name.lower():
            interactable_labels[vertex_idx] = 1

    return movable_labels, interactable_labels


# ===== ADDED: Main preprocessing function =====
def preprocess_scene(scene_id, articulate_root, scannetpp_root, output_root):
    """Preprocess a single scene and save labels.

    Args:
        scene_id: scene identifier
        articulate_root: Articulate3D root
        scannetpp_root: ScanNet++ root
        output_root: output directory for labels

    Returns:
        success: bool
    """
    try:
        # Load data
        mesh, parts_dict, artic_dict = load_scene_annotations(
            scene_id, articulate_root, scannetpp_root
        )

        # Compute labels
        movable_labels, interactable_labels = compute_per_vertex_labels(
            mesh, parts_dict, artic_dict
        )

        # Save labels
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        movable_path = output_root / f"{scene_id}_movable_label.npy"
        interactable_path = output_root / f"{scene_id}_interactable_label.npy"

        np.save(movable_path, movable_labels)
        np.save(interactable_path, interactable_labels)

        return True
    except Exception as e:
        print(f"Error processing {scene_id}: {e}")
        return False


# ===== ADDED: Main entry point =====
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Articulate3D annotations to per-point labels"
    )
    parser.add_argument(
        "--articulate_root",
        type=str,
        required=True,
        help="Root directory of Articulate3D dataset",
    )
    parser.add_argument(
        "--scannetpp_root",
        type=str,
        required=True,
        help="Root directory of ScanNet++ data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/articulate3d_labels",
        help="Output directory for preprocessed labels",
    )

    args = parser.parse_args()

    # Get list of scene IDs (assumes they are top-level directories)
    articulate_root = Path(args.articulate_root)
    scene_ids = [d.name for d in articulate_root.iterdir() if d.is_dir()]

    print(f"Processing {len(scene_ids)} scenes...")

    # Process each scene
    successful = 0
    for scene_id in tqdm(scene_ids):
        success = preprocess_scene(
            scene_id,
            args.articulate_root,
            args.scannetpp_root,
            args.output_root,
        )
        if success:
            successful += 1

    print(f"\nSuccessfully processed {successful}/{len(scene_ids)} scenes")
    print(f"Labels saved to {args.output_root}")


if __name__ == "__main__":
    main()
