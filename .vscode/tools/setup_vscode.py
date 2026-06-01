#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Setup VSCode settings to add IsaacLab paths to Python analysis."""

import argparse
import json
import os


def setup_vscode_settings(isaaclab_path: str = None):
    """Setup or update the VSCode settings.json file with IsaacLab paths.
    
    Args:
        isaaclab_path: Optional path to IsaacLab directory. If not provided, assumes
                      IsaacLab is located at ../IsaacLab relative to workspace.
    """
    # Get the workspace folder (repository root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vscode_dir = os.path.join(repo_root, ".vscode")
    settings_path = os.path.join(vscode_dir, "settings.json")
    
    # Create .vscode directory if it doesn't exist
    os.makedirs(vscode_dir, exist_ok=True)
    
    # Define the extra paths to add
    extra_paths = [
        "${workspaceFolder}/robolab",
        "${workspaceFolder}/IsaacLab/source/isaaclab_rl",
        "${workspaceFolder}/IsaacLab/source/isaaclab_tasks",
        "${workspaceFolder}/IsaacLab/source/isaaclab_assets",
        "${workspaceFolder}/IsaacLab/source/isaaclab_mimic",
        "${workspaceFolder}/IsaacLab/source/isaaclab_contrib",
        "${workspaceFolder}/IsaacLab/source/isaaclab",
        "${workspaceFolder}/rsl_rl"
    ]
    
    # Load existing settings or create new ones
    settings = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing {settings_path}, creating new settings")
            settings = {}
    
    # Update python.analysis.extraPaths
    if "python.analysis.extraPaths" not in settings:
        settings["python.analysis.extraPaths"] = []
    
    # Add paths that don't already exist
    existing_paths = settings["python.analysis.extraPaths"]
    for path in extra_paths:
        if path not in existing_paths:
            existing_paths.append(path)
    
    settings["python.analysis.extraPaths"] = existing_paths
    
    # Write updated settings
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    
    print(f"VSCode settings updated successfully at: {settings_path}")
    print(f"Added {len(extra_paths)} paths to python.analysis.extraPaths")


def main():
    """Main function to run the setup."""
    parser = argparse.ArgumentParser(description="Setup VSCode settings for RoboLab with IsaacLab paths")
    parser.add_argument(
        "--isaaclab-path",
        type=str,
        default=None,
        help="Optional path to IsaacLab directory (default: ../IsaacLab relative to workspace)"
    )
    
    args = parser.parse_args()
    setup_vscode_settings(args.isaaclab_path)


if __name__ == "__main__":
    main()
