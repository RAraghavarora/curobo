#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Script to visualize end-effector trajectories in 3D space
# Reads trajectory NPZ file and plots end-effector poses using forward kinematics

# Standard Library
import argparse

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize end-effector trajectories from NPZ file"
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Path to NPZ trajectory file",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot config file (if not in metadata)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ee_trajectories_3d.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively instead of saving",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to plot (None = all)",
    )
    return parser.parse_args()


def compute_ee_trajectory(joint_positions, robot_config_file, tensor_args):
    """Compute end-effector positions from joint positions using forward kinematics.
    
    Args:
        joint_positions: Joint positions array (T, DOF)
        robot_config_file: Robot configuration file name
        tensor_args: Tensor device configuration
        
    Returns:
        End-effector positions array (T, 3)
    """
    # Load robot config
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    
    # Use MotionGen to get kinematics
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_config_file,
        None,  # No world
        tensor_args,
    )
    motion_gen = MotionGen(motion_gen_config)
    
    # Convert joint positions to tensor
    joint_positions_tensor = tensor_args.to_device(joint_positions)
    
    # Create JointState
    joint_state = JointState.from_position(
        joint_positions_tensor,
        joint_names=motion_gen.kinematics.joint_names,
    )
    
    # Compute forward kinematics
    kin_state = motion_gen.compute_kinematics(joint_state)
    
    # Extract end-effector positions
    ee_positions = kin_state.ee_pos_seq.cpu().numpy()
    
    return ee_positions


def plot_ee_trajectories(trajectories_ee_pos, output_file=None, show_plot=False):
    """Plot end-effector trajectories in 3D space.
    
    Args:
        trajectories_ee_pos: List of end-effector position arrays
        output_file: Output filename (if not showing)
        show_plot: Whether to show plot interactively
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories_ee_pos)))
    
    # Plot each trajectory
    for i, ee_pos in enumerate(trajectories_ee_pos):
        # Plot trajectory line
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], 
                color=colors[i], alpha=0.6, linewidth=1.5)
        
        # Mark start point with larger marker
        ax.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2],
                  color=colors[i], marker='o', s=80, alpha=0.9, 
                  edgecolors='black', linewidths=1)
        
        # Mark end point with different marker
        ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2],
                  color=colors[i], marker='s', s=80, alpha=0.9,
                  edgecolors='black', linewidths=1)
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=10)
    ax.set_title(f'End-Effector Trajectories ({len(trajectories_ee_pos)} trajectories)\n'
                 f'○ = Start, □ = End', fontsize=14, pad=20)
    
    # Set equal aspect ratio for better visualization
    all_points = np.vstack(trajectories_ee_pos)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', label='Start',
               markerfacecolor='gray', markersize=10, linestyle='None',
               markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='s', color='gray', label='End',
               markerfacecolor='gray', markersize=10, linestyle='None',
               markeredgecolor='black', markeredgewidth=1),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_file}")
        plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("End-Effector Trajectory Visualization")
    print("=" * 80)
    print(f"Loading trajectories from: {args.npz_file}")
    
    # Load NPZ file
    data = np.load(args.npz_file, allow_pickle=True)
    metadata = data['metadata'].item()
    
    # Get robot config
    robot_config = args.robot if args.robot else metadata.get('robot_config')
    if not robot_config:
        raise ValueError("Robot config not found in metadata and not provided via --robot")
    
    print(f"Robot: {robot_config}")
    print(f"Total trajectories in file: {metadata['num_trajectories']}")
    
    # Determine how many trajectories to plot
    num_traj = metadata['num_trajectories']
    if args.max_trajectories:
        num_traj = min(num_traj, args.max_trajectories)
    print(f"Plotting: {num_traj} trajectories")
    
    # Initialize tensor args
    tensor_args = TensorDeviceType()
    
    # Compute end-effector trajectories
    print("\nComputing forward kinematics for all trajectories...")
    trajectories_ee_pos = []
    
    for i in range(num_traj):
        joint_positions = data[f'trajectory_{i}_position']
        
        # Compute end-effector positions
        ee_pos = compute_ee_trajectory(joint_positions, robot_config, tensor_args)
        trajectories_ee_pos.append(ee_pos)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_traj:
            print(f"  Processed {i + 1}/{num_traj} trajectories")
    
    print(f"\n✓ Forward kinematics computed for {num_traj} trajectories")
    
    # Plot statistics
    print("\nTrajectory Statistics:")
    all_points = np.vstack(trajectories_ee_pos)
    print(f"  X range: [{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}] m")
    print(f"  Y range: [{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}] m")
    print(f"  Z range: [{all_points[:, 2].min():.3f}, {all_points[:, 2].max():.3f}] m")
    
    # Calculate workspace volume
    x_range = all_points[:, 0].max() - all_points[:, 0].min()
    y_range = all_points[:, 1].max() - all_points[:, 1].min()
    z_range = all_points[:, 2].max() - all_points[:, 2].min()
    volume = x_range * y_range * z_range
    print(f"  Approximate workspace volume: {volume:.4f} m³")
    
    # Plot trajectories
    print("\nGenerating 3D plot...")
    plot_ee_trajectories(trajectories_ee_pos, args.output, args.show)
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

