import os
import shutil
from pathlib import Path

source_folder = "path_to_BridgeDrive/BridgeDrive_adaptation_LEAD" 
destination_folder = "path_to_LEAD"

def copy_items_with_structure(item_paths, source_dir, dest_dir):
    """
    Copy a list of files and folders while maintaining relative path structure.
    
    Args:
        item_paths: List of relative paths (files and folders)
        source_dir: Base directory containing source items
        dest_dir: Base directory where items will be copied
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'files_copied': 0,
        'folders_copied': 0,
        'skipped': 0,
        'errors': 0
    }
    
    for rel_path in item_paths:
        source_item = source_dir / rel_path
        dest_item = dest_dir / rel_path
        
        # Check if source exists
        if not source_item.exists():
            print(f"⚠️  Warning: {source_item} does not exist")
            results['skipped'] += 1
            continue
        
        try:
            if source_item.is_file():
                # Copy file
                dest_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_item, dest_item)
                print(f"✅ Copied file: {rel_path}")
                results['files_copied'] += 1
                
            elif source_item.is_dir():
                # Copy folder
                if dest_item.exists():
                    # If destination exists, merge contents
                    print(f"📁 Merging folder: {rel_path}")
                    for root, dirs, files in os.walk(source_item):
                        rel_root = Path(root).relative_to(source_item)
                        dest_root = dest_item / rel_root
                        dest_root.mkdir(parents=True, exist_ok=True)
                        
                        for file in files:
                            src_file = Path(root) / file
                            dst_file = dest_root / file
                            shutil.copy2(src_file, dst_file)
                else:
                    # Copy entire directory
                    shutil.copytree(source_item, dest_item)
                    print(f"📁 Copied folder: {rel_path}")
                results['folders_copied'] += 1
                
        except Exception as e:
            print(f"❌ Error copying {rel_path}: {e}")
            results['errors'] += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Copy process completed!")
    print(f"✅ Copied: {results['files_copied']} files, {results['folders_copied']} folders")
    print(f"⚠️  Skipped: {results['skipped']} items")
    print(f"❌ Errors: {results['errors']} items")
    
    return results


items_to_transfer= [
    "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py",
    "anchor_utils",   
    "data/benchmark_routes/bench2drive_split/bench2drive_111.xml",
    "lead/inference/closed_loop_inference_bridgedrive.py", 
    "lead/inference/config_closed_loop.py", 
    "lead/inference/open_loop_inference_bridgedrive.py", 
    "lead/inference/sensor_agent_bridgedrive.py",
    "lead/tfv6/diffusion_modules", 
    "lead/tfv6/planning_decoder_bridgedrive.py", 
    "lead/tfv6/tfv6_bridgedrive.py", 
    "lead/training/config_training.py",
    "lead/training/logger.py", 
    "lead/training/train_bridgedrive.py", 
    "lead/training/training_utils_bridgedrive.py", 
    "lead/visualization/visualizer_bridgedrive.py",
    "scripts/posttrain_bridgedrive.sh",  
    "scripts/eval_bench2drive_bridgedrive.sh", 
    "utils_file_transfer.py"
]

copy_items_with_structure(items_to_transfer, source_folder, destination_folder)