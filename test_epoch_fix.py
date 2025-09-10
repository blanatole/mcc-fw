#!/usr/bin/env python3
"""
Test script to verify that the epoch fix works correctly.
This script runs a quick training session to check if all epochs are saved.
"""

import subprocess
import sys
import os

def test_epoch_fix():
    """Test the epoch fix by running a short training session."""
    
    print("🧪 Testing epoch fix...")
    print("=" * 50)
    
    # Change to the project directory
    os.chdir('/root/mcc-fw')
    
    # Run a quick test with 3 epochs
    cmd = [
        'python3', 'models/run_mm_late.py',
        '--txt_model_name', 'bertweet',
        '--img_model_name', 'vit', 
        '--fusion_name', 'concat',
        '--use_clip_loss', '--beta_itc', '0.1',
        '--task', '8',
        '--epochs', '3',
        '--testing'  # Use testing mode for faster execution
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print()
            
            # Check the results
            check_results()
            
        else:
            print("❌ Training failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (this is expected for testing)")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    
    return True

def check_results():
    """Check if the results files have the correct number of epochs."""
    
    print("🔍 Checking results...")
    
    # Check for metrics files
    results_dir = "results/mm_late/testing/"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Find metrics files
    import glob
    val_files = glob.glob(f"{results_dir}*_metrics_val.csv")
    test_files = glob.glob(f"{results_dir}*_metrics_test.csv")
    
    print(f"Found {len(val_files)} validation files and {len(test_files)} test files")
    
    for file_path in val_files + test_files:
        print(f"\n📊 Checking {os.path.basename(file_path)}:")
        
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Count epochs (columns starting with 'epoch-')
            epoch_cols = [col for col in df.columns if col.startswith('epoch-')]
            num_epochs = len(epoch_cols)
            
            print(f"   Epochs found: {num_epochs}")
            print(f"   Epoch columns: {epoch_cols}")
            
            if num_epochs >= 3:  # We ran 3 epochs
                print("   ✅ Correct number of epochs!")
            else:
                print("   ❌ Missing epochs!")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")

if __name__ == "__main__":
    print("🚀 Starting epoch fix test...")
    print()
    
    success = test_epoch_fix()
    
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
