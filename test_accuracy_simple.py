#!/usr/bin/env python3
"""
Simple test to verify accuracy calculation works.
"""

import torch
import sys
import os

# Add the models directory to the path
sys.path.append('/root/mcc-fw/models')

from utils import compute_metrics

def test_accuracy_calculation():
    """Test accuracy calculation with different tensor shapes."""
    
    print("🧪 Testing accuracy calculation...")
    
    # Get device from utils
    from utils import device
    print(f"Using device: {device}")
    
    # Test case 1: Binary classification with 1D tensors
    print("\n1. Testing binary classification (1D tensors):")
    y_pred_1d = torch.tensor([0.8, 0.3, 0.9, 0.1]).to(device)  # Predictions
    y_true_1d = torch.tensor([1, 0, 1, 0]).to(device)  # True labels
    
    res_1d = {
        "labels": y_true_1d,
        "predictions": y_pred_1d,
        "loss": 0.5
    }
    
    try:
        metrics_1d = compute_metrics(res_1d, num_classes=2, multi_label=False)
        print(f"   ✅ Success! Accuracy: {metrics_1d['result'][metrics_1d['metric'].index('accuracy')]:.2f}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test case 2: Binary classification with 2D tensors
    print("\n2. Testing binary classification (2D tensors):")
    y_pred_2d = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]]).to(device)  # Predictions
    y_true_2d = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]).to(device)  # True labels (one-hot)
    
    res_2d = {
        "labels": y_true_2d,
        "predictions": y_pred_2d,
        "loss": 0.5
    }
    
    try:
        metrics_2d = compute_metrics(res_2d, num_classes=2, multi_label=False)
        print(f"   ✅ Success! Accuracy: {metrics_2d['result'][metrics_2d['metric'].index('accuracy')]:.2f}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test case 3: Multilabel classification
    print("\n3. Testing multilabel classification:")
    y_pred_ml = torch.tensor([[0.8, 0.3], [0.2, 0.9], [0.7, 0.1]]).to(device)  # Predictions
    y_true_ml = torch.tensor([[1, 0], [0, 1], [1, 0]]).to(device)  # True labels
    
    res_ml = {
        "labels": y_true_ml,
        "predictions": y_pred_ml,
        "loss": 0.5
    }
    
    try:
        metrics_ml = compute_metrics(res_ml, num_classes=2, multi_label=True)
        print(f"   ✅ Success! Accuracy: {metrics_ml['result'][metrics_ml['metric'].index('accuracy')]:.2f}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting accuracy calculation test...")
    
    success = test_accuracy_calculation()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
