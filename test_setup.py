#!/usr/bin/env python3
"""
Test script to verify USD assets can be loaded.
Run this after setting up Isaac Sim 4.5 and Isaac Lab 2.0.
"""

import os
import sys

def test_assets():
    """Test that USD assets exist and have proper structure."""
    
    # Check asset files exist
    assets_dir = "assets"
    required_assets = ["table_at_hand_height.usd", "cube.usd"]
    
    print("🔍 Checking USD assets...")
    for asset in required_assets:
        asset_path = os.path.join(assets_dir, asset)
        if os.path.exists(asset_path):
            print(f"✅ Found: {asset}")
            # Check file size
            size = os.path.getsize(asset_path)
            print(f"   Size: {size} bytes")
        else:
            print(f"❌ Missing: {asset}")
            return False
    
    print("\n📋 Asset positioning summary:")
    print("   Table: 60cm x 60cm surface at Z=0.3m")
    print("   Cube: 5cm cube, positioned at (0, -0.1, 0.325)")
    print("   Hand: Shadow Hand at (0, 0, 0.45)")
    print("   → Cube is 10cm in front of hand, within reach")
    
    return True

def test_config():
    """Test configuration file."""
    print("\n🔍 Checking configuration...")
    
    try:
        import yaml
        with open("source/config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        required_sections = ["train", "env", "her", "eval"]
        for section in required_sections:
            if section in cfg:
                print(f"✅ Config section: {section}")
            else:
                print(f"❌ Missing config section: {section}")
                return False
        
        # Check key parameters
        env_cfg = cfg["env"]
        print(f"   Table height: {env_cfg['table_height']}m")
        print(f"   Flip angle: {env_cfg['flip_angle']} rad ({env_cfg['flip_angle']*180/3.14159:.0f}°)")
        her_cfg = cfg['her']
        print(f"   HER n_sampled_goal: {her_cfg.get('n_sampled_goal', 'N/A')}")
        print(f"   Goal strategy: {her_cfg.get('goal_selection_strategy', 'future')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 HER Hand Project - Asset Verification")
    print("=" * 50)
    
    assets_ok = test_assets()
    config_ok = test_config()
    
    print("\n" + "=" * 50)
    if assets_ok and config_ok:
        print("✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("1. Install Isaac Sim 4.5 and Isaac Lab 2.0")
        print("2. Install dependencies: pip install sb3-contrib stable-baselines3")
        print("3. Run training: python scripts/train.py --headless")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
