"""
Installation Verification Script
Checks if all packages match Kaggle versions exactly
"""

import sys

print("\n" + "=" * 70)
print("🌾 AGRITECH AI - INSTALLATION VERIFICATION")
print("=" * 70)

# Expected Kaggle versions (from your notebooks)
KAGGLE_VERSIONS = {
    "Python": "3.11.13",
    "TensorFlow": "2.18.0",
    "NumPy": "1.26.4",
    "Pandas": "2.2.3",
    "scikit-learn": "1.2.2",
    "joblib": "1.5.1",
    "Pillow": "11.2.1"
}

# Check Python version first
python_version = sys.version.split()[0]
print(f"\n🐍 Python Version: {python_version}")

if python_version.startswith("3.11"):
    print("   ✅ Python 3.11.x detected - Perfect!")
elif python_version.startswith("3.10"):
    print("   ⚠️  Python 3.10.x - Should work, but 3.11 is recommended")
else:
    print(f"   ⚠️  Warning: Expected 3.11.x, got {python_version}")
    print("   💡 Consider using Python 3.11 for best compatibility")

print("\n" + "-" * 70)
print("📦 Checking Package Versions:")
print("-" * 70)

packages_to_check = [
    ("tensorflow", "TensorFlow"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("PIL", "Pillow"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
]

installed_versions = {}
missing_packages = []
version_mismatches = []

for import_name, display_name in packages_to_check:
    try:
        if import_name == "sklearn":
            import sklearn as pkg
        elif import_name == "PIL":
            from PIL import Image as pkg
        else:
            pkg = __import__(import_name)
        
        version = getattr(pkg, "__version__", "Unknown")
        installed_versions[display_name] = version
        
        # Check if it's a critical package that needs version match
        if display_name in KAGGLE_VERSIONS:
            expected = KAGGLE_VERSIONS[display_name]
            if version == expected:
                print(f"✅ {display_name:15s} : {version:12s} (matches Kaggle)")
            else:
                print(f"⚠️  {display_name:15s} : {version:12s} (Kaggle: {expected})")
                version_mismatches.append((display_name, version, expected))
        else:
            # FastAPI/Uvicorn/Pydantic - just show version
            print(f"✅ {display_name:15s} : {version}")
            
    except ImportError as e:
        print(f"❌ {display_name:15s} : NOT INSTALLED")
        missing_packages.append(display_name)

print("\n" + "=" * 70)

# Summary
if missing_packages:
    print("❌ MISSING PACKAGES DETECTED")
    print("-" * 70)
    for pkg in missing_packages:
        print(f"   • {pkg}")
    print("\n💡 Fix this by running:")
    print("   pip install -r requirements.txt")
    print("=" * 70)
    sys.exit(1)

if version_mismatches:
    print("⚠️  VERSION MISMATCHES DETECTED")
    print("-" * 70)
    print("\nThe following packages have different versions:")
    print()
    for pkg, local, expected in version_mismatches:
        print(f"   📦 {pkg}")
        print(f"      Local:    {local}")
        print(f"      Expected: {expected}")
        print()
    
    print("💡 This might cause model loading errors!")
    print("\n🔧 To fix, run these commands:")
    print("-" * 70)
    for pkg, local, expected in version_mismatches:
        pkg_install_name = pkg.lower().replace("-", "")
        if pkg == "scikit-learn":
            pkg_install_name = "scikit-learn"
        elif pkg == "Pillow":
            pkg_install_name = "Pillow"
        print(f"   pip uninstall {pkg_install_name} -y")
        print(f"   pip install {pkg_install_name}=={expected}")
    print("=" * 70)
    
    # Still allow to continue but warn
    print("\n⚠️  You can proceed, but models might not load correctly.")
    print("   It's recommended to fix version mismatches first.")
    print("=" * 70)
else:
    print("🎉 ALL VERSIONS MATCH PERFECTLY!")
    print("-" * 70)
    print("✅ Python version: OK")
    print("✅ TensorFlow version: OK")
    print("✅ NumPy version: OK")
    print("✅ All ML dependencies: OK")
    print("✅ FastAPI dependencies: OK")
    print("\n🚀 You're ready for the next steps:")
    print("   1. Ensure model files are in backend/models/")
    print("   2. Run: python test_models.py")
    print("   3. Run: python main.py")
    print("=" * 70)

# Show detailed summary table
print("\n📊 DETAILED VERSION SUMMARY:")
print("-" * 70)
print(f"{'Package':<20} {'Installed':<15} {'Required':<15} {'Status'}")
print("-" * 70)

for pkg, expected in KAGGLE_VERSIONS.items():
    local = installed_versions.get(pkg, "Not installed")
    if local == expected:
        status = "✅ Match"
    elif local == "Not installed":
        status = "❌ Missing"
    else:
        status = "⚠️  Mismatch"
    print(f"{pkg:<20} {local:<15} {expected:<15} {status}")

# Show FastAPI related packages
print(f"{'FastAPI':<20} {installed_versions.get('FastAPI', 'N/A'):<15} {'Any':<15} {'✅ OK' if 'FastAPI' in installed_versions else '❌ Missing'}")
print(f"{'Uvicorn':<20} {installed_versions.get('Uvicorn', 'N/A'):<15} {'Any':<15} {'✅ OK' if 'Uvicorn' in installed_versions else '❌ Missing'}")
print(f"{'Pydantic':<20} {installed_versions.get('Pydantic', 'N/A'):<15} {'Any':<15} {'✅ OK' if 'Pydantic' in installed_versions else '❌ Missing'}")

print("=" * 70)

# Final status
if not missing_packages and not version_mismatches:
    print("\n✅ INSTALLATION VERIFIED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    sys.exit(0)
elif missing_packages:
    print("\n❌ INSTALLATION INCOMPLETE - Missing packages")
    print("=" * 70 + "\n")
    sys.exit(1)
else:
    print("\n⚠️  INSTALLATION COMPLETE - But with version warnings")
    print("=" * 70 + "\n")
    sys.exit(0)