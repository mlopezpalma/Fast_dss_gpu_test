#!/bin/bash

echo "======================================="
echo "Fast DDS Installation Verification"
echo "======================================="
echo ""

# Check libraries
echo "Fast DDS Libraries:"
echo "-------------------"
find /usr/local/lib -name "*fastdds*" -o -name "*fastrtps*" -o -name "*fastcdr*" 2>/dev/null | head -10

echo ""
echo "Header Files:"
echo "-------------"
ls -la /usr/local/include/ | grep -E "fast|dds"

echo ""
echo "Binary Tools:"
echo "-------------"
ls -la /usr/local/bin/ | grep -E "fast|dds" || echo "No Fast DDS binaries found in /usr/local/bin"

echo ""
echo "Example Programs:"
echo "-----------------"
find /usr/local -name "*DDSHelloWorld*" 2>/dev/null | head -5

echo ""
echo "Library Path:"
echo "-------------"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ""
echo "Fast DDS is ready for use!"
echo "======================================="
