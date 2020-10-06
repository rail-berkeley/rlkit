#!/bin/bash
NVIDIA_DRIVER_VERSION=418.43

echo "detected nvidia driver: "${NVIDIA_DRIVER_VERSION}

sed -i "s/\${nvidia_driver}/${NVIDIA_DRIVER_VERSION}/" /usr/share/vulkan/icd.d/nvidia_icd.json
cat /usr/share/vulkan/icd.d/nvidia_icd.json

wget -q http://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run
sh NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run --extract-only
cp -r NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}/* /usr/lib/x86_64-linux-gnu/