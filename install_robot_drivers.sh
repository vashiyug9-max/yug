#!/bin/bash
# Installs the VineBot sensor drivers on the robot PC (Ubuntu 22.04 + ROS 2 Humble).
# Run it in a normal Ubuntu terminal (it asks for your password once, for sudo):
#   bash ~/robo_project_ws/robo_project_ws/install_robot_drivers.sh
#
#   1. Ubuntu packages: RealSense D455 driver, BNO055 IMU driver, IMU filter, USB cameras
#   2. Livox Mid-360: Livox-SDK2 (built from source, installed to /usr/local) and livox_ros_driver2
#      (built in ~/drivers_ws, separate from our workspace)
# Safe to run again (skips what is already there).
set -e
source /opt/ros/humble/setup.bash

echo "=== 1/3 Ubuntu packages"
sudo apt update
sudo apt install -y \
  ros-humble-realsense2-camera ros-humble-realsense2-description \
  ros-humble-bno055 ros-humble-imu-tools \
  ros-humble-usb-cam ros-humble-v4l2-camera \
  ros-humble-pcl-ros libpcl-dev libapr1-dev git cmake build-essential

echo "=== 2/3 Livox-SDK2 -> /usr/local"
mkdir -p ~/drivers_ws/src
cd ~/drivers_ws
[ -d Livox-SDK2 ] || git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir -p build && cd build
cmake .. && make -j"$(nproc)"
sudo make install
sudo ldconfig

echo "=== 3/3 livox_ros_driver2 -> ~/drivers_ws"
cd ~/drivers_ws/src
[ -d livox_ros_driver2 ] || git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
cp -f package_ROS2.xml package.xml            # what its build.sh does for ROS 2
cp -rf launch_ROS2/ launch/
# build with the SYSTEM python: with conda active the ROS message generator fails
#   ("module 'em' has no attribute 'BUFFERED_OPT'")
if command -v conda >/dev/null; then eval "$(conda shell.bash hook)"; conda deactivate; conda deactivate; fi
cd ~/drivers_ws
rm -rf build/livox_ros_driver2 build/livox_sdk2 install/livox_ros_driver2 install/livox_sdk2
colcon build --packages-select livox_ros_driver2 \
  --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=humble -DPython3_EXECUTABLE=/usr/bin/python3

echo
echo "=== done"
echo "Use the drivers (add to ~/.bashrc so every terminal has them, once):"
echo "  echo 'source ~/drivers_ws/install/setup.bash' >> ~/.bashrc"
echo "Livox network: the default config expects the PC at 192.168.1.5 and the Mid-360 at 192.168.1.12."
echo "  Set the PC's wired connection to IPv4 manual 192.168.1.5 / 255.255.255.0. The Mid-360's IP is"
echo "  192.168.1.1XX (XX = last two digits of its serial number); if it is not .12, change \"ip\" in"
echo "  ~/drivers_ws/src/livox_ros_driver2/config/MID360_config.json and rebuild (system python!):"
echo "    conda deactivate; cd ~/drivers_ws && colcon build --packages-select livox_ros_driver2 \\"
echo "      --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=humble -DPython3_EXECUTABLE=/usr/bin/python3"
echo "  Keep extrinsic_parameter at 0 (our rig rotates the points)."
