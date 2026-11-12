from glob import glob
from setuptools import setup

package_name = "vinebot_bringup"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vashiyug9-max",
    description="Real VineBot bringup (Roboteq, PS4 joystick, twist_mux)",
    license="Apache-2.0",
    entry_points={"console_scripts": ["joy_gate = vinebot_bringup.joy_gate:main",
                                     "motion_confirm = vinebot_bringup.motion_confirm:main",
                                     "wheel_imu_odom = vinebot_bringup.wheel_imu_odom:main"]},
)
