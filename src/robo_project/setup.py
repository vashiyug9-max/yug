import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robo_project'


def generate_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ]

    # Include config files and the CMN model folder recursively in the installed share folder.
    dirs_to_include = ['config', 'cmn/model']

    for directory in dirs_to_include:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            source_files = [
                os.path.join(root, f)
                for f in files
                if not f.endswith(('.zip', '.pyc', '.pyo'))
            ]
            if source_files:
                dest_dir = os.path.join('share', package_name, root)
                data_files.append((dest_dir, source_files))

    return data_files


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name] + [f"{package_name}.{pkg}" for pkg in find_packages(where='.', exclude=['test'])],
    package_dir={package_name: '.'},
    data_files=generate_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='VineBot project team',
    maintainer_email='robot_@todo.todo',
    description='Integrated VineBot simulation, ML, CMN localization, and planning package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization_navigation_bridge = robo_project.ros.localization_navigation_bridge:main',
            'localization_motion_gate = robo_project.ros.localization_motion_gate:main',

            'live_global_pose_node = robo_project.localization.live_global_pose_node:main',
            'localization_error_monitor_node = robo_project.localization.localization_error_monitor_node:main',
            'sketch_pf_node = robo_project.localization.sketch_pf_node:main',

            # One folder per team task: sim/, ml/, localization/, cmn/, planning/;
            # ros/ holds the glue nodes, common/ the shared helpers.

            # sim/
            'bridge = robo_project.sim.habitat_bridge:main',

            # planning/
            'motion_planner = robo_project.planning.motion_planner:main',

            # ros/: path follower + map, map server
            'runner_node = robo_project.ros.runner_node:main',
            'map_server_node = robo_project.ros.map_server_node:main',

            # ml/: fusion obstacle model, ML pose estimation
            'fusion_obstacle_node = robo_project.ml.fusion.fusion_obstacle_node:main',
            'ml_pose_node = robo_project.ml.pose_estimation.ml_pose_node:main',

            # localization/: particle filter validation (Jay Week-10)
        ],
    },
)
