import os
from glob import glob
from setuptools import setup

package_name = 'robo_project'


def install_tree(base_dir):
    paths = []
    for root, _, files in os.walk(base_dir):
        if files:
            paths.append((
                os.path.join('share', package_name, root),
                [os.path.join(root, f) for f in files],
            ))
    return paths


setup(
    name=package_name,
    version='0.0.0',
    packages=[
        'robo_project',
        'robo_project.scripts',
        'robo_project.scripts.cmn',
        'robo_project.scripts.cmn.model',
        'robo_project.scripts.rotated_rectangle_crop_opencv',
    ],
    package_dir={
        'robo_project': '.',
        'robo_project.scripts': 'scripts',
        'robo_project.scripts.cmn': 'scripts/cmn',
        'robo_project.scripts.cmn.model': 'scripts/cmn/model',
        'robo_project.scripts.rotated_rectangle_crop_opencv': 'scripts/rotated_rectangle_crop_opencv',
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + install_tree('config') + install_tree('launch') + install_tree('scripts/cmn/model'),
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'runner_node = robo_project.runner_node:main',
        ],
    },
)
