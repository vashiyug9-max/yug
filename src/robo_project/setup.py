from setuptools import find_packages, setup
from glob import glob

package_name = 'robo_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/config/maps',
            glob('config/maps/*')),
        ('share/' + package_name + '/launch',
        ['launch/ml_pose.launch.py'],
	),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot_',
    maintainer_email='robot_@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'runner_node = robo_project.runner_node2:main',
            'ml_pose_node = robo_project.ml_pose_node:main',
        ],
    },
)
