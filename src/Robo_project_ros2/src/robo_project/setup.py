from setuptools import setup, find_packages

package_name = 'robo_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yug',
    maintainer_email='yug@todo.todo',
    description='ROS2 UDP bridge for Habitat',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'runner_node = robo_project.runner_node:main',
        ],
    },
)
