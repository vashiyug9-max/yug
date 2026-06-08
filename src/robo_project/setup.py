from setuptools import find_packages, setup

setup(
    name='robo_project',
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/robo_project']),
        ('share/robo_project', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'runner_node = robo_project.runner_node2:main',
            'action = robo_project.action_executive:main',
            'planner = robo_project.pipeline:main',
            'bridge = robo_project.habitat_bridge_vinebot_2:main',
        ],
    },
)
