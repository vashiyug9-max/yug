import os

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument,RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():


    package_name='vinebot_description'
    world_package_name='vineyard_world'


    default_world = os.path.join(
        get_package_share_directory(world_package_name),
        'world',
        'real_vineyard_for_result.sdf'
    )

    world = LaunchConfiguration('world')

    world_args= DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='sdf to load'
        )
    

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("vinebot_description"),
            "config",
            "my_controller.yaml",
        ]
    )
    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','robot_launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                    launch_arguments={'gz_args': ['-r -v4 --render-engine=ogre2 ', world], 'on_exit_shutdown': 'true'}.items()
             )

    
    spawn_entity = Node(package='ros_gz_sim', executable='create',
                        arguments=['-topic', 'robot_description',
                                   '-name', 'vinebot',
                                   '-y','5.0',
                                   '-z','10.0'],
                        output='screen')
    
    
    
    diff_drive_spawner = Node(
    package="controller_manager",
    executable="spawner",
    arguments=[
        "diff_cont",
        "--param-file",           
        robot_controllers,
        "--controller-manager-timeout", "20",   
        "--switch-timeout", "30"                
        ],
    output="screen"
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_broad",
            "--controller-manager-timeout", "20",
            "--switch-timeout", "30"
            # "--inactive",
        ],
    output="screen"
    )

    
    
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            'config_file:=/home/vamsi/code/thesis/vineyard_ws/src/vinebot_description/config/gz_bridge.yaml',
        ]
    )

    return LaunchDescription([
        
        rsp,
        world_args,
        gazebo,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[diff_drive_spawner]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=diff_drive_spawner,
                on_exit=[joint_broad_spawner]
            )
        ),

        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=joint_broad_spawner,
        #         on_exit=[ros_gz_bridge]
        #     )
        # ),
        # ros_gz_bridge,
            
    ])