from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        Node(
            package="robo_project",
            executable="ml_pose_node",
            name="ml_pose_node",
            output="screen",

            parameters=[
                {
                    "image_folder":
                    "/home/het/robo_project_ws/src/robo_project/habitat_dataset/front"
                }
            ]
        )

    ])
