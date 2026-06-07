import os

import rclpy

from rclpy.node import Node

from geometry_msgs.msg import Twist

from robo_project.infer import predict_pose


class MLPoseNode(Node):

    def __init__(self):

        super().__init__("ml_pose_node")

        self.publisher_ = self.create_publisher(
            Twist,
            "/diff_cont/cmd_vel_unstamped",
            10
        )

        self.declare_parameter(
            "image_folder",
            os.path.expanduser(
                "~/robo_project_ws/src/robo_project/habitat_dataset/front"
            )
        )

        self.image_folder = self.get_parameter(
            "image_folder"
        ).value

        self.image_files = sorted(
            os.listdir(self.image_folder)
        )

        self.image_index = 0

        self.timer = self.create_timer(
            1.0,
            self.run_navigation
        )

        self.get_logger().info(
            "ML Pose Navigation Node Started"
        )

    def run_navigation(self):

        if self.image_index >= len(self.image_files):

            self.get_logger().info(
                "Finished all dataset images"
            )

            rclpy.shutdown()

            return

        image_name = self.image_files[self.image_index]

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        x, z = predict_pose(image_path)

        self.get_logger().info(
            f"Image: {image_name}"
        )

        self.get_logger().info(
            f"Predicted Position -> X: {x:.2f}, Z: {z:.2f}"
        )

        twist = Twist()

        if x > 0.5:

            twist.angular.z = -0.5

            command = "TURN RIGHT"

        elif x < -0.5:

            twist.angular.z = 0.5

            command = "TURN LEFT"

        else:

            twist.linear.x = 0.5

            command = "MOVE FORWARD"

        self.publisher_.publish(twist)

        self.get_logger().info(
            f"Published Command: {command}"
        )

        self.image_index += 1


def main(args=None):

    rclpy.init(args=args)

    node = MLPoseNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()