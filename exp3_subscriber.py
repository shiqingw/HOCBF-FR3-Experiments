import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class BallSubscriber(Node):
    def __init__(self):
        super().__init__('ball_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/vicon/ball/ball',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('Received message: \n%s' % msg)

def main(args=None):
    rclpy.init(args=args)
    ball_subscriber = BallSubscriber()
    rclpy.spin(ball_subscriber)
    ball_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()