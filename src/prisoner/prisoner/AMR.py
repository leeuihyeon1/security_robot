import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import cv2
import numpy as np
from rclpy.qos import QoSProfile


class TurtleBotFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_follower')

        # 퍼블리셔 설정
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_publisher = self.create_publisher(CompressedImage, 'ARM/image/compressed', 10)
        self.tracking_mode_publisher = self.create_publisher(Bool, 'tracking_mode', 10)

        # 서브스크라이버 설정: CommendCenter의 emergency_zone_status 구독
        self.emergency_status_subscriber = self.create_subscription(
            Bool,
            'emergency_zone_status',
            self.emergency_status_callback,
            10
        )

        # YOLOv8 모델 로드
        try:
            self.model = YOLO('yolo_weight/best.pt')  # YOLOv8 모델 가중치 경로
            self.get_logger().info('YOLO 모델이 성공적으로 로드되었습니다.')
        except Exception as e:
            self.get_logger().error(f'YOLO 모델 로드 실패: {e}')
            raise e

        # 웹캠 초기화 (기본 카메라: 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('카메라를 열 수 없습니다. 장치 번호: 0')
            raise IOError("Cannot open camera 0")

        # 타이머 설정 (15 FPS)
        timer_period = 1.0 / 15.0  # 초 단위
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 제어 파라미터
        self.target_size = 450   # 목표 객체 크기 (픽셀 단위)
        self.size_tolerance = 30  # 크기 허용 오차
        self.horizontal_tolerance = 20  # 화면 중심에서의 위치 허용 오차

        # 상태 변수
        self.tracking_mode = False  # 추적 모드 활성화 여부
        self.emergency_zone_arrived = False  # Emergency Zone 도착 여부

        self.get_logger().info('TurtleBotFollower 노드가 시작되었습니다.')

    def emergency_status_callback(self, msg):
        """
        CommendCenter에서 퍼블리시한 emergency_zone_status 메시지를 처리합니다.
        비상 구역에 도착했을 때만 control_turtlebot을 실행합니다.
        """
        self.emergency_zone_arrived = msg.data
        if self.emergency_zone_arrived:
            self.get_logger().info('Emergency Zone에 도착했습니다.')
            # 비상 구역에 도착했을 때 추가 동작이 필요하면 여기서 구현

    def timer_callback(self):
        ret, frame = self.cap.read()
        current_time = self.get_clock().now()

        if not ret or frame is None:
            self.get_logger().error('카메라에서 프레임을 읽어올 수 없습니다.')
            return

        # YOLOv8을 사용한 객체 탐지
        results = self.model(frame)

        # 감지된 객체 중 'prisoner' 클래스 찾기
        prisoner_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])

                if confidence < 0.8:
                    continue

                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if cls_name.lower() == 'prisoner':
                    prisoner_detected = True

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 사각 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls_name} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    self.get_logger().info(f'객체: {cls_name}, 좌표: ({x1}, {y1}), ({x2}, {y2}), 신뢰도: {confidence:.2f}')

                    if not self.tracking_mode:
                        self.tracking_mode = True
                        tracking_mode_msg = Bool()
                        tracking_mode_msg.data = True
                        self.tracking_mode_publisher.publish(tracking_mode_msg)
                        self.get_logger().info('Tracking mode를 True로 퍼블리시했습니다.')

                    # tracking_mode가 True이므로 control_turtlebot 실행
                    if self.should_control_turtlebot():
                        self.control_turtlebot(x1, y1, x2, y2, frame.shape)
                    break  # 'prisoner' 객체를 찾았으므로 더 이상 탐색하지 않음
            if prisoner_detected:
                break

        if not prisoner_detected:
            if self.tracking_mode:
                # tracking_mode=True이고 'prisoner'를 감지하지 못한 경우 지속적으로 회전하여 탐색
                self.get_logger().info('Tracking mode 활성화 상태. Prisoner를 찾기 위해 회전합니다.')
                self.control_turtlebot(None, None, None, None, frame.shape)
            else:
                self.get_logger().info('Prisoner 객체가 감지되지 않았습니다.')

        # 이미지 퍼블리시 (사각 박스 그린 후)
        self.publish_image(frame, current_time)

    def should_control_turtlebot(self):
        """
        control_turtlebot을 실행해야 하는지 여부를 판단합니다.
        두 가지 상황:
        1. 비상 구역으로 이동 중에 prisoner를 감지한 경우 (tracking_mode=True)
        2. emergency_zone에 도착한 경우 (emergency_zone_arrived=True)
        """
        return self.tracking_mode or self.emergency_zone_arrived

    def publish_image(self, frame, current_time):
        # 압축된 이미지를 퍼블리시
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            compressed_image = CompressedImage()
            compressed_image.header.stamp = current_time.to_msg()
            compressed_image.format = "jpeg"
            compressed_image.data = buffer.tobytes()
            self.image_publisher.publish(compressed_image)
            self.get_logger().info('사각 박스가 그려진 이미지를 ARM/image/compressed 토픽으로 퍼블리시했습니다.')
        except Exception as e:
            self.get_logger().error(f'이미지 퍼블리시 실패: {e}')

    def control_turtlebot(self, x1, y1, x2, y2, frame_shape):
        twist = Twist()
        linear_vel = 0.0
        angular_vel = 0.0

        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            # 이미지의 중심 좌표 계산
            image_center_x = frame_shape[1] / 2  # frame_shape[1] is width
            image_center_y = frame_shape[0] / 2  # frame_shape[0] is height

            # 객체의 중심 좌표 계산
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # 중심 좌표 간의 차이 계산
            error_x = object_center_x - image_center_x

            # 사각형 박스의 크기 계산 (너비와 높이의 평균)
            object_width = x2 - x1
            object_height = y2 - y1
            object_size = (object_width + object_height) / 2

            # 목표 크기에 도달했는지 확인
            size_error = self.target_size - object_size

            # 회전 속도 계산 (좌우 이동)
            if abs(error_x) > self.horizontal_tolerance:
                angular_vel = -0.002 * error_x  # 에러에 비례하여 회전 속도 설정
                self.get_logger().info(f"객체가 {'오른쪽' if error_x > 0 else '왼쪽'}에 있습니다. 회전 속도: {angular_vel:.2f}")
            else:
                self.get_logger().info("객체가 중앙에 있습니다.")

            # 직진 속도 계산 (전후 이동)
            if abs(size_error) > self.size_tolerance:
                linear_vel = 0.0005 * size_error  # 에러에 비례하여 직진 속도 설정
                self.get_logger().info(f"객체와의 거리 조정 중. 선속도: {linear_vel:.2f}")
            else:
                self.get_logger().info("객체와의 거리가 적절합니다.")
        else:
            # 객체가 감지되지 않은 경우, 회전하여 객체를 찾음
            angular_vel = 0.4  # 지속적인 회전 속도
            self.get_logger().info("Prisoner 객체가 감지되지 않았습니다. 지속적으로 회전하여 객체를 찾습니다.")

        # 속도 제한 설정
        max_linear_speed = 0.2
        max_angular_speed = 0.5
        linear_vel = max(-max_linear_speed, min(max_linear_speed, linear_vel))
        angular_vel = max(-max_angular_speed, min(max_angular_speed, angular_vel))

        # 속도 명령 생성 및 퍼블리싱
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"속도 명령 퍼블리시 - 선속도: {linear_vel:.2f}, 각속도: {angular_vel:.2f}")

    def destroy_node(self):
        self.cap.release()
        self.stop_robot()
        super().destroy_node()

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info("로봇을 정지합니다.")


def main(args=None):
    rclpy.init(args=args)
    turtlebot_follower = TurtleBotFollower()
    try:
        rclpy.spin(turtlebot_follower)
    except KeyboardInterrupt:
        turtlebot_follower.get_logger().info('Shutting down TurtleBotFollower node.')
    finally:
        turtlebot_follower.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()