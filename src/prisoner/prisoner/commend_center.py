import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from interface.msg import DetectedObject  
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from threading import Lock
import time

class CommendCenter(Node):
    def __init__(self):
        super().__init__('commend_center')

        # 서브스크립션 설정
        self.subscription_image = self.create_subscription(
            CompressedImage,
            '/security_cam/image/compressed',
            self.image_callback,
            10
        )
        self.subscription_object_info = self.create_subscription(
            DetectedObject,
            '/security_cam/object_info',
            self.object_info_callback,
            10
        )
        self.subscription_arm_image = self.create_subscription(
            CompressedImage,
            'ARM/image/compressed',
            self.amr_image_callback,
            10
        )

        # 퍼블리셔 추가: Emergency Zone 도착 상태 알림 (bool 형식)
        self.status_publisher = self.create_publisher(
            Bool,
            'emergency_zone_status',
            10
        )

        # 퍼블리셔 추가: 죄수 발생 시 상태 알림 (bool 형식)
        self.prisoner_publisher = self.create_publisher(
            Bool,
            'prison_break',
            10
        )
        
        # **추가된 부분**: 처리된 이미지를 퍼블리시할 퍼블리셔 생성
        self.image_publisher = self.create_publisher(
            CompressedImage,
            'commend_center/image/compressed',
            10
        )

        # 액션 클라이언트 설정
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 비상 상태 변수 초기화
        self.emergency_state = False
        self.emergency_start_time = None
        self.emergency_lock = Lock()

        # 비상 네비게이션 완료 플래그
        self.emergency_navigation_done = False

        # 구역 정의
        self.zone_coordinates = {
            0: (0.5, 0.3),  # 안전 구역 (사용하지 않음)
            1: (-0.88348, -0.13553),
            2: (-0.85885, -0.389579),
            3: (-0.53394, -0.17463),
            4: (-0.55284, -0.69495),
            5: (-0.33947, -0.17189),
            6: (-0.31382, -0.64412),
        }

        # 순차적으로 이동할 웨이포인트 목록 (웨이포인트 1, 2, 3)
        self.waypoints = [
            (0.274173, -0.175006),  # Waypoint 1
            (0.231953, -0.564693),  # Waypoint 2
            (-0.07914, -0.580167)   # Waypoint 3
        ]

        # 비상 구역 목록 (3열 × 2행)
        self.emergency_zones = [
            self.zone_coordinates[1],
            self.zone_coordinates[2],
            self.zone_coordinates[3],
            self.zone_coordinates[4],
            self.zone_coordinates[5],
            self.zone_coordinates[6]
        ]

        # 현재 선택된 비상 구역 좌표 (초기에는 None)
        self.selected_emergency_zone = None

        self.zones = None

        # 객체 추적 변수
        self.current_objects = []
        self.objects_lock = Lock()
        self.prisoner_in_emergency_zone = False

        # 네비게이션 상태 관리
        self.navigation_goals = []          # 전체 네비게이션 목표 리스트 (웨이포인트 + 비상 구역)
        self.current_goal_index = 0         # 현재 목표 인덱스
        self.current_goal_x = 0.0           # 현재 목표 x 좌표
        self.current_goal_y = 0.0           # 현재 목표 y 좌표
        self._goal_handle = None            # 현재 목표 핸들
        self.navigating = False              # 현재 네비게이션 중인지 여부

        # 비상 상태를 주기적으로 확인하는 타이머 설정
        self.timer = self.create_timer(0.1, self.check_emergency_state)

        # prisoner가 속한 비상 구역 인덱스 초기화
        self.prisoner_zone_index = -1

        # **추가된 부분**: 죄수 상태 추적 변수 초기화
        self.prisoner_state = False  # 현재 죄수 상태 (False: 발생하지 않음, True: 발생)

        self.get_logger().info('CommendCenter 노드가 시작되었습니다.')

    def define_zones(self, width, height):
        zones = []
        # 안전 구역
        safe_zone_width = width // 3  # 왼쪽 1/3
        zones.append((0, 0, safe_zone_width, height))  # 안전 구역

        # 비상 구역을 3열 × 2행으로 나누기
        emergency_zone_width = (width * 2) // 3  # 오른쪽 2/3
        emergency_cols = 3
        emergency_rows = 2
        emergency_zone_width_individual = emergency_zone_width // emergency_cols
        emergency_zone_height_individual = height // emergency_rows

        for row in range(emergency_rows):
            for col in range(emergency_cols):
                x1 = safe_zone_width + col * emergency_zone_width_individual
                y1 = row * emergency_zone_height_individual
                x2 = safe_zone_width + (col + 1) * emergency_zone_width_individual if (col + 1) < emergency_cols else width
                y2 = (row + 1) * emergency_zone_height_individual if (row + 1) < emergency_rows else height
                zones.append((x1, y1, x2, y2))

        return zones

    def image_callback(self, msg):
        # 압축된 이미지를 OpenCV 형식으로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.zones is None:
            height, width, _ = cv_image.shape
            self.zones = self.define_zones(width, height)
            self.get_logger().info(f'Zones defined: {len(self.zones)} zones.')

        # 현재 객체 복사
        with self.objects_lock:
            objects_to_draw = self.current_objects.copy()
            self.current_objects.clear()

        prisoner_in_zone = False
        prisoner_zone_index = -1

        # 감지된 객체 그리기
        for obj in objects_to_draw:
            x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
            class_name = obj.class_name
            confidence = obj.confidence
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(cv_image, f'{class_name}: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # prisoner가 비상 구역에 있는지 확인
            if class_name.lower() == 'prisoner':
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                zone_idx = self.get_zone_index(x_center, y_center, self.zones)
                if zone_idx > 0:
                    prisoner_in_zone = True
                    prisoner_zone_index = zone_idx
                    self.get_logger().info(f'Prisoner is in Emergency Zone {zone_idx}.')
                else:
                    self.get_logger().info('Prisoner is in Safe Zone or outside defined zones.')

        # 'prisoner_in_emergency_zone' 플래그 업데이트
        with self.emergency_lock:
            self.prisoner_in_emergency_zone = prisoner_in_zone
            if prisoner_in_zone:
                self.prisoner_zone_index = prisoner_zone_index
            else:
                self.prisoner_zone_index = -1

        # 이미지에 구역 그리기
        for idx, (x1, y1, x2, y2) in enumerate(self.zones):
            if idx == 0:
                # 안전 구역 - 녹색
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, 'Safe Zone', (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 비상 구역 - 빨간색
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cv_image, f'Emergency{idx}', (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # prisoner가 비상 구역에 있는 경우 색상 반전
        if prisoner_in_zone:
            cv_image = cv2.bitwise_not(cv_image)  # 색상 반전

        # **추가된 부분**: 처리된 이미지를 퍼블리시
        try:
            # CvBridge를 사용하여 OpenCV 이미지를 CompressedImage 메시지로 변환
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image, dst_format='jpeg')
            # 퍼블리시
            self.image_publisher.publish(compressed_msg)
            self.get_logger().debug('Processed image published to commend_center/image/compressed.')
        except Exception as e:
            self.get_logger().error(f'Failed to publish processed image: {e}')

        # 이미지 시각화
        cv2.imshow('Security Camera', cv_image)
        cv2.waitKey(1)

    def amr_image_callback(self, msg):
        # 압축된 이미지를 OpenCV 형식으로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 이미지 시각화
        cv2.imshow('ARM Image', cv_image)
        cv2.waitKey(1)

    def object_info_callback(self, msg):
        # 객체를 current_objects에 추가
        with self.objects_lock:
            self.current_objects.append(msg)

    def get_zone_index(self, x, y, zones):
        for idx, (x1, y1, x2, y2) in enumerate(zones):
            if x1 <= x < x2 and y1 <= y < y2:
                return idx
        return -1  # 구역 외

    def check_emergency_state(self):
        with self.emergency_lock:
            if self.prisoner_in_emergency_zone and not self.emergency_navigation_done:
                if not self.emergency_state:
                    self.emergency_state = True
                    self.emergency_start_time = time.time()
                    self.get_logger().info('Emergency state started.')
                else:
                    elapsed_time = time.time() - self.emergency_start_time
                    if elapsed_time >= 3.0:
                        # 비상 상태가 3초 이상 지속됨
                        self.emergency_state = False  # 비상 상태 초기화
                        self.get_logger().info('Emergency state persisted for 3 seconds, taking action.')
                        self.send_navigation_goal(self.prisoner_zone_index)
                
                # **추가된 부분**: 죄수 발생 시 퍼블리시
                if not self.prisoner_state:
                    self.prisoner_state = True
                    self.publish_prisoner_state(True)
            else:
                # 비상 구역에 prisoner가 없거나 이미 비상 네비게이션이 완료됨
                if self.emergency_state:
                    self.emergency_state = False
                    self.get_logger().info('Emergency state reset.')
                
                # **추가된 부분**: 죄수 발생 상태 해제 시 퍼블리시
                if self.prisoner_state:
                    self.prisoner_state = False
                    self.publish_prisoner_state(False)
                # 추가적으로 필요한 로직이 있다면 여기에 구현

    def publish_prisoner_state(self, state: bool):
        """
        죄수 발생 상태를 퍼블리시하는 함수.
        :param state: 죄수 발생 상태 (True: 발생, False: 해제)
        """
        status_msg = Bool()
        status_msg.data = state
        self.prisoner_publisher.publish(status_msg)
        state_str = '발생' if state else '해제'
        self.get_logger().info(f'Prisoner state published: {state_str}.')

    def send_navigation_goal(self, zone_index):
        if self.emergency_navigation_done:
            self.get_logger().info('Emergency navigation has already been completed. No further actions.')
            return

        if self.navigating:
            self.get_logger().info('Already navigating. Waiting for current navigation to complete.')
            return

        # 구역 인덱스에 따른 좌표 매핑
        if zone_index in self.zone_coordinates:
            target_x, target_y = self.zone_coordinates[zone_index]
            self.get_logger().info(f'Preparing to navigate to Emergency Zone {zone_index}: x={target_x}, y={target_y}')
        else:
            self.get_logger().error(f'Invalid zone index: {zone_index}')
            return

        # 네비게이션 목표 리스트 설정: 웨이포인트 1~3 + 비상 구역
        self.navigation_goals = self.waypoints.copy()  # Waypoints 1~3
        self.navigation_goals.append((target_x, target_y))  # Emergency zone

        self.current_goal_index = 0  # 시작 인덱스
        self.navigating = True       # 네비게이션 중으로 설정

        self.get_logger().info('Starting sequential navigation through waypoints.')

        # 첫 번째 목표로 네비게이션 시작
        next_x, next_y = self.navigation_goals[self.current_goal_index]
        self.navigate_to_pose(next_x, next_y)

    def navigate_to_pose(self, x, y):
        self.current_goal_x = x
        self.current_goal_y = y

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose 액션 서버를 찾을 수 없습니다.')
            self.navigating = False
            return

        goal_msg = NavigateToPose.Goal()

        # 목표 위치 설정
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0  # 정면 향함

        goal_msg.pose = goal_pose

        self.get_logger().info(f'Sending navigation goal to x: {x}, y: {y}')
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Navigation goal to x: {self.current_goal_x}, y: {self.current_goal_y} was rejected.')
            self.navigating = False
            return

        self.get_logger().info(f'Navigation goal to x: {self.current_goal_x}, y: {self.current_goal_y} accepted.')

        self._goal_handle = goal_handle

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        x = self.current_goal_x
        y = self.current_goal_y

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Navigation to x: {x}, y: {y} succeeded.')

            # 현재 목표가 비상 구역인지 확인
            if self.current_goal_index == len(self.navigation_goals) - 1:
                # 마지막 목표인 비상 구역에 도착
                self.emergency_navigation_done = True
                self.get_logger().info('Emergency navigation to the emergency zone completed.')

                # 퍼블리셔를 통해 비상 구역 도착 알림 발행 (bool 형식)
                status_msg = Bool()
                status_msg.data = True
                self.status_publisher.publish(status_msg)
                self.get_logger().info('Emergency navigation completion status published.')
            else:
                # 다음 웨이포인트로 이동
                self.current_goal_index += 1
                if self.current_goal_index < len(self.navigation_goals):
                    next_x, next_y = self.navigation_goals[self.current_goal_index]
                    self.get_logger().info(f'Moving to next goal: x: {next_x}, y: {next_y}')
                    self.navigate_to_pose(next_x, next_y)
                else:
                    # 모든 목표 완료 (비상 구역에 이미 도착한 경우)
                    self.navigating = False
                    self.emergency_navigation_done = True
                    self.get_logger().info('All navigation goals completed.')

                    # 퍼블리셔를 통해 비상 구역 도착 알림 발행 (bool 형식)
                    status_msg = Bool()
                    status_msg.data = True
                    self.status_publisher.publish(status_msg)
                    self.get_logger().info('Emergency navigation completion status published.')
        else:
            self.get_logger().warn(f'Navigation to x: {x}, y: {y} failed with status: {status}.')
            self.navigating = False

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # 피드백 처리 (필요 시 구현)
        pass

    def destroy_node(self):
        # OpenCV 창 닫기
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    commend_center = CommendCenter()
    try:
        rclpy.spin(commend_center)
    except KeyboardInterrupt:
        commend_center.get_logger().info('Shutting down CommendCenter node.')
    finally:
        commend_center.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
