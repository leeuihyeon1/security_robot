import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from interface.msg import DetectedObject
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class SecurityCamNode(Node):
    def __init__(self):
        super().__init__('security_cam_node')

        # 퍼블리셔 생성
        self.image_publisher = self.create_publisher(CompressedImage, 'security_cam/image/compressed', 10)
        self.object_info_publisher = self.create_publisher(DetectedObject, 'security_cam/object_info', 10)

        # CvBridge 초기화
        self.bridge = CvBridge()

        # YOLOv8 모델 로드
        self.model = YOLO('yolo_weight/best.pt')
        self.get_logger().info('YOLO 모델이 로드되었습니다.')

        # 웹캠 초기화 (기본 카메라: 0)
        self.cap = cv2.VideoCapture(0)

        # 타이머 설정 (30 FPS)
        timer_period = 1.0 / 30.0  # 초 단위
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('SecurityCamNode가 시작되었습니다.')

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('카메라에서 프레임을 읽을 수 없습니다.')
            return

        # 현재 시간 캡처
        current_time = self.get_clock().now()

        # YOLO 모델로 객체 감지 및 트래킹 수행
        results = self.model.track(source=frame, persist=True)

        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 신뢰도 (confidence)
                confidence = float(box.conf[0])

                if confidence < 0.8:
                    # 신뢰도가 0.8 미만인 경우 건너뜀
                    continue

                # 사각 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 클래스 ID 및 이름
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                # 트래킹 ID
                track_id = int(box.id[0]) if box.id is not None else -1

                # DetectedObject 메시지 생성
                object_info = DetectedObject()
                object_info.header = Header()
                object_info.header.stamp = current_time.to_msg()  # 동일한 타임스탬프 할당
                object_info.class_name = cls_name
                object_info.confidence = confidence
                object_info.x1 = x1
                object_info.y1 = y1
                object_info.x2 = x2
                object_info.y2 = y2
                object_info.track_id = track_id  # 트래킹 ID 추가
                detected_objects.append(object_info)

                # 사각 박스 그리기 및 ID 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{track_id} {cls_name} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                self.get_logger().info(
                    f'객체: {cls_name}, ID: {track_id}, 좌표: ({x1}, {y1}), ({x2}, {y2}), 감도: {confidence:.2f}'
                )

        # DetectedObject 메시지 퍼블리시
        for object_info in detected_objects:
            self.object_info_publisher.publish(object_info)

        # 압축된 이미지를 퍼블리시 (사각 박스 그린 후)
        _, buffer = cv2.imencode('.jpg', frame)  # 수정된 frame 사용
        compressed_image = CompressedImage()
        compressed_image.header.stamp = current_time.to_msg()  # 동일한 타임스탬프 할당
        compressed_image.format = "jpeg"
        compressed_image.data = buffer.tobytes()
        self.image_publisher.publish(compressed_image)

        # OpenCV 윈도우에 이미지 디스플레이
        cv2.imshow("Security Camera Feed", frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 코드
            self.destroy_node()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()  # OpenCV 윈도우 닫기
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SecurityCamNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()