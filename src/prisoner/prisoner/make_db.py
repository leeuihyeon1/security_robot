import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from interface.msg import DetectedObject
import sqlite3
import json
import os
import ament_index_python

class DetectionSubscriberNode(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        
        # 데이터베이스 경로 설정
        self.db_path = os.path.join('src', 'prisoner', 'prisoner', 'cctv_db.db')
        
        # 데이터베이스 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 데이터베이스 연결
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # 테이블 생성 호출 추가
        self.create_table()
        
        #self.bridge = CvBridge()
        self.latest_frame = None  # 최신 이미지 저장 변수
        
        # 객체 탐지 데이터를 받아오는 구독자
        self.detection_sub = self.create_subscription(
            DetectedObject,
            'security_cam/object_info', 
            self.listener_callback, 
            10
        )

        # 카메라 이미지를 받아오는 구독자
        self.image_sub = self.create_subscription(
            CompressedImage,
            'commend_center/image/compressed',
            self.image_callback,
            10
        )

        # amr 이미지를 받아오는 구독자
        self.amr_image_sub = self.create_subscription(
            CompressedImage,
            'ARM/image/compressed',
            self.image_callback,
            10
        )

        self.detection_sub  # 객체를 유지하여 구독을 활성화

    def image_callback(self, msg):
        """ 이미지 콜백 함수 """
        try:
            # CompressedImage를 OpenCV 이미지로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)  # 바이너리 데이터를 numpy 배열로 변환
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV 이미지로 디코딩
            # JPEG로 인코딩
            ret1, jpeg = cv2.imencode('.jpg', self.image)
            if ret1:
                self.latest_frame = jpeg.tobytes()  # 최신 이미지를 바이트 데이터로 저장
        except Exception as e:
            print(f"Error in image callback: {e}")
            self.get_logger().error(f"Error in image callback: {e}")
            self.latest_frame = None

    def create_table(self):
        """ 
        테이블 구조:
        - id: 각 레코드의 고유 식별자 (자동 증가)
        - class_name: 탐지된 객체의 클래스 이름
        - confidence: 객체 탐지의 신뢰도 점수
        - x1, y1, x2, y2: 탐지된 객체의 좌표
        - timestamp: 탐지 시점의 시간 데이터
        - image: 탐지 시점의 이미지 데이터 (BLOB 형식)  # 바이너리 데이터를 저장할 때 데이터 타입
        """
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                class_name TEXT,
                                confidence REAL,
                                x1 INTEGER,
                                y1 INTEGER,
                                x2 INTEGER,
                                y2 INTEGER,
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                image BLOB
                            )''')
            self.conn.commit()
        except sqlite3.Error as e:
            self.get_logger().error(f"Error creating table: {e}")

    def listener_callback(self, msg):
        """ 객체 탐지 데이터를 받아오는 콜백 함수 """
        try:
            # 신뢰도가 0.8 이상인 경우에만 저장
            if msg.confidence >= 0.8:
                self.save_detection_data(msg)
            else:
                self.get_logger().info(f"낮은 신뢰도로 인해 무시됨: {msg.confidence}")
        except Exception as e:
            self.get_logger().error(f"Failed to process detection data: {e}")

    def save_detection_data(self, detection_msg):
        try:
            # 데이터베이스에 삽입
            self.cursor.execute('''INSERT INTO detections 
                                (class_name, confidence, x1, y1, x2, y2, image)
                                VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                                (detection_msg.class_name,
                                detection_msg.confidence,
                                detection_msg.x1,
                                detection_msg.y1,
                                detection_msg.x2,
                                detection_msg.y2,
                                self.latest_frame))
            self.conn.commit()
            self.get_logger().info(f"Saved detection: {detection_msg.class_name}")
        except sqlite3.Error as e:
            self.get_logger().error(f"Error saving detection data: {e}")

    def destroy_node(self):
        self.conn.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('DetectionSubscriberNode 노드가 종료됩니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()