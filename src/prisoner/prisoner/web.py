import os
import time
from threading import Thread, Lock
import simpleaudio as sa
import sqlite3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import cv2
import numpy as np

# Flask 애플리케이션 설정
app = Flask(__name__)
app.secret_key = 'your_secret_key'

USERNAME = 'user'
PASSWORD = 'password'


class FlaskNode(Node):
    def __init__(self):
        super().__init__('flask_node')
        
        # 패키지 설치 경로에서 리소스 경로 가져오기
        import ament_index_python
        package_share_directory = ament_index_python.get_package_share_directory('prisoner')
        
        # 템플릿과 정적 파일 경로 설정
        self.template_path = os.path.join(package_share_directory, 'templates')
        self.static_path = os.path.join(package_share_directory, 'static')
        self.alarm_path = os.path.join(package_share_directory, 'audio', 'alarm.wav')
        
        # 데이터베이스 경로를 홈 디렉토리 아래로 변경
        self.db_path = os.path.expanduser('~/prisoner_db/cctv_db.db')
        
        # 데이터베이스 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Flask 앱 설정 업데이트
        app.template_folder = self.template_path
        app.static_folder = self.static_path

        # ROS 2 subscriptions
        self.create_subscription(Bool, 'emergency_zone_status', self.emergency_zone_status_callback, 10)
        self.create_subscription(Bool, 'prison_break', self.prison_break_callback, 10)
        self.create_subscription(CompressedImage, 'commend_center/image/compressed', self.commend_center_image_callback, 10)
        self.create_subscription(CompressedImage, 'ARM/image/compressed', self.amr_image_callback, 10)
        self.create_subscription(Bool, 'tracking_mode', self.tracking_mode_callback, 10)

        # 상태 변수
        self.prison_break_status = False
        self.prev_prison_break_status = False  # 이전 상태 추적
        self.emergency_zone_status = False
        self.tracking_mode = False

        # 비디오 스트림 프레임
        self.commend_center_frame = None
        self.amr_frame = None

        # 스레드 락
        self.lock = Lock()

        # 알람 상태 변수
        self.alarm_playing = False

        # 새로운 Bool 퍼블리셔 추가 (홈 버튼용)
        self.home_publisher = self.create_publisher(Bool, 'home_button_topic', 10)
        self.get_logger().info("Home publisher created on topic 'home_button_topic'")

        # Flask 서버를 별도 스레드에서 실행
        flask_thread = Thread(target=self.start_flask)
        flask_thread.setDaemon(True)
        flask_thread.start()

    def emergency_zone_status_callback(self, msg):
        with self.lock:
            self.emergency_zone_status = msg.data
        self.get_logger().info(f"Emergency Zone Status: {'Arrived' if msg.data else 'Not Arrived'}")

    def prison_break_callback(self, msg):
        with self.lock:
            self.prison_break_status = msg.data
        self.get_logger().info(f"Prison Break Status: {'Prison Break Detected' if msg.data else 'No Issues'}")

        # 상태 전환 감지: False -> True
        if self.prison_break_status and not self.prev_prison_break_status:
            if not self.alarm_playing:
                self.alarm_playing = True
                self.get_logger().info("탈옥자 발생! 알람 재생 시작.")
                Thread(target=self.play_alarm).start()
        elif not self.prison_break_status and self.prev_prison_break_status:
            self.get_logger().info("탈옥 상태 해제.")

        # 이전 상태 업데이트
        self.prev_prison_break_status = self.prison_break_status

    def play_alarm(self):
        try:
            wave_obj = sa.WaveObject.from_wave_file(self.alarm_path)
            self.get_logger().info(f"알람 파일 경로: {self.alarm_path}")  # 경로 확인 로그
            self.get_logger().info("알람 재생 중...")
            play_obj = wave_obj.play()
            play_obj.wait_done()  # 재생이 끝날 때까지 대기
        except FileNotFoundError:
            self.get_logger().error(f"알람 파일을 찾을 수 없습니다: {self.alarm_path}")
        except Exception as e:
            self.get_logger().error(f"알람 재생 중 오류 발생: {e}")
        finally:
            self.alarm_playing = False
            self.get_logger().info("알람 종료.")

    def commend_center_image_callback(self, msg):
        try:
            with self.lock:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.commend_center_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.get_logger().info('Received commend center image')
        except Exception as e:
            self.get_logger().error(f'Error in commend center callback: {str(e)}')

    def amr_image_callback(self, msg):
        try:
            with self.lock:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.amr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.get_logger().info('Received AMR image')
        except Exception as e:
            self.get_logger().error(f'Error in AMR callback: {str(e)}')

    def tracking_mode_callback(self, msg):
        with self.lock:
            self.tracking_mode = msg.data
        self.get_logger().info(f"Tracking Mode: {'Active' if msg.data else 'Inactive'}")

    def start_flask(self):
        try:
            app.run(host='0.0.0.0', port=5001)  # 포트를 5001로 변경
        except Exception as e:
            self.get_logger().error(f'Flask server error: {e}')


# Flask 라우트 설정
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('monitoring'))
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('monitoring'))
        else:
            flash('Invalid username or password!', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))


@app.route('/monitoring', methods=['GET', 'POST'])
def monitoring():
    if 'username' not in session:
        flash('Please log in first!', 'warning')
        return redirect(url_for('login'))
    
    with node.lock:
        prison_break = node.prison_break_status
        emergency_zone = node.emergency_zone_status
        tracking = node.tracking_mode

    # 상태 결정
    if prison_break:
        if emergency_zone:
            if tracking:
                status_message = "현장 도착 완료. 탈옥수 발견. 추적 중."
            else:
                status_message = "현장 도착 완료. 탈옥수 발견. 출동 중."
        else:
            if tracking:
                status_message = "탈옥수 발생. 추적 중."
            else:
                status_message = "탈옥수 발생. 출동 중."
    else:
        status_message = "교도소 정상 상태."

    return render_template('monitoring.html', username=session['username'], status=status_message)


@app.route('/home_button', methods=['POST'])
def home_button():
    """홈 버튼이 클릭되면 Bool 메시지를 퍼블리시"""
    msg = Bool()
    msg.data = True
    node.home_publisher.publish(msg)
    node.get_logger().info("Home button pressed. Bool message published.")
    flash('Home button pressed!', 'success')
    return redirect(url_for('monitoring'))


@app.route('/video_commend')
def video_commend():
    """CCTV 영상 스트림"""
    def generate():
        while True:
            with node.lock:
                if node.commend_center_frame is not None:
                    try:
                        _, buffer = cv2.imencode('.jpg', node.commend_center_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        node.get_logger().error(f'Error in video stream: {str(e)}')
                else:
                    node.get_logger().warn('No frame available')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_amr')
def video_amr():
    """AMR 영상 스트림"""
    def generate():
        while True:
            with node.lock:
                if node.amr_frame is not None:
                    try:
                        _, buffer = cv2.imencode('.jpg', node.amr_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        node.get_logger().error(f'Error in video stream: {str(e)}')
            time.sleep(0.033)  # 약 30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    with node.lock:
        prison_break = node.prison_break_status
        emergency_zone = node.emergency_zone_status
        tracking = node.tracking_mode

    # 상태 결정
    if prison_break:
        if emergency_zone:
            if tracking:
                status_message = "현장 도착 완료. 탈옥수 발견. 추적 중."
            else:
                status_message = "현장 도착 완료. 탈옥수 발견. 출동 중."
        else:
            if tracking:
                status_message = "탈옥수 발생. 추적 중."
            else:
                status_message = "탈옥수 발생. 출동 중."
    else:
        status_message = "교도소 정상 상태."

    return {"status": status_message}


def calculate_detection_by_time(records):
    time_counts = {}
    for record in records:
        # 시간대만 추출 (시:분)
        time_key = record['timestamp'].split()[1][:5]
        time_counts[time_key] = time_counts.get(time_key, 0) + 1
    return time_counts

def calculate_object_distribution(records):
    object_counts = {}
    for record in records:
        obj_type = record['class_name']
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    return object_counts

@app.route('/db')
def db():
    """데이터베이스에서 기록 조회 및 렌더링"""
    if 'username' not in session:
        node.get_logger().warning('DB 페이지 접근: 로그인되지 않은 사용자')
        flash('로그인이 필요합니다!', 'warning')
        return redirect(url_for('login'))
    
    node.get_logger().info(f'데이터베이스 경로: {node.db_path}')
    
    try:
        # 데이터베이스 파일 존재 여부 확인
        if not os.path.exists(node.db_path):
            node.get_logger().error(f'데이터베이스 파일이 존재하지 않음: {node.db_path}')
            flash('데이터베이스 파일을 찾을 수 없습니다.', 'error')
            return redirect(url_for('monitoring'))
            
        conn = sqlite3.connect(node.db_path)
        cursor = conn.cursor()
        
        # 테이블 존재 여부 확인
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='detections'
        ''')
        
        if not cursor.fetchone():
            node.get_logger().warning('detections 테이블이 존재하지 않음')
            # 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT,
                    confidence REAL,
                    x1 REAL,
                    y1 REAL,
                    x2 REAL,
                    y2 REAL,
                    timestamp TEXT,
                    image BLOB
                )
            ''')
            conn.commit()
            
        # 데이터 조회
        cursor.execute('''
            SELECT id, class_name, confidence, x1, y1, x2, y2, timestamp 
            FROM detections 
            ORDER BY timestamp DESC
        ''')
        
        records = [dict(zip(
            ['id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'timestamp'], 
            row)) for row in cursor.fetchall()]
        
        node.get_logger().info(f'조회된 레코드 수: {len(records)}')
        
        # 빈 데이터 처리
        if not records:
            node.get_logger().info('저장된 데이터가 없음')
            return render_template('db.html', 
                                records=[], 
                                timestamps=[], 
                                x_movement=[], 
                                y_movement=[], 
                                confidences=[],
                                detection_by_time={}, 
                                object_types=[], 
                                object_counts=[])
        
        # 데이터 처리
        timestamps = [record['timestamp'] for record in records]
        x_movement = [(record['x2'] - record['x1']) for record in records]
        y_movement = [(record['y2'] - record['y1']) for record in records]
        confidences = [record['confidence'] for record in records]
        
        detection_by_time = calculate_detection_by_time(records)
        
        object_distribution = calculate_object_distribution(records)
        object_types = list(object_distribution.keys())
        object_counts = list(object_distribution.values())
        
        conn.close()
        
        return render_template('db.html',
                             records=records,
                             timestamps=timestamps,
                             x_movement=x_movement,
                             y_movement=y_movement,
                             confidences=confidences,
                             detection_by_time=detection_by_time,
                             object_types=object_types,
                             object_counts=object_counts)
                             
    except sqlite3.OperationalError as e:
        node.get_logger().error(f'데이터베이스 작업 오류: {str(e)}')
        flash(f'데이터베이스 오류: {str(e)}', 'error')
        return redirect(url_for('monitoring'))
    except Exception as e:
        node.get_logger().error(f'예기치 않은 오류: {str(e)}')
        flash(f'오류 발생: {str(e)}', 'error')
        return redirect(url_for('monitoring'))
    finally:
        if 'conn' in locals():
            conn.close()


@app.route('/get_video/<int:record_id>')
def get_video(record_id):
    if 'username' not in session:
        return redirect(url_for('login'))
        
    try:
        conn = sqlite3.connect(node.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM detections WHERE id = ?', (record_id,))
        image_data = cursor.fetchone()
        conn.close()
        
        if image_data and image_data[0]:
            return Response(image_data[0], mimetype='image/jpeg')
        return '이미지를 찾을 수 없습니다.', 404
    except Exception as e:
        return f'오류 발생: {str(e)}', 500


def main(args=None):
    global node
    rclpy.init(args=args)
    node = FlaskNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Flask Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
