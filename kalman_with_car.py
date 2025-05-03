import cv2
from ultralytics import YOLO
import numpy as np
import os

# ---------------- Kalman Filter definition ---------------- #
class KalmanFilter(object):
    def __init__(self, dt, INIT_POS_STD, INIT_VEL_STD, ACCEL_STD, GPS_POS_STD):
        self.dt = dt

        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))

        # Covariance
        self.P = np.diag([INIT_POS_STD**2, INIT_POS_STD**2,
                          INIT_VEL_STD**2, INIT_VEL_STD**2])

        # State-transition
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Process noise
        q = np.diag([ACCEL_STD**2, ACCEL_STD**2])
        L = np.array([[0.5*dt**2,         0],
                      [        0, 0.5*dt**2],
                      [       dt,         0],
                      [        0,        dt]])
        self.Q = L @ q @ L.T

        # Measurement mapping and noise
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.diag([GPS_POS_STD**2, GPS_POS_STD**2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(2, 1)
        y  = z - self.H @ self.x                      # Innovation
        S  = self.H @ self.P @ self.H.T + self.R      # Innovation cov
        K  = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x += K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x[0, 0], self.x[1, 0]

# --------------- helper: find first car centre ------------- #
def get_bounding_box_center_frame(frame, model, names, object_class='car'):
    centers = []
    results = model(frame)
    for result in results:
        for (x1, y1, x2, y2, score, class_id) in result.boxes.data.tolist():
            if names[int(class_id)] == object_class and score > 0.5:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))
                return centers   # take the first detection only
    return centers

# ----------------------------- main ------------------------ #
if __name__ == '__main__':
    # ---- 1. open the video file instead of the webcam ---- #
    video_path = r"C:\Users\Асан\Downloads\car2.mp4"
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    # ---- 2. set FPS and Kalman timestep from the video ---- #
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:          # NaN check
        fps = 30                        # sensible default
    dt = 1 / fps

    # Load YOLOv8 nano
    model = YOLO('yolov8n.pt')
    names = model.names

    # Kalman filter hyper-parameters
    kf = KalmanFilter(dt, INIT_POS_STD=10, INIT_VEL_STD=10,
                      ACCEL_STD=40, GPS_POS_STD=1)

    # ---- 3. prepare the output writer (optional) ---- #
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter('car_tracking_output.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break   # video finished

        centers = get_bounding_box_center_frame(frame, model, names, object_class='car')

        # Kalman predict step (blue)
        x_pred, y_pred = kf.predict()
        cv2.circle(frame, (int(x_pred), int(y_pred)), 8, (255, 0, 0), 4)

        # Kalman update step if measurement exists (green/red)
        if centers:
            m = centers[0]
            cv2.circle(frame, m, 8, (0, 255, 0), 4)  # measurement
            x_upd, y_upd = kf.update(m)
            cv2.circle(frame, (int(x_upd), int(y_upd)), 8, (0, 0, 255), 4)  # updated state

        # Simple legend
        legend = [('Measurement', (0,255,0), 20),
                  ('Prediction' , (255,0,0), 50),
                  ('Update'     , (0,0,255), 80)]
        for text, clr, y in legend:
            cv2.circle(frame, (20, y), 6, clr, -1)
            cv2.putText(frame, text, (40, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

        out.write(frame)
        cv2.imshow('Car Tracking', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
