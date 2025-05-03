import cv2
from ultralytics import YOLO
import numpy as np

ver = 0
# Kalman Filter class definition
class KalmanFilter(object):
    def __init__(self, dt, INIT_POS_STD, INIT_VEL_STD, ACCEL_STD, GPS_POS_STD):
        """
        Initialize the Kalman Filter for 2D position tracking.

        :param dt: Sampling time (time for 1 cycle)
        :param INIT_POS_STD: Initial position standard deviation in x-direction
        :param INIT_VEL_STD: Initial position standard deviation in y-direction
        :param ACCEL_STD: Process noise magnitude (acceleration standard deviation)
        :param GPS_POS_STD: Standard deviation of the measurement
        """
        # Define sampling time
        self.dt = dt

        # Initial state [x, y, vx, vy]^T, initialized to zeros
        self.x = np.zeros((4, 1))

        # State estimate covariance matrix
        cov = np.zeros((4, 4))
        cov[0, 0] = INIT_POS_STD ** 2
        cov[1, 1] = INIT_POS_STD ** 2
        cov[2, 2] = INIT_VEL_STD ** 2
        cov[3, 3] = INIT_VEL_STD ** 2
        self.P = cov

        # State transition matrix
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Process noise covariance
        q = np.zeros((2, 2))
        q[0, 0] = ACCEL_STD ** 2
        q[1, 1] = ACCEL_STD ** 2
        self.q = q

        # Process model sensitivity matrix
        L = np.zeros((4, 2))
        L[0, 0] = 0.5 * self.dt ** 2
        L[1, 1] = 0.5 * self.dt ** 2
        L[2, 0] = self.dt
        L[3, 1] = self.dt
        self.L = L

        # Process noise covariance matrix
        self.Q = np.dot(self.L, np.dot(self.q, self.L.T))

        # Measurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Measurement covariance matrix
        R = np.zeros((2, 2))
        R[0, 0] = GPS_POS_STD ** 2
        R[1, 1] = GPS_POS_STD ** 2
        self.R = R

    def predict(self):
        """Perform the prediction step of the Kalman Filter."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        x_pred = self.x[0, 0]
        y_pred = self.x[1, 0]
        return x_pred, y_pred

    def update(self, z):
        """Perform the update step of the Kalman Filter with a measurement."""
        z = np.array(z).reshape(2, 1)
        z_hat = np.dot(self.H, self.x)
        self.y = z - z_hat
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        self.K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(self.S)))
        I = np.eye(4)
        self.x = self.x + np.dot(self.K, self.y)
        self.P = np.dot((I - np.dot(self.K, self.H)), self.P)
        x_updated = self.x[0, 0]
        y_updated = self.x[1, 0]
        return x_updated, y_updated

# Function to get bounding box center from a frame
def get_bounding_box_center_frame(frame, model, names, object_class='person'):
    """
    Detect persons in a frame and return the center of the first detected person's bounding box.

    :param frame: Input frame from the camera
    :param model: YOLO model instance
    :param names: Class names dictionary from the YOLO model
    :param object_class: Class to detect (default: 'person')
    :return: List of centers, empty if no person detected
    """
    centers = []
    results = model(frame)
    person_detected = False

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            class_name = names.get(int(class_id))
            if class_name == object_class and score > 0.5:
                if not person_detected:
                    person_detected = True
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    centers.append((center_x, center_y))
                    break  # Only take the first detected person
    return centers


# Main script
if __name__ == '__main__':
    # Load the YOLO model
    # Ensure 'yolov8n.pt' is in the same directory as this script or provide the full path
    model = YOLO('yolov8n.pt')
    names = model.names

    # Kalman filter parameters
    dt = 1 / 30  # Assuming 30 FPS for simplicity
    INIT_POS_STD = 10    # Initial position uncertainty
    INIT_VEL_STD = 10    # Initial velocity uncertainty
    ACCEL_STD = 40       # Process noise (acceleration)
    GPS_POS_STD = 1      # Measurement noise

    # Initialize the Kalman Filter
    kf = KalmanFilter(dt, INIT_POS_STD, INIT_VEL_STD, ACCEL_STD, GPS_POS_STD)

    # Initialize video capture from the laptop's default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Get camera properties for the output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Default FPS for output video

    # Set up video writer (optional, saves the output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_camera.mp4', fourcc, fps, (width, height))

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect person's bounding box center
        centers = get_bounding_box_center_frame(frame, model, names)

        # Perform Kalman Filter prediction
        x_pred, y_pred = kf.predict()
        # Draw predicted position (blue circle)
        cv2.circle(frame, (int(x_pred + np.random.normal(0, 100)*ver), int(y_pred + np.random.normal(0, 100)*ver)), radius=8, color=(255, 0, 0), thickness=4)

        # If a person is detected, update the Kalman Filter
        if len(centers) > 0:
            center = centers[0]
            # Draw measurement position (green circle)
            cv2.circle(frame, center, radius=8, color=(0, 255, 0), thickness=4)
            # Update the filter with the measurement
            x_updt, y_updt = kf.update(center)
            # Draw updated position (red circle)
            cv2.circle(frame, (int(x_updt), int(y_updt)), radius=8, color=(0, 0, 255), thickness=4)

        # Add legend to the frame
        legend_positions = {
            'Measurement': ((20, 20), (0, 255, 0)),  # Green
            'Prediction': ((20, 50), (255, 0, 0)),   # Blue
            'Update': ((20, 80), (0, 0, 255))        # Red
        }
        for text, (pos, color) in legend_positions.items():
            cv2.circle(frame, pos, radius=6, color=color, thickness=-1)
            cv2.putText(frame, text, (40, pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Person Tracking', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()