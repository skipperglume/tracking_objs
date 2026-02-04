import numpy as np


class KalmanFilter2D:
    """
    Class for a simple 2D Kalman Filter for tracking position and velocity.
    """

    def __init__(self, cx, cy, dt=1.0):
        self.dt = dt

        # State: [cx, cy, vx, vy]
        self.x = np.array([[cx], [cy], [0.0], [0.0]])

        # State transition
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Measurement model (we measure position only)
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # Covariances
        self.P = np.eye(4) * 500.0  # state uncertainty
        self.R = np.eye(2) * 10.0  # measurement noise
        self.Q = np.eye(4) * 1.0  # process noise

        self.I = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, cx, cy):
        z = np.array([[cx], [cy]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def predict_n_steps(self, n):
        x = self.x.copy()
        P = self.P.copy()

        points = []

        for _ in range(n):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            points.append(x[:2].flatten())

        return points

    @property
    def position(self):
        return self.x[0, 0], self.x[1, 0]

    @property
    def velocity(self):
        return self.x[2, 0], self.x[3, 0]
