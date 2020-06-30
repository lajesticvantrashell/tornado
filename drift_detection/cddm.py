
"""
The Tornado Framework medical Triage Edition
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Calibrated Drift Detection Method (CDDM) Implementation ***
Paper: None
Published in: None
URL: None
"""

import numpy as np

import warnings

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class CDDM(SuperDetector):
    """The Calibrated Drift Detection Method (CDDM) class."""

    DETECTOR_NAME = TornadoDic.CDDM

    def __init__(self, drift_confidence=0.1, warning_confidence=0.2, n=100):

        super().__init__()

        # Parameters
        self.warning_threshold = warning_confidence / n
        self.drift_threshold = drift_confidence / n
        self.window_size = n

        # Data storage
        self.window = []
        self.n_samples = 0
        self.total = 0

    def run(self, pr, confidence):

        self.window.append(pr-conf)
        if len(self.window) > self.window_size:
            self.total -= self.window.pop(0)

        self.n_samples += 1
        div = min(self.n_samples, self.window_size)
        pr_drift = 2*np.exp( - self.total**2 / 2 / div )


        warning_status, drift_status = False, False
        if pr_drift < self.drift_threshold:
            drift_status = True
        elif pr_drift < self.warning_threshold:
            warning_status = True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.window =  []

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence) + "." + str(self.window_size),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper() + ", " +
                "$N$:" + str(self.window_size).upper()]
