
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

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class CDDM(SuperDetector):
    """The Calibrated Drift Detection Method (CDDM) class."""

    DETECTOR_NAME = TornadoDic.CDDM

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, window_size=1000):

        # Parameters
        self.warning_threshold = warning_confidence / window_size
        self.drift_threshold = drift_confidence / window_size
        self.window_size = window_size

        # Data storage
        self.window = []
        self.n_samples = 0

    def get_x0(self, pr, conf):
        return pr-confidence

    def run(self, pr, confidence):

        x0 = self.get_x0(pr, confidence)

        window = self.window = [x0] + self.window

        if len(self.window) > self.window_size:
            window = self.window = self.window[1:]

        probs = np.exp( np.cumsum(window)**2 / 2 / (1+np.arange(len(self.window))) )

        pr_drift = min(probs)

        if pr_drift < self.drift_confidence:
            warning_status = False
            drift_status = True
        elif pr_drift < self.warning_confidence:
            warning_status = True
            drift_status = False

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.window =  []

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence) + "." + str(self.window_size),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper() + ", " +
                "$N$:" + str(self.window_size).upper()]


class CDDM2(CDDM):

    DETECTOR_NAME = TornadoDic.CDDM2

    def get_x0(self, pr, conf):
        return (pr-confidence)/(confidence*(1-confidence))
