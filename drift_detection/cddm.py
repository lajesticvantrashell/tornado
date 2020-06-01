
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

    def get_x0(self, pr, conf):
        return pr-conf

    def run(self, pr, confidence):

        self.n_samples += 1 # don't actually need this

        x0 = self.get_x0(pr, confidence)

        window = self.window = [x0] + self.window

        if len(self.window) > self.window_size:
            window = self.window = self.window[:-1]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            probs = np.exp( - np.cumsum(window)**2 / 2 / (1+np.arange(len(self.window))) )

        pr_drift = min(probs)

        if pr_drift < self.drift_threshold:
            warning_status = False
            drift_status = True
        elif pr_drift < self.warning_threshold:
            warning_status = True
            drift_status = False
        else:
            warning_status = False
            drift_status = False

        # self.total += x0
        # mean = self.total / self.n_samples
        #
        # print(pr_drift, warning_status, drift_status, sum(window), mean)

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
        if conf==0 or conf==1:
            return (pr-conf)/(1e-5)
        else:
            return (pr-conf)/(conf*(1-conf))
