
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

from scipy.special import betaln
import numpy as np

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class BDDM(SuperDetector):
    """Bayesian Drift Detection Method (BDDM) class."""

    DETECTOR_NAME = TornadoDic.BDDM

    def __init__(self, drift_confidence=0.01, warning_confidence=0.05, drift_rate=0.001, n=1000):

        super().__init__()

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.drift_rate = drift_rate
        self.win_len = n

        self.a = [] # all the successful trials
        self.b = [] # all the unsuccessful trials
        self.N = 0 # total number of trials

    def log_prior(self, t):
        if t==0:
            return np.log(1-self.drift_rate)*self.N
        else:
            return np.log(1-self.drift_rate)*(t-1) + np.log(self.drift_rate)

    def log_likelihood(self, drift_point):
        a1 = sum(self.a[:drift_point])
        b1 = sum(self.b[:drift_point])
        a2 = sum(self.a[drift_point:])
        b2 = sum(self.b[drift_point:])
        return betaln(a1+1, b1+1) + betaln(a2+1, b2+1)

    def run(self, pr):

        self.a.append(pr)
        self.b.append(1-pr)
        self.N += 1

        # if we have exceeded the window length, then combine the first two items in the window
        if self.win_len and len(self.a) > self.win_len:
            self.a = [sum(self.a[: 2])] + self.a[2: ]
            self.b = [sum(self.b[: 2])] + self.b[2: ]

        log_posteriors = []
        for k in range(self.N):
            log_posteriors.append( self.log_likelihood(k) + self.log_prior(k) )
        log_posteriors -= max(log_posteriors)
        posteriors = np.exp(log_posteriors)


        pr_stable = posteriors[0] / sum(posteriors)

        warning_status, drift_status = False, False
        if pr_drift < self.drift_threshold:
            drift_status = True
        elif pr_drift < self.warning_threshold:
            warning_status = True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.a = []
        self.b = []
        self.N = 0

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence) + "." + str(self.drift_rate) + "." + str(self.win_len),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper() + ", " +
                "$\lambda$:" + str(self.drift_rate).upper() + ", " +
                "$N$:" + str(self.win_len).upper() + ", " ]
