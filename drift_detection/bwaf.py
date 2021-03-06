
"""
The Tornado Framework medical Triage Edition
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** Beta With Adaptive Forgetfulness (BWAF) Implementation ***
Paper: None
Published in: None
URL: None
"""

import numpy as np
from scipy import integrate
from scipy.special import betainc
from scipy.stats import beta
from scipy.special import beta as beta_func

import functools
import collections

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector

class BWAF(SuperDetector):
    """Beta With Adaptive Forgetfulness (BWAF) class."""

    DETECTOR_NAME = TornadoDic.BWAF

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005):

        super().__init__()

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.a = 0
        self.b = 0
        self.A = 0
        self.B = 0
        self.gamma = 0.5

    @staticmethod
    def beta_median(a, b):
        return (a-1/3)/(a+b-2/3)

    @staticmethod
    def pr_drift(a, b, A, B):
        # MIN_FOR_GAUSSIAN = 30
        # if all(np.array([a,b,A-a,B-b]) > MIN_FOR_GAUSSIAN):
        #
        q0_median = BWAF.beta_median(A-a+1, B-b+1)
        return 1 - 0.5 * (1-betainc(a+1,b+1, q0_median))

    # @staticmethod
    # def integrand(x, a, b, A, B):
    #     return beta.pdf(x, A-a+1, B-b+1)*betainc(a+1,b+1, x)
    #
    # @staticmethod
    # def pr_drift(a, b, A, B):
    #     int = lambda x: BWAF.integrand(x, a, b, A, B)
    #     return integrate.quad(int, 0, 1, limit=20, epsabs=0.001)[0]

    def run(self, pr):

        self.a = pr + self.gamma*self.a
        self.b = 1-pr + self.gamma*self.b
        self.A += pr
        self.B += 1-pr
        a, b, A, B = round(self.a, 2), round(self.b, 2), round(self.A, 2), round(self.B, 2)
        pr_drift = BWAF.pr_drift(self.a, self.b, self.A, self.B)
        self.gamma = pr_drift

        if 1-pr_drift < self.drift_confidence:
            warning_status = False
            drift_status = True
        elif 1-pr_drift < self.warning_confidence:
            warning_status = True
            drift_status = False
        else:
            warning_status = False
            drift_status = False

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.a = 0
        self.b = 0
        self.A = 0
        self.B = 0
        self.gamma = 0.5

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper()]
