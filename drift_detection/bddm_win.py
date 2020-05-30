
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

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class CDDM(SuperDetector):
    """The Calibrated Drift Detection Method (CDDM) class."""

    DETECTOR_NAME = TornadoDic.CDDM

    def __init__(self, drift_threshold=0.01, window_size=100):

        # Parameters
        self.drift_threshold = drift_threshold
        self.drift_threshold /= window_size # Bonferroni correction
        self.window_size = window_size

        # Data storage
        self.buffer_x = [ None for i in range(window_size) ]
        self.buffer_y = [ None for i in range(window_size) ]

        # Calcalations
        self.hoeffding_bounds = []
        self.warning = False
        self.drift = False
        self.n_samples = 0

    def run(self, probability_distribution, true_label, mode='normal', x=None):

        if mode=='tornado':
            labels = list(probability_distribution.keys())
            true_label = [labels.index(true_label)]
            probability_distribution = np.array([[ probability_distribution[label] for label in labels ]])

        self.n_samples += 1

        phi = np.argmax(probability_distribution)
        y = [ 1 if i==true_label[0] else 0 for i in range(probability_distribution.shape[1]) ]
        x = probability_distribution[0, phi] - y[phi]
        buffer_x = self.buffer_x
        buffer_x = [x] + buffer_x[:-1]
        self.buffer_x = buffer_x
        k = [ np.mean(buffer_x[:i]) for i in range(1, min(len(buffer_x), self.n_samples)) ]
        t = np.arange(min(len(buffer_x), self.n_samples)-1) + 1
        self.hoeffding_bounds = np.exp(- np.array(k)**2 * t / 2)



        warning_status = False
        drift_status = False

        try:
            min_hoeffding = min(self.hoeffding_bounds)
        except:
            min_hoeffding = 1

        if min_hoeffding < self.drift_threshold:
            warning_status = True
            drift_status = True

        return warning_status, drift_status

    def needs_retrain(self):

        # Get the index of the lowest-probability mean.
        # In the case of draws, go with the older one so that
        # we have more training data.
        min_i = 0
        min_prob = np.inf
        for i, prob in enumerate(self.hoeffding_bounds[:self.n_samples]):
            if prob <= min_prob:
                min_prob = prob
                min_i = i

        # If none of the means are sufficiently improbable
        if min_prob > self.drift_threshold:
            return None

        # Otherwise return the data needed for retraining
        if min_i >= self.buffer_size:
            return self.buffer_x, self.buffer_y
        else:
            return self.buffer_x[:min_i], self.buffer_y[:min_i]

    def reset(self):
        self.RUNTIME = 0
        # Data storage
        self.buffer_x = [ None for i in range(self.window_size) ]
        self.buffer_y = [ None for i in range(self.window_size) ]

    def get_settings(self):
        pass

        # Calcalations
        self.hoeffding_bounds = []
        self.warning = False
        self.drift = False
        self.n_samples = 0
