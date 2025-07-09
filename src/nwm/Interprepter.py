import warnings
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class LogTransformer:
    @staticmethod
    def transform(x, y):
        if np.any(y <= 0):
            warnings.warn(
                "Log transformation input y <= 0, setting such y values to a small positive number"
            )
            y = np.maximum(y, 1e-10)
        return x, np.log(y)

    @staticmethod
    def inverse_transform(y):
        return np.exp(y)


class CubicSplineMethod:
    @staticmethod
    def interpolate(x, y, x_new):
        spline_interp = CubicSpline(x, y, bc_type="not-a-knot")
        return spline_interp(x_new)


class LinearMethod:
    @staticmethod
    def interpolate(x, y, x_new):
        linear_interp = interp1d(x, y, kind="linear", fill_value="extrapolate")
        return linear_interp(x_new)


class BaseInterpolator:
    def __init__(self, method, x, y, use_log=False):
        if np.all(np.diff(x) < 0):
            x = np.flip(x)
            y = np.flip(y)

        self.x = x
        self.y = y
        self.method = method
        self.use_log = use_log

        if self.use_log:
            self.transformer = LogTransformer()
            self.x, self.y = self.transformer.transform(self.x, self.y)

    def interpolate(self, x_new):
        y_new = self.method.interpolate(self.x, self.y, x_new)
        if self.use_log:
            y_new = self.transformer.inverse_transform(y_new)
        return y_new


class Interpreter:
    def __init__(self, method, x, y, use_log=False):
        # x=np.asarray(x)
        # y=np.asarray(y)
        self.interpolator = BaseInterpolator(method, x, y, use_log)

    def interpolate(self, x_new):
        return self.interpolator.interpolate(x_new)


class LogCubicSplineInterpolator(Interpreter):
    def __init__(self, x, y):
        super().__init__(CubicSplineMethod(), x, y, use_log=True)


class CubicSplineInterpolator(Interpreter):
    def __init__(self, x, y):
        super().__init__(CubicSplineMethod(), x, y)


class LogLinearInterpolator(Interpreter):
    def __init__(self, x, y):
        super().__init__(LinearMethod(), x, y, use_log=True)


class LinearInterpolator(Interpreter):
    def __init__(self, x, y):
        super().__init__(LinearMethod(), x, y)
