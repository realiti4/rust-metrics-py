import time
import numpy as np

from scipy.stats import linregress
from rust_metrics_py import analyze_trend, analyze_trend_rust
from pybind_app import analyze_trend as analyze_trend_pybind


def calculate_volatility_bounds(historical_data, confidence_interval=1.96):
    # Calculate daily returns
    returns = np.diff(historical_data) / historical_data[:-1]

    # Calculate volatility (standard deviation of returns)
    volatility = np.std(returns)

    # Use the last known price instead of the average
    last_price = historical_data[-1]

    # Calculate lower and upper bounds
    lower_bound = last_price - (confidence_interval * volatility * last_price)
    upper_bound = last_price + (confidence_interval * volatility * last_price)

    return volatility, lower_bound, upper_bound


def calculate_ema(data: np.ndarray, period: int = 5) -> np.ndarray:
    alpha = 2 / (period + 1)
    ema = [sum(data[:period]) / period]

    for price in data[period:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))

    return np.array(ema)


def trend_analysis_with_volatility(
    data: np.ndarray, lower_bound: float, upper_bound: float, window_size=5
):
    """
    Perform trend analysis on the given data, incorporating volatility measures.

    This function uses linear regression and moving averages to determine the trend
    of the input data. It also calculates an exponential moving average (EMA) for
    recent average calculation.

    Parameters:
    -----------
    data : np.ndarray
        The input data array for trend analysis.
    lower_bound : float
        The lower threshold for determining a downward trend.
    upper_bound : float
        The upper threshold for determining an upward trend.
    window_size : int, optional
        The size of the window for calculating moving averages (default is 5).
    """

    # Linear regression
    x = np.arange(len(data))
    slope, _, _, _, _ = linregress(x, data)

    # Moving average
    if len(data) >= window_size:
        # moving_avg = np.convolve(data, np.ones(window_size), "valid") / window_size
        # oldest_avg = moving_avg[0]
        # recent_avg = moving_avg[-1]
        # overall_avg = np.mean(data)

        ema = calculate_ema(data, period=5)
        recent_avg = ema[-1]

        # # Convert this to logger
        # print("Oldest avg: ", oldest_avg, " Recent avg: ", recent_avg, " Overall avg: ", overall_avg)

        # Determine trend based on both linear regression and moving average
        if slope > 0 and recent_avg > upper_bound:
            trend = "Up"
            action = 1
        elif slope < 0 and recent_avg < lower_bound:
            trend = "Down"
            action = -1
        else:
            trend = "Neutral"
            action = 0
    else:
        trend = "Insufficient data"

    return trend, action


def generate_timeseries(
    length: int = 1024, trend: float = 1.0, noise: float = 0.2
) -> np.ndarray:
    """
    Generate a random time series of given length.

    Returns:
        np.ndarray: Random time series.
    """

    # Generate a small upward trend
    trend = np.linspace(0, trend, length)

    # Generate a sine wave
    frequency = 5  # Adjust frequency as needed
    sine_wave = np.sin(np.linspace(0, 2 * np.pi * frequency, length))

    # Generate random noise
    noise = np.random.normal(0, noise, length)

    # Combine the trend and the noise
    time_series = trend + sine_wave + noise

    return time_series


def main():
    np.random.seed(44)

    context_length = 720
    length = context_length + 196

    window = generate_timeseries(length=1024)
    window = window.astype(np.float32)

    forecast_index = range(context_length, length)

    context_window = window[:context_length]
    prediction_window = window[forecast_index]

    num = 100000

    # # Python
    # start = time.time()

    # for i in range(num):
    #     volatility, lower_bound, upper_bound = calculate_volatility_bounds(
    #         context_window
    #     )

    #     trend, action = trend_analysis_with_volatility(
    #         prediction_window, lower_bound, upper_bound, window_size=5
    #     )

    # end = time.time()
    # print(f"Python Duration {(end - start):.2f} seconds")

    # Rust
    start = time.time()

    for i in range(num):
        action = analyze_trend(context_window, prediction_window)
        # action_rust = analyze_trend_rust(context_window, prediction_window)

    end = time.time()
    print(f"Rust Duration {(end - start):.2f} seconds")

    # C++
    start = time.time()

    for i in range(num):
        # action_rust = analyze_trend_rust(context_window, prediction_window)
        action_cpp = analyze_trend_pybind(context_window, prediction_window)

        # print("here")

    end = time.time()
    print(f"C++ Duration {(end - start):.2f} seconds")

    # print("Trend: ", trend)


main()
