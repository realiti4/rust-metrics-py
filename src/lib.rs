use pyo3::prelude::*;
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use rayon::prelude::*;


#[pyfunction]
fn analyze_trend(context_window: PyReadonlyArray1<f32>, prediction_window: PyReadonlyArray1<f32>) -> PyResult<i32> {
    let context_window = context_window.as_slice()?;
    let prediction_window = prediction_window.as_slice()?;

    // Calculate volatility bounds using the context window
    let (_, lower_bound, upper_bound) = calculate_volatility_bounds(&context_window);

    // Use the combined data for trend analysis
    let window_size = 5; // You can adjust this value as needed
    let action = trend_analysis_with_volatility(&prediction_window, lower_bound, upper_bound, window_size);

    Ok(action)
}

fn covariance(x: &[f32], y: &[f32], bias: bool) -> (f32, f32, f32) {
    let n = x.len() as f32;
    let x_mean = x.par_iter().sum::<f32>() / n;
    let y_mean = y.par_iter().sum::<f32>() / n;

    let (ssxm, ssxym, ssym) = x.par_iter().zip(y)
        .fold(|| (0.0, 0.0, 0.0), |(mut ssxm, mut ssxym, mut ssym), (&xi, &yi)| {
            let dx = xi - x_mean;
            let dy = yi - y_mean;
            ssxm += dx * dx;
            ssxym += dx * dy;
            ssym += dy * dy;
            (ssxm, ssxym, ssym)
        })
        .reduce(|| (0.0, 0.0, 0.0), |(ssxm1, ssxym1, ssym1), (ssxm2, ssxym2, ssym2)| {
            (ssxm1 + ssxm2, ssxym1 + ssxym2, ssym1 + ssym2)
        });

    let divisor = if bias { n } else { n - 1.0 };

    (ssxm / divisor, ssxym / divisor, ssym / divisor)
}

fn linregress2(x: &[f32], y: &[f32]) -> f32 {
    if x.is_empty() || y.is_empty() {
        panic!("Input arrays are empty");
    }

    let (ssxm, ssxym, _) = covariance(&x, &y, true);
    ssxym / ssxm
}

fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size > data.len() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len() - window_size + 1);
    let mut sum: f32 = data[..window_size].par_iter().sum();
    result.push(sum / window_size as f32);

    for i in window_size..data.len() {
        sum += data[i] - data[i - window_size];
        result.push(sum / window_size as f32);
    }

    result
}

fn calculate_volatility_bounds(historical_data: &[f32]) -> (f32, f32, f32) {
    let confidence_interval = 1.96;

    // Calculate daily returns
    let returns: Vec<f32> = historical_data.windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
        .collect();

    // Calculate volatility (standard deviation of returns)
    let mean = returns.par_iter().sum::<f32>() / returns.len() as f32;
    let volatility = returns.par_iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f32>() / returns.len() as f32;
    let volatility = volatility.sqrt();

    // Use the last known price instead of the average
    let last_price = *historical_data.last().unwrap_or(&0.0);

    // Calculate lower and upper bounds
    let lower_bound = last_price - (confidence_interval * volatility * last_price);
    let upper_bound = last_price + (confidence_interval * volatility * last_price);

    (volatility, lower_bound, upper_bound)
}

fn calculate_ema(data: &[f32], period: usize) -> Vec<f32> {
    let alpha = 2.0 / (period as f32 + 1.0);
    let mut ema = Vec::with_capacity(data.len());
    ema.push(data[..period].par_iter().sum::<f32>() / period as f32);

    for &price in data[period..].iter() {
        ema.push((price * alpha) + (ema.last().unwrap() * (1.0 - alpha)));
    }

    ema
}

fn trend_analysis_with_volatility(data: &[f32], lower_bound: f32, upper_bound: f32, window_size: usize) -> i32 {
    let x: Array1<f32> = Array1::range(0.0, data.len() as f32, 1.0);
    let slope = linregress2(x.as_slice().unwrap(), data);

    let mut action = 0;

    let ema = calculate_ema(data, 5);
    let recent_avg = ema.last().copied().unwrap();

    // Determine trend based on both linear regression and moving average
    if slope > 0.0 && recent_avg > upper_bound {
        action = 1;
    } else if slope < 0.0 && recent_avg < lower_bound {
        action = -1;
    } else {
        action = 0;
    }
    if data.len() >= window_size {
    } else {
        panic!("Data length is less than window size");
    }

    action
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_metrics_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_trend, m)?)?;
    Ok(())
}
