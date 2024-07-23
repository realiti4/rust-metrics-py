#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use rust_metrics_py;
use ndarray::Array1;
use std::time::Instant;
// use ndarray_stats::QuantileExt;
// use linregress::linear_regression;


fn linregress_slice(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = x.len();
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_x_sq: f64 = x.iter().map(|x| x * x).sum();
    let sum_y_sq: f64 = y.iter().map(|y| y * y).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum();
    
    let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_x_sq - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n as f64;
    let r = (n as f64 * sum_xy - sum_x * sum_y) / ((n as f64 * sum_x_sq - sum_x * sum_x) * (n as f64 * sum_y_sq - sum_y * sum_y)).sqrt();
    let r_sq = r * r;
    let stderr = ((sum_y_sq - sum_y * sum_y / n as f64 - slope * slope * (sum_x_sq - sum_x * sum_x / n as f64)) / (n - 2) as f64).sqrt();
    
    return (slope, intercept, r, r_sq, stderr);
}

fn covariance(x: &[f32], y: &[f32], bias: bool) -> (f32, f32, f32) {
    let n = x.len() as f32;
    let x_mean = x.iter().sum::<f32>() / n;
    let y_mean = y.iter().sum::<f32>() / n;

    let mut ssxm = 0.0;
    let mut ssxym = 0.0;
    let mut ssym = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ssxm += dx * dx;
        ssxym += dx * dy;
        ssym += dy * dy;
    }

    let divisor = if bias { n } else { n - 1.0 };

    ssxm /= divisor;
    ssym /= divisor;
    ssxym /= divisor;

    return (ssxm, ssxym, ssym);
}

fn linregress2(x: &[f32], y: &[f32]) -> f32 {
    let tiny = 1.0e-20;

    if x.len() == 0 || y.len() == 0 {
        panic!("Input arrays are empty");
    }

    let num = x.len();
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();
    let xmean = sum_x / num as f32;
    let ymean = sum_y / num as f32;

    let (ssxm, ssxym, ssym) = covariance(&x, &y, true);

    let slope = ssxym / ssxm;

    return slope;
}

fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if window_size > data.len() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(data.len() - window_size + 1);
    let mut sum: f32 = data[..window_size].iter().sum();
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
    let returns: Vec<f32> = historical_data
        .windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
        .collect();

    // Calculate volatility (standard deviation of returns)
    let volatility = returns.iter().fold(0.0, |acc, x| acc + (x - (returns.iter().sum::<f32>() / returns.len() as f32)).powi(2));
    let volatility = (volatility / returns.len() as f32).sqrt();

    // Use the last known price instead of the average
    let last_price = *historical_data.last().unwrap_or(&0.0);
    
    // Calculate lower and upper bounds
    let lower_bound = last_price - (confidence_interval * volatility * last_price);
    let upper_bound = last_price + (confidence_interval * volatility * last_price);

    return (volatility, lower_bound, upper_bound);
}

fn calculate_ema(data: &[f32], period: usize) -> Vec<f32> {
    let alpha = 2.0 / (period as f32 + 1.0);
    let mut ema = Vec::with_capacity(data.len());
    ema.push(data[..period].iter().sum::<f32>() / period as f32);

    for price in data[period..].iter() {
        ema.push((price * alpha) + (ema.last().unwrap() * (1.0 - alpha)));
    }

    ema
}

fn trend_analysis_with_volatility(data: &[f32], lower_bound: f32, upper_bound: f32, window_size: usize) -> i32 {
    let x: Array1<f32> = Array1::range(0.0, data.len() as f32, 1.0);
    let slope = linregress2(x.as_slice().unwrap(), data);

    let mut trend = "";
    let mut action = 0;

    if data.len() >= window_size {
        let moving_avg = moving_average(&data, window_size);

        let ema = calculate_ema(data, 5);
        let recent_avg = ema.last().copied().unwrap();

        // Determine trend based on both linear regression and moving average
        if slope > 0.0 && recent_avg > upper_bound {
            trend = "Up";
            action = 1;
        } else if slope < 0.0 && recent_avg < lower_bound {
            trend = "Down";
            action = -1;
        } else {
            trend = "Neutral";
            action = 0;
            
        }
    } else {
        panic!("Data length is less than window size");
    }

    return action
}

fn test_method() {
    let context_window: Vec<f32> = vec![10.0, 2.0, 30.0, 400.0, 500.0, 600.0, 700.0];
    let prediction_window: Vec<f32> = vec![800.0, 900.0, 1000.0, 1100.0, 1200.0];

    let (volatility, lower_bound, upper_bound) = calculate_volatility_bounds(&context_window);
    let trend = trend_analysis_with_volatility(&prediction_window, lower_bound, upper_bound, 5);

    println!("Volatility: {}", volatility);
    println!("Lower Bound: {}", lower_bound);
    println!("Upper Bound: {}", upper_bound);
    println!("Trend: {}", trend);


    // let nums = 1000000;

    // let start = Instant::now();

    
    // for i in 0..nums {
    //     let my_vector: Vec<f64> = (0..720).map(|_| rand::random()).collect();
    //     let my_array: [f64; 720] = my_vector.try_into().unwrap();
    //     // let my_array: [f64; 32] = rand::random();

    //     let (slope, intercept, r, r_sq, stderr) = linregress_slice(&my_array, &my_array);

    //     // assert!(slope == 137.8);
    // }

    // let duration = start.elapsed();
    // println!("Elapsed time: {:.2?}", duration);
    // println!("Elapsed time: {:?}", duration);

    // let start = Instant::now();

    
    // for i in 0..nums {
    //     let my_vector: Vec<f32> = (0..720).map(|_| rand::random()).collect();
    //     let my_array: [f32; 720] = my_vector.try_into().unwrap();
    //     // let my_array = my_array.map(|x| x as f32);

        
    //     let slope = linregress2(&my_array, &my_array);

    //     // assert!(slope == 137.8);
    // }

    // let duration = start.elapsed();
    // println!("Elapsed time: {:.2?}", duration);
}

fn main() {
    println!("Hello, world!");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    // let (slope, intercept, r, r_sq, stderr) = linregress(x, y);

    // println!("Slope: {}", slope);
    // println!("Intercept: {}", intercept);
    // println!("R: {}", r);
    // println!("R-squared: {}", r_sq);
    // println!("Standard Error: {}", stderr);

    test_method();

    println!("Done");
}