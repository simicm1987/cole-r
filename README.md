# Cole Model Estimation from Re{Z}

Browser-based tool for estimating the Cole-impedance model parameters (R0, R∞, τ, α) using only the real part of the measured impedance. Everything runs locally in the browser, no data is uploaded.

**Live demo:** https://USERNAME.github.io/cole-r/

## Features

- Estimation of R0, R∞, τ, fc and α from the resistance spectrum only
- Nyquist plot of the measured data and the estimated Cole model
- Local relative residuals of Re{Z}, 100·(fit - meas)/meas, with their mean and mean absolute value
- SSE, RMSE and estimation time
- Frequencies in Hz or kHz

## Input format

CSV file with three columns, one row per measurement point:

```
f in Hz, R in ohms, |X| n ohms
3000, 119.87, 4.33
...
```
Both "," and ";" separators are supported (";" with decimal comma).

## Usage

1. Open the page.
2. Select the frequency unit (Hz or kHz).
3. Click **Open CSV** (or press O) and choose the file.

## Citation

TBD

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
