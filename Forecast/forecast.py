import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


class DelayPredictor:
    """ARIMA-based predictor for average delay.

    - Trains an ARIMA model on monthly average delays (only once).
    - Falls back to a mean-based heuristic when insufficient data.
    """
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.mean_delay = None
        self._trained_series = None

    def train(self, df, metric="delay_minutes"):
        """
        Train the ARIMA model using historical data grouped by month.

        Expects `df["date"]` to be a datetime series. We compute monthly
        averages and fit an ARIMA(1,1,0) model when there are at least 3
        periods available. Otherwise store the mean and mark as untrained.
        """
        # Ensure month-based grouping (use year-month to avoid mixing years)
        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")
        monthly_avg = df.groupby("year_month")[metric].mean().astype(float)

        # keep a copy of series for diagnostics/plotting
        self._trained_series = monthly_avg

        if len(monthly_avg) >= 3:
            # Fit a simple ARIMA model. Order (1,1,0) is a reasonable default.
            try:
                self.model = ARIMA(monthly_avg.values, order=(1, 1, 0)).fit()
                self.mean_delay = float(np.nanmean(monthly_avg.values))
                self.is_trained = True
            except Exception:
                # If ARIMA fails for any reason, fall back to mean
                self.model = None
                self.mean_delay = float(np.nanmean(monthly_avg.values))
                self.is_trained = False
        else:
            # Not enough data to fit ARIMA; use mean as fallback
            self.model = None
            self.mean_delay = float(np.nanmean(monthly_avg.values)) if len(monthly_avg) > 0 else 0.0
            self.is_trained = False

    def predict_next_week(self):
        """
        Predict the next period (next month's average delay).

        If the ARIMA model is trained, use it to forecast one step ahead.
        Otherwise, return a slight inflation of the historical mean.
        """
        if not self.is_trained or self.model is None:
            return self.mean_delay * 1.05 if self.mean_delay is not None else 0.0

        try:
            forecast = self.model.get_forecast(steps=1)
            pred = float(forecast.predicted_mean[0])
            return pred
        except Exception:
            # On any forecasting error, return mean-based fallback
            return self.mean_delay * 1.05

    def plot_prediction(self):
        """
        Create visualization of historical monthly averages vs predicted next.
        Returns (fig, current_mean, predicted).
        """
        current = self.mean_delay if self.mean_delay is not None else 0.0
        predicted = self.predict_next_week()

        fig, ax = plt.subplots()

        # If we have historical series, plot it as a line for context
        if self._trained_series is not None and len(self._trained_series) > 0:
            # convert PeriodIndex to string labels for plotting
            labels = [str(p) for p in self._trained_series.index]
            ax.plot(labels, self._trained_series.values, marker="o", label="Historical (monthly avg)")
            ax.set_xticklabels(labels, rotation=45, ha="right")

        # Bar for current vs predicted
        ax.bar(["Current", "Predicted"], [current, predicted], color=["gray", "orange"], alpha=0.8)
        ax.set_ylabel("Average Delay (minutes)")
        ax.set_title("Predicted Average Delay for Next Month")
        ax.legend()
        plt.tight_layout()

        return fig, current, predicted