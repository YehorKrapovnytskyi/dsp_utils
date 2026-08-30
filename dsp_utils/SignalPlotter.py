import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


__all__ = ['SignalPlotter']

class SignalPlotter:
    def __init__(self, theme='seaborn-v0_8-muted', font_family='sans-serif'):
        """Initializes global styling for all plots in the session."""
        plt.style.use(theme)
        plt.rcParams.update({
            'font.family': font_family,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5
        })


    def _apply_standard_labels(self, ax, title, xlabel, ylabel):
        """Internal helper to keep labeling consistent."""
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color='black', lw=0.8, alpha=0.5) # Zero reference line


    def plot_real(self, t, signal, title="Real-Valued Signal", label="Signal"):
        """Plots a single-channel real signal."""
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        
        ax.plot(t, signal, label=label)
        self._apply_standard_labels(ax, title, "Time (s)", "Amplitude")
        
        if label:
            ax.legend(loc='upper right')
        return fig, ax


    def plot_complex(self, t, z, title="Complex Signal"):
        """
        Plots a complex signal as two subplots: 
        Real (In-phase) and Imaginary (Quadrature).
        """
        fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(10, 6), 
                                           sharex=True, layout="constrained")

        # Real Part
        ax_re.plot(t, z.real, color='C0', label=r'$\mathcal{Re}\{z\}$')
        self._apply_standard_labels(ax_re, f"{title}: Real Part", "", "Amplitude")
        ax_re.legend(loc='upper right')

        # Imaginary Part
        ax_im.plot(t, z.imag, color='C1', label=r'$\mathcal{Im}\{z\}$')
        self._apply_standard_labels(ax_im, f"{title}: Imaginary Part", "Time (s)", "Amplitude")
        ax_im.legend(loc='upper right')

        return fig, (ax_re, ax_im)


    def plot_histogram(self, data, bins=50, title="Signal Distribution", 
                       xlabel="Amplitude", density=True, fit_gaussian=False):
        """
        Plots a histogram of the signal values.
        
        Args:
            data: The signal array.
            bins: Number of histogram bins.
            title: Title of the plot.
            xlabel: Label for the x-axis (default: "Amplitude").
            density: If True, plots Probability Density instead of count.
            fit_gaussian: If True, overlays a normal distribution curve.
        """
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        
        # Flatten data in case it's a 2D array or complex
        clean_data = np.real(data).flatten()
        
        # Plot Histogram
        ax.hist(clean_data, bins=bins, density=density, 
                alpha=0.6, color='C0', edgecolor='white', 
                label='Signal PDF' if density else 'Counts')
        
        # Overlay Gaussian Fit
        if fit_gaussian:
            mu, std = norm.fit(clean_data)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r', linewidth=2, label=f'Fit ($\mu={mu:.2f}, \sigma={std:.2f}$)')
            
        # Updated to use the xlabel parameter
        y_label = "Probability Density" if density else "Count"
        self._apply_standard_labels(ax, title, xlabel, y_label)
        
        ax.legend(loc='upper right')
        
        return fig, ax