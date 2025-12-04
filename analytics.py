"""
Traffic Analytics and Visualization Module
Comprehensive analytics dashboard for road directorate traffic statistics.

Author: [Your Name]
Version: 1.0.0
License: MIT
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
except ImportError:
    raise ImportError(
        "Matplotlib and Seaborn required. Install with: pip install matplotlib seaborn"
    )

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class TrafficAnalytics:
    """Container for traffic analytics data."""
    total_vehicles: int
    vehicle_breakdown: Dict[str, int]
    frames_processed: int
    duration_seconds: float
    vehicles_per_minute: float
    peak_vehicle_type: str
    timestamp: datetime

    @classmethod
    def from_stats_file(cls, stats_path: Path) -> 'TrafficAnalytics':
        """Load analytics from statistics JSON file."""
        with open(stats_path, 'r') as f:
            data = json.load(f)

        total = data.get('total_vehicles', 0)
        breakdown = data.get('class_counts', {})
        frames = data.get('frames_processed', 0)
        duration = data.get('duration_seconds', 1)

        # Calculate metrics
        vehicles_per_min = (total / duration) * 60 if duration > 0 else 0
        peak_type = max(breakdown.items(), key=lambda x: x[1])[0] if breakdown else 'none'

        timestamp_str = data.get('start_time', datetime.now().isoformat())
        timestamp = datetime.fromisoformat(timestamp_str)

        return cls(
            total_vehicles=total,
            vehicle_breakdown=breakdown,
            frames_processed=frames,
            duration_seconds=duration,
            vehicles_per_minute=vehicles_per_min,
            peak_vehicle_type=peak_type,
            timestamp=timestamp
        )


class TrafficAnalyticsDashboard:
    """
    Comprehensive analytics dashboard for traffic statistics.
    Generates visualizations and reports for road directorate.
    """

    # Color scheme for different vehicle types
    VEHICLE_COLORS = {
        'car': '#3498db',
        'truck': '#e74c3c',
        'bus': '#2ecc71',
        'motorcycle': '#f39c12',
        'bicycle': '#9b59b6',
        'train': '#1abc9c'
    }

    def __init__(self, output_dir: Path = Path('./output/analytics')):
        """
        Initialize analytics dashboard.

        Args:
            output_dir: Directory to save analytics outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_vehicle_composition_chart(
            self,
            analytics: TrafficAnalytics,
            save_path: Optional[Path] = None
    ) -> Path:
        """
        Generate pie chart showing vehicle type composition.

        Args:
            analytics: Traffic analytics data
            save_path: Optional path to save figure

        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart
        vehicle_types = list(analytics.vehicle_breakdown.keys())
        counts = list(analytics.vehicle_breakdown.values())
        colors = [self.VEHICLE_COLORS.get(v.lower(), '#95a5a6') for v in vehicle_types]

        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=[v.capitalize() for v in vehicle_types],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )

        ax1.set_title('Vehicle Type Distribution', fontsize=14, weight='bold', pad=20)

        # Bar chart
        ax2.barh(
            [v.capitalize() for v in vehicle_types],
            counts,
            color=colors,
            edgecolor='black',
            linewidth=1.5
        )
        ax2.set_xlabel('Number of Vehicles', fontsize=12, weight='bold')
        ax2.set_title('Vehicle Count by Type', fontsize=14, weight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3)

        # Add count labels on bars
        for i, count in enumerate(counts):
            ax2.text(count + max(counts) * 0.01, i, str(count),
                     va='center', fontsize=10, weight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f'vehicle_composition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Vehicle composition chart saved: {save_path}")
        return save_path

    def generate_traffic_flow_metrics(
            self,
            analytics: TrafficAnalytics,
            save_path: Optional[Path] = None
    ) -> Path:
        """
        Generate dashboard with key traffic flow metrics.

        Args:
            analytics: Traffic analytics data
            save_path: Optional path to save figure

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Key metrics boxes
        metrics = [
            ('Total Vehicles', analytics.total_vehicles, '#3498db'),
            ('Vehicles/Minute', f'{analytics.vehicles_per_minute:.1f}', '#2ecc71'),
            ('Duration (min)', f'{analytics.duration_seconds / 60:.1f}', '#e74c3c'),
            ('Peak Type', analytics.peak_vehicle_type.capitalize(), '#f39c12'),
        ]

        for idx, (label, value, color) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, idx if idx < 3 else 0])
            ax.axis('off')

            # Create metric box
            rect = Rectangle((0.1, 0.2), 0.8, 0.6,
                             facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=3)
            ax.add_patch(rect)

            ax.text(0.5, 0.7, str(value),
                    ha='center', va='center',
                    fontsize=32, weight='bold', color=color)
            ax.text(0.5, 0.35, label,
                    ha='center', va='center',
                    fontsize=14, weight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Vehicle type breakdown (horizontal bar)
        ax_breakdown = fig.add_subplot(gs[1, :])
        vehicle_types = list(analytics.vehicle_breakdown.keys())
        counts = list(analytics.vehicle_breakdown.values())
        colors = [self.VEHICLE_COLORS.get(v.lower(), '#95a5a6') for v in vehicle_types]

        bars = ax_breakdown.barh(
            [v.capitalize() for v in vehicle_types],
            counts,
            color=colors,
            edgecolor='black',
            linewidth=2
        )

        ax_breakdown.set_xlabel('Number of Vehicles', fontsize=12, weight='bold')
        ax_breakdown.set_title('Vehicle Count Distribution', fontsize=14, weight='bold', pad=15)
        ax_breakdown.grid(axis='x', alpha=0.3)

        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            percentage = (count / total) * 100
            ax_breakdown.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                              f'{count} ({percentage:.1f}%)',
                              va='center', fontsize=10, weight='bold')

        # Traffic density indicator
        ax_density = fig.add_subplot(gs[2, :2])

        # Calculate density levels
        vpm = analytics.vehicles_per_minute
        if vpm < 5:
            density_level = 'Very Low'
            density_color = '#2ecc71'
        elif vpm < 15:
            density_level = 'Low'
            density_color = '#3498db'
        elif vpm < 30:
            density_level = 'Moderate'
            density_color = '#f39c12'
        elif vpm < 50:
            density_level = 'High'
            density_color = '#e67e22'
        else:
            density_level = 'Very High'
            density_color = '#e74c3c'

        # Density gauge
        ax_density.barh(['Traffic Density'], [vpm], color=density_color, height=0.5)
        ax_density.set_xlim(0, max(60, vpm * 1.2))
        ax_density.set_xlabel('Vehicles per Minute', fontsize=12, weight='bold')
        ax_density.set_title(f'Traffic Density Level: {density_level}',
                             fontsize=14, weight='bold', pad=15)
        ax_density.text(vpm / 2, 0, f'{vpm:.1f} veh/min',
                        ha='center', va='center',
                        fontsize=16, weight='bold', color='white')

        # Information panel
        ax_info = fig.add_subplot(gs[2, 2])
        ax_info.axis('off')

        info_text = f"""
Analysis Summary
─────────────────
Date: {analytics.timestamp.strftime('%Y-%m-%d')}
Time: {analytics.timestamp.strftime('%H:%M:%S')}

Processing Info:
• Frames: {analytics.frames_processed:,}
• Duration: {analytics.duration_seconds:.1f}s

Traffic Stats:
• Total: {analytics.total_vehicles} vehicles
• Rate: {analytics.vehicles_per_minute:.1f} veh/min
• Dominant: {analytics.peak_vehicle_type.capitalize()}
        """

        ax_info.text(0.1, 0.5, info_text,
                     fontsize=10, family='monospace',
                     verticalalignment='center')

        fig.suptitle('Traffic Flow Analysis Dashboard',
                     fontsize=18, weight='bold', y=0.98)

        if save_path is None:
            save_path = self.output_dir / f'traffic_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Traffic metrics dashboard saved: {save_path}")
        return save_path

    def generate_comparative_analysis(
            self,
            analytics_list: List[TrafficAnalytics],
            labels: Optional[List[str]] = None,
            save_path: Optional[Path] = None
    ) -> Path:
        """
        Generate comparative analysis across multiple monitoring sessions.

        Args:
            analytics_list: List of traffic analytics data
            labels: Optional labels for each session
            save_path: Optional path to save figure

        Returns:
            Path to saved figure
        """
        if labels is None:
            labels = [f'Session {i + 1}' for i in range(len(analytics_list))]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Total vehicles comparison
        ax1 = axes[0, 0]
        totals = [a.total_vehicles for a in analytics_list]
        bars = ax1.bar(labels, totals, color='#3498db', edgecolor='black', linewidth=2)
        ax1.set_ylabel('Total Vehicles', fontsize=12, weight='bold')
        ax1.set_title('Total Vehicle Count Comparison', fontsize=14, weight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(total)}',
                     ha='center', va='bottom', fontsize=10, weight='bold')

        # Vehicles per minute comparison
        ax2 = axes[0, 1]
        vpms = [a.vehicles_per_minute for a in analytics_list]
        bars = ax2.bar(labels, vpms, color='#2ecc71', edgecolor='black', linewidth=2)
        ax2.set_ylabel('Vehicles per Minute', fontsize=12, weight='bold')
        ax2.set_title('Traffic Flow Rate Comparison', fontsize=14, weight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar, vpm in zip(bars, vpms):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{vpm:.1f}',
                     ha='center', va='bottom', fontsize=10, weight='bold')

        # Vehicle type distribution (stacked bar)
        ax3 = axes[1, 0]

        all_vehicle_types = set()
        for a in analytics_list:
            all_vehicle_types.update(a.vehicle_breakdown.keys())

        vehicle_types = sorted(list(all_vehicle_types))

        bottom = np.zeros(len(analytics_list))
        for v_type in vehicle_types:
            counts = [a.vehicle_breakdown.get(v_type, 0) for a in analytics_list]
            color = self.VEHICLE_COLORS.get(v_type.lower(), '#95a5a6')
            ax3.bar(labels, counts, bottom=bottom, label=v_type.capitalize(),
                    color=color, edgecolor='black', linewidth=1)
            bottom += counts

        ax3.set_ylabel('Number of Vehicles', fontsize=12, weight='bold')
        ax3.set_title('Vehicle Type Distribution by Session', fontsize=14, weight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # Duration comparison
        ax4 = axes[1, 1]
        durations = [a.duration_seconds / 60 for a in analytics_list]
        bars = ax4.bar(labels, durations, color='#e74c3c', edgecolor='black', linewidth=2)
        ax4.set_ylabel('Duration (minutes)', fontsize=12, weight='bold')
        ax4.set_title('Monitoring Duration Comparison', fontsize=14, weight='bold')
        ax4.grid(axis='y', alpha=0.3)

        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{duration:.1f}',
                     ha='center', va='bottom', fontsize=10, weight='bold')

        plt.tight_layout()
        fig.suptitle('Multi-Session Comparative Analysis',
                     fontsize=18, weight='bold', y=1.00)

        if save_path is None:
            save_path = self.output_dir / f'comparative_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Comparative analysis saved: {save_path}")
        return save_path

    def generate_time_series_analysis(
            self,
            analytics_list: List[TrafficAnalytics],
            save_path: Optional[Path] = None
    ) -> Path:
        """
        Generate time-series analysis of traffic patterns.

        Args:
            analytics_list: List of traffic analytics sorted by timestamp
            save_path: Optional path to save figure

        Returns:
            Path to saved figure
        """
        # Sort by timestamp
        analytics_list = sorted(analytics_list, key=lambda x: x.timestamp)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        timestamps = [a.timestamp for a in analytics_list]

        # Total vehicles over time
        ax1 = axes[0]
        totals = [a.total_vehicles for a in analytics_list]
        ax1.plot(timestamps, totals, marker='o', linewidth=2,
                 markersize=8, color='#3498db', label='Total Vehicles')
        ax1.fill_between(timestamps, totals, alpha=0.3, color='#3498db')
        ax1.set_ylabel('Total Vehicles', fontsize=12, weight='bold')
        ax1.set_title('Traffic Volume Over Time', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Traffic flow rate over time
        ax2 = axes[1]
        vpms = [a.vehicles_per_minute for a in analytics_list]
        ax2.plot(timestamps, vpms, marker='s', linewidth=2,
                 markersize=8, color='#2ecc71', label='Vehicles/Minute')
        ax2.fill_between(timestamps, vpms, alpha=0.3, color='#2ecc71')
        ax2.set_ylabel('Vehicles per Minute', fontsize=12, weight='bold')
        ax2.set_title('Traffic Flow Rate Over Time', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Add density level shading
        ax2.axhspan(0, 5, alpha=0.1, color='green', label='Low')
        ax2.axhspan(5, 15, alpha=0.1, color='blue', label='Moderate')
        ax2.axhspan(15, 30, alpha=0.1, color='orange', label='High')

        # Vehicle type trends
        ax3 = axes[2]

        all_types = set()
        for a in analytics_list:
            all_types.update(a.vehicle_breakdown.keys())

        for v_type in sorted(all_types):
            counts = [a.vehicle_breakdown.get(v_type, 0) for a in analytics_list]
            color = self.VEHICLE_COLORS.get(v_type.lower(), '#95a5a6')
            ax3.plot(timestamps, counts, marker='o', linewidth=2,
                     markersize=6, color=color, label=v_type.capitalize())

        ax3.set_xlabel('Time', fontsize=12, weight='bold')
        ax3.set_ylabel('Vehicle Count', fontsize=12, weight='bold')
        ax3.set_title('Vehicle Type Trends Over Time', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, loc='best')

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        fig.suptitle('Time-Series Traffic Pattern Analysis',
                     fontsize=18, weight='bold', y=1.00)

        if save_path is None:
            save_path = self.output_dir / f'time_series_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Time-series analysis saved: {save_path}")
        return save_path

    def generate_comprehensive_report(
            self,
            analytics: TrafficAnalytics,
            save_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate all analytics visualizations for a comprehensive report.

        Args:
            analytics: Traffic analytics data
            save_dir: Optional directory to save all figures

        Returns:
            Dictionary mapping chart types to file paths
        """
        if save_dir is None:
            save_dir = self.output_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating comprehensive analytics report...")

        report_paths = {
            'composition': self.generate_vehicle_composition_chart(
                analytics,
                save_dir / 'vehicle_composition.png'
            ),
            'metrics': self.generate_traffic_flow_metrics(
                analytics,
                save_dir / 'traffic_metrics.png'
            )
        }

        self.logger.info(f"Comprehensive report generated in: {save_dir}")
        return report_paths

    def load_and_analyze(self, stats_path: Path) -> TrafficAnalytics:
        """
        Load statistics file and return analytics object.

        Args:
            stats_path: Path to statistics JSON file

        Returns:
            TrafficAnalytics object
        """
        return TrafficAnalytics.from_stats_file(stats_path)

    def batch_analyze(
            self,
            stats_dir: Path,
            pattern: str = "stats_*.json"
    ) -> List[TrafficAnalytics]:
        """
        Load and analyze multiple statistics files.

        Args:
            stats_dir: Directory containing statistics files
            pattern: Glob pattern for statistics files

        Returns:
            List of TrafficAnalytics objects
        """
        stats_files = sorted(stats_dir.glob(pattern))

        if not stats_files:
            self.logger.warning(f"No statistics files found in {stats_dir}")
            return []

        analytics_list = []
        for stats_file in stats_files:
            try:
                analytics = self.load_and_analyze(stats_file)
                analytics_list.append(analytics)
                self.logger.info(f"Loaded: {stats_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {stats_file.name}: {e}")

        return analytics_list


def main():
    """Example usage of the analytics dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize dashboard
    dashboard = TrafficAnalyticsDashboard()

    # Example: Load and analyze a single statistics file
    stats_file = Path('./output/statistics/stats_vid3_20251204_082008.json')

    if stats_file.exists():
        analytics = dashboard.load_and_analyze(stats_file)

        # Generate comprehensive report
        report_paths = dashboard.generate_comprehensive_report(analytics)

        print("\n" + "=" * 70)
        print("ANALYTICS REPORT GENERATED")
        print("=" * 70)
        for chart_type, path in report_paths.items():
            print(f"{chart_type.capitalize()}: {path}")
        print("=" * 70)

    # Example: Batch analysis and comparative charts
    stats_dir = Path('./output/statistics')
    if stats_dir.exists():
        analytics_list = dashboard.batch_analyze(stats_dir)

        if len(analytics_list) > 1:
            # Generate comparative analysis
            dashboard.generate_comparative_analysis(analytics_list)

            # Generate time-series analysis
            dashboard.generate_time_series_analysis(analytics_list)


if __name__ == "__main__":
    main()