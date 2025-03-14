import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DriverPerformanceAnalyzer:
    def __init__(self):
        self.data = self._load_data()
        self.scaler = StandardScaler()
        
    def _load_data(self):
        """Load and prepare all necessary data."""
        return {
            'circuits': pd.read_csv('data/circuits.csv'),
            'races': pd.read_csv('data/races.csv'),
            'results': pd.read_csv('data/results.csv'),
            'qualifying': pd.read_csv('data/qualifying.csv'),
            'lap_times': pd.read_csv('data/lap_times.csv'),
            'drivers': pd.read_csv('data/drivers.csv')
        }

    def analyze_lap_time_consistency(self):
        """Analyze driver lap time consistency."""
        lap_data = pd.merge(
            pd.merge(self.data['lap_times'], self.data['races'][['raceId', 'year']], on='raceId'),
            self.data['drivers'][['driverId', 'surname']], on='driverId'
        )
        
        # Calculate lap time statistics
        lap_stats = lap_data.groupby(['driverId', 'surname', 'raceId']).agg({
            'milliseconds': ['std', 'mean', 'count']
        }).reset_index()
        
        # Flatten column names
        lap_stats.columns = ['driverId', 'surname', 'raceId', 'lap_std', 'lap_mean', 'lap_count']
        
        # Calculate consistency score (lower is better)
        lap_stats['consistency_score'] = lap_stats['lap_std'] / lap_stats['lap_mean']
        
        # Get average consistency score per driver
        driver_consistency = lap_stats.groupby(['driverId', 'surname'])['consistency_score'].mean().reset_index()
        
        return driver_consistency

    def analyze_race_craft(self):
        """Analyze driver race craft including overtaking ability."""
        race_results = pd.merge(
            self.data['results'],
            self.data['drivers'][['driverId', 'surname']],
            on='driverId'
        )
        
        # Convert position columns to numeric
        race_results['grid'] = pd.to_numeric(race_results['grid'], errors='coerce')
        race_results['position'] = pd.to_numeric(race_results['position'], errors='coerce')
        
        # Calculate position changes and points per start
        race_stats = race_results.groupby(['driverId', 'surname']).agg({
            'grid': 'count',  # Number of races
            'points': ['sum', 'mean'],  # Points statistics
            'position': 'mean',  # Average finish position
        })
        
        # Flatten column names
        race_stats.columns = ['race_count', 'points_total', 'points_avg', 'avg_position']
        race_stats = race_stats.reset_index()
        
        # Calculate positions gained/lost
        valid_mask = race_results['grid'].notna() & race_results['position'].notna()
        race_results['positions_gained'] = race_results.loc[valid_mask, 'grid'] - race_results.loc[valid_mask, 'position']
        
        position_changes = race_results[valid_mask].groupby(['driverId', 'surname'])['positions_gained'].mean().reset_index()
        position_changes = position_changes.rename(columns={'positions_gained': 'avg_positions_gained'})
        
        # Merge all race craft metrics
        race_craft = pd.merge(race_stats, position_changes, on=['driverId', 'surname'])
        
        return race_craft

    def analyze_qualifying_performance(self):
        """Analyze qualifying performance."""
        qual_data = pd.merge(
            self.data['qualifying'],
            self.data['drivers'][['driverId', 'surname']],
            on='driverId'
        )
        
        # Convert qualifying times to numeric
        for col in ['q1', 'q2', 'q3']:
            qual_data[col] = pd.to_numeric(qual_data[col], errors='coerce')
        
        # Calculate qualifying statistics
        qual_stats = qual_data.groupby(['driverId', 'surname']).agg({
            'position': ['mean', 'std'],  # Average and consistency of qualifying position
            'q3': 'count',  # Number of Q3 appearances
        })
        
        # Flatten column names
        qual_stats.columns = ['qual_position_mean', 'qual_position_std', 'q3_appearances']
        qual_stats = qual_stats.reset_index()
        
        return qual_stats

    def analyze_circuit_specific_performance(self):
        """Analyze driver performance at specific circuits."""
        circuit_results = pd.merge(
            pd.merge(self.data['results'], self.data['races'][['raceId', 'circuitId']], on='raceId'),
            self.data['drivers'][['driverId', 'surname']], on='driverId'
        )
        
        # Convert numeric columns
        circuit_results['points'] = pd.to_numeric(circuit_results['points'], errors='coerce')
        circuit_results['position'] = pd.to_numeric(circuit_results['position'], errors='coerce')
        
        # Calculate performance metrics per circuit
        circuit_stats = circuit_results.groupby(['driverId', 'surname', 'circuitId']).agg({
            'points': ['mean', 'sum'],
            'position': 'mean'
        })
        
        # Flatten column names
        circuit_stats.columns = ['points_mean', 'points_total', 'avg_position']
        circuit_stats = circuit_stats.reset_index()
        
        # Calculate circuit specialization score
        circuit_variation = circuit_stats.groupby(['driverId', 'surname'])['points_mean'].agg(['std', 'mean']).reset_index()
        
        # Handle cases where mean is 0 to avoid division by zero
        circuit_variation['mean'] = circuit_variation['mean'].replace(0, np.nan)
        circuit_variation['specialization_score'] = circuit_variation['std'] / circuit_variation['mean']
        circuit_variation['specialization_score'] = circuit_variation['specialization_score'].fillna(0)
        
        return circuit_variation

    def analyze_team_impact(self):
        """Analyze impact of team performance on driver results."""
        team_results = pd.merge(
            self.data['results'],
            self.data['drivers'][['driverId', 'surname']],
            on='driverId'
        )
        
        # Calculate team-normalized performance
        team_avg = team_results.groupby(['raceId', 'constructorId'])['points'].transform('mean')
        team_results['team_normalized_points'] = team_results['points'] - team_avg
        
        team_impact = team_results.groupby(['driverId', 'surname']).agg({
            'team_normalized_points': ['mean', 'std']
        })
        
        # Flatten column names
        team_impact.columns = ['team_norm_points_mean', 'team_norm_points_std']
        team_impact = team_impact.reset_index()
        
        return team_impact

    def analyze_historical_progression(self):
        """Analyze driver performance progression over time."""
        progression_data = pd.merge(
            pd.merge(self.data['results'], self.data['races'][['raceId', 'year']], on='raceId'),
            self.data['drivers'][['driverId', 'surname']], on='driverId'
        )
        
        # Convert numeric columns
        progression_data['points'] = pd.to_numeric(progression_data['points'], errors='coerce')
        progression_data['position'] = pd.to_numeric(progression_data['position'], errors='coerce')
        progression_data['year'] = pd.to_numeric(progression_data['year'], errors='coerce')
        
        # Calculate yearly performance metrics
        yearly_stats = progression_data.groupby(['driverId', 'surname', 'year']).agg({
            'points': 'sum',
            'position': 'mean'
        }).reset_index()
        
        # Calculate progression metrics
        def calculate_trend(group):
            if len(group) < 2:  # Need at least 2 points for a trend
                return pd.Series({'performance_trend': 0, 'years_active': 1})
            
            try:
                slope = np.polyfit(group['year'], group['points'], 1)[0]
            except:
                slope = 0
                
            return pd.Series({
                'performance_trend': slope,
                'years_active': len(group['year'].unique())
            })
        
        progression = yearly_stats.groupby(['driverId', 'surname']).apply(calculate_trend).reset_index()
        
        return progression

    def calculate_overall_performance_score(self):
        """Calculate comprehensive driver performance score."""
        # Gather all metrics
        lap_consistency = self.analyze_lap_time_consistency()
        race_craft = self.analyze_race_craft()
        qualifying = self.analyze_qualifying_performance()
        circuit_spec = self.analyze_circuit_specific_performance()
        team_impact = self.analyze_team_impact()
        progression = self.analyze_historical_progression()
        
        # Merge all metrics
        performance_metrics = pd.merge(
            pd.merge(
                pd.merge(
                    pd.merge(
                        pd.merge(lap_consistency, race_craft, on=['driverId', 'surname']),
                        qualifying, on=['driverId', 'surname']
                    ),
                    circuit_spec, on=['driverId', 'surname']
                ),
                team_impact, on=['driverId', 'surname']
            ),
            progression, on=['driverId', 'surname']
        )
        
        # Select metrics for normalization
        metrics_to_normalize = [
            'consistency_score', 'avg_positions_gained', 
            'points_avg', 'avg_position',
            'performance_trend', 'specialization_score'
        ]
        
        # Normalize metrics
        normalized_metrics = self.scaler.fit_transform(performance_metrics[metrics_to_normalize])
        
        # Calculate weighted performance score
        weights = {
            'consistency': 0.2,
            'race_craft': 0.25,
            'qualifying': 0.2,
            'circuit_spec': 0.15,
            'team_impact': 0.1,
            'progression': 0.1
        }
        
        performance_metrics['overall_score'] = np.average(normalized_metrics, 
                                                        weights=list(weights.values()), 
                                                        axis=1)
        
        return performance_metrics.sort_values('overall_score', ascending=False)

    def visualize_driver_performance(self, performance_metrics):
        """Create visualizations for driver performance analysis."""
        # 1. Overall Performance Rankings
        plt.figure(figsize=(15, 8))
        sns.barplot(data=performance_metrics.head(10), 
                   x='surname', y='overall_score')
        plt.title('Top 10 Drivers by Overall Performance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('driver_overall_performance.png')
        plt.close()
        
        # 2. Performance Radar Chart for Top 5 Drivers
        top_20_drivers = performance_metrics.head(20)
        metrics = ['consistency_score', 'avg_positions_gained', 
                  'points_avg', 'performance_trend', 'specialization_score']
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        for _, driver in top_20_drivers.iterrows():
            values = [driver[m] for m in metrics]
            values = np.concatenate((values, [values[0]]))
            angles_plot = np.concatenate((angles, [angles[0]]))
            ax.plot(angles_plot, values, label=driver['surname'])
            ax.fill(angles_plot, values, alpha=0.1)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(['Consistency', 'Race Craft', 'Points', 
                           'Progression', 'Circuit Specialization'])
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Driver Performance Radar Chart')
        plt.tight_layout()
        plt.savefig('driver_performance_radar.png')
        plt.close()

def main():
    analyzer = DriverPerformanceAnalyzer()
    print("Analyzing driver performance...")
    performance_metrics = analyzer.calculate_overall_performance_score()
    analyzer.visualize_driver_performance(performance_metrics)
    
    # Save detailed results
    performance_metrics.to_csv('driver_performance_analysis.csv', index=False)
    print("\nAnalysis complete! Generated files:")
    print("1. driver_overall_performance.png - Top 10 drivers ranking")
    print("2. driver_performance_radar.png - Performance radar chart for top 5 drivers")
    print("3. driver_performance_analysis.csv - Detailed performance metrics")

if __name__ == "__main__":
    main() 