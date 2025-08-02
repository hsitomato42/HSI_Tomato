# feature_selection/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime
import src.config as config


class FeatureSelectionVisualizer:
    """
    Comprehensive visualization and reporting system for feature selection analysis.
    
    Generates scientific-quality plots and detailed reports suitable for publication.
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations and reports
        """
        self.save_dir = save_dir or getattr(config, 'FEATURE_SELECTION_RESULTS_DIR', 'results/feature_selection')
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'reports'), exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Approximate wavelength mapping for 204 bands (400-1000 nm)
        self.wavelengths = np.linspace(400, 1000, 204)
    
    def create_comprehensive_report(
        self,
        selection_analysis: Dict[str, Any],
        attention_analysis: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a comprehensive report with all visualizations and analysis.
        
        Args:
            selection_analysis: Output from get_selected_bands_analysis()
            attention_analysis: Attention weights and analysis
            performance_metrics: Model performance metrics
            experiment_config: Experiment configuration
            
        Returns:
            report_path: Path to the generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.save_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(os.path.join(report_dir, 'figures'), exist_ok=True)
        
        # Generate all visualizations
        figure_paths = {}
        
        # 1. Band selection overview
        figure_paths['band_selection'] = self.plot_selected_bands_overview(
            selection_analysis, 
            save_path=os.path.join(report_dir, 'figures', 'band_selection_overview.png')
        )
        
        # 2. Attention heatmaps
        if attention_analysis is not None:
            figure_paths['attention_heatmap'] = self.plot_attention_heatmaps(
                attention_analysis,
                save_path=os.path.join(report_dir, 'figures', 'attention_heatmaps.png')
            )
            
            figure_paths['band_importance'] = self.plot_band_importance_analysis(
                attention_analysis,
                save_path=os.path.join(report_dir, 'figures', 'band_importance.png')
            )
        
        # 3. Spectral analysis
        figure_paths['spectral_analysis'] = self.plot_spectral_signature_analysis(
            selection_analysis,
            save_path=os.path.join(report_dir, 'figures', 'spectral_analysis.png')
        )
        
        # 4. Diversity analysis
        if attention_analysis is not None:
            figure_paths['diversity_analysis'] = self.plot_diversity_analysis(
                attention_analysis,
                selection_analysis,
                save_path=os.path.join(report_dir, 'figures', 'diversity_analysis.png')
            )
        
        # 5. Performance comparison
        if performance_metrics is not None:
            figure_paths['performance'] = self.plot_performance_comparison(
                performance_metrics,
                save_path=os.path.join(report_dir, 'figures', 'performance_comparison.png')
            )
        
        # Generate HTML report
        report_path = self._generate_html_report(
            report_dir, figure_paths, selection_analysis, 
            attention_analysis, performance_metrics, experiment_config
        )
        
        # Generate data files
        self._save_data_files(report_dir, selection_analysis, attention_analysis)
        
        print(f"Comprehensive report generated: {report_path}")
        return report_path
    
    def plot_selected_bands_overview(self, selection_analysis: Dict[str, Any], save_path: str = None) -> str:
        """
        Create an overview plot of selected bands and their importance.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        selected_indices = selection_analysis['selected_band_indices'].numpy()
        selected_wavelengths = selection_analysis['selected_wavelengths']
        gate_logits = selection_analysis['gate_logits'].numpy()
        
        # Plot 1: Selected bands on spectrum
        ax1.plot(self.wavelengths, np.zeros_like(self.wavelengths), 'k-', alpha=0.3, linewidth=1)
        for i, (idx, wl) in enumerate(zip(selected_indices, selected_wavelengths)):
            ax1.axvline(wl, color=f'C{i}', linewidth=3, alpha=0.8, label=f'Band {idx} ({wl:.1f}nm)')
        
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Selection')
        ax1.set_title('Selected Spectral Bands')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gate logits distribution
        ax2.hist(gate_logits, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        for idx in selected_indices:
            ax2.axvline(gate_logits[idx], color='red', linestyle='--', alpha=0.8)
        
        ax2.set_xlabel('Gate Logits')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Gate Logits Distribution\n(Red lines: selected bands)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Band importance vs wavelength
        importance_scores = tf.nn.softmax(gate_logits).numpy()
        ax3.plot(self.wavelengths, importance_scores, 'b-', linewidth=1, alpha=0.7)
        ax3.scatter(selected_wavelengths, importance_scores[selected_indices], 
                   c='red', s=80, marker='o', zorder=5, label='Selected')
        
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Importance Score')
        ax3.set_title('Band Importance Across Spectrum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spectral regions analysis
        # Define spectral regions
        regions = {
            'Visible (400-700nm)': (400, 700),
            'NIR (700-900nm)': (700, 900),
            'SWIR (900-1000nm)': (900, 1000)
        }
        
        region_counts = []
        region_names = []
        
        for region_name, (start, end) in regions.items():
            count = sum(1 for wl in selected_wavelengths if start <= wl <= end)
            region_counts.append(count)
            region_names.append(f'{region_name}\n({count} bands)')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax4.bar(region_names, region_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, region_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Number of Selected Bands')
        ax4.set_title('Distribution Across Spectral Regions')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_attention_heatmaps(self, attention_analysis: Dict[str, Any], save_path: str = None) -> str:
        """
        Create attention heatmap visualizations.
        """
        attention_weights = attention_analysis['attention_weights'].numpy()  # (batch, heads, bands, bands)
        
        # Average across batch and heads for visualization
        avg_attention = np.mean(attention_weights, axis=(0, 1))  # (bands, bands)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Full attention matrix
        im1 = ax1.imshow(avg_attention, cmap='viridis', aspect='auto')
        ax1.set_title('Band-to-Band Attention Matrix\n(Average across heads and batch)')
        ax1.set_xlabel('Band Index')
        ax1.set_ylabel('Band Index')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # Plot 2: Attention pattern analysis - bands focusing on others
        band_attention_out = np.sum(avg_attention, axis=1)  # How much each band attends to others
        ax2.plot(self.wavelengths, band_attention_out, 'b-', linewidth=2)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Total Outgoing Attention')
        ax2.set_title('Band Attention Patterns\n(How much each band focuses on others)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Attention received by each band
        band_attention_in = np.sum(avg_attention, axis=0)  # How much attention each band receives
        ax3.plot(self.wavelengths, band_attention_in, 'r-', linewidth=2)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Total Incoming Attention')
        ax3.set_title('Band Attention Reception\n(How much attention each band receives)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Self-attention vs cross-attention
        self_attention = np.diag(avg_attention)
        cross_attention = band_attention_out - self_attention
        
        x_pos = np.arange(len(self.wavelengths))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, self_attention, width, label='Self-attention', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, cross_attention, width, label='Cross-attention', alpha=0.8)
        
        # Only show every 20th wavelength label to avoid crowding
        step = 20
        ax4.set_xticks(x_pos[::step])
        ax4.set_xticklabels([f'{wl:.0f}' for wl in self.wavelengths[::step]], rotation=45)
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Attention Weight')
        ax4.set_title('Self vs Cross-Attention by Band')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_band_importance_analysis(self, attention_analysis: Dict[str, Any], save_path: str = None) -> str:
        """
        Create detailed band importance analysis plots.
        """
        raw_importance = attention_analysis['raw_importance_scores'].numpy()  # (batch, bands)
        band_importance = attention_analysis.get('band_importance', tf.nn.softmax(raw_importance, axis=-1)).numpy()
        
        # Average across batch
        avg_raw_importance = np.mean(raw_importance, axis=0)
        avg_importance = np.mean(band_importance, axis=0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Raw importance scores
        ax1.plot(self.wavelengths, avg_raw_importance, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(self.wavelengths, avg_raw_importance, alpha=0.3)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Raw Importance Score')
        ax1.set_title('Raw Band Importance Scores\n(Before softmax normalization)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized importance scores
        ax2.plot(self.wavelengths, avg_importance, 'r-', linewidth=2, alpha=0.8)
        ax2.fill_between(self.wavelengths, avg_importance, alpha=0.3, color='red')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Normalized Importance')
        ax2.set_title('Normalized Band Importance\n(After softmax - probability distribution)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top important bands
        top_k = 20
        top_indices = np.argsort(avg_importance)[-top_k:]
        top_wavelengths = self.wavelengths[top_indices]
        top_importances = avg_importance[top_indices]
        
        bars = ax3.barh(range(top_k), top_importances, alpha=0.8)
        ax3.set_yticks(range(top_k))
        ax3.set_yticklabels([f'{wl:.1f}nm\n(band {idx})' for wl, idx in zip(top_wavelengths, top_indices)])
        ax3.set_xlabel('Importance Score')
        ax3.set_title(f'Top {top_k} Most Important Bands')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Color bars by spectral region
        for i, (bar, wl) in enumerate(zip(bars, top_wavelengths)):
            if wl < 700:
                bar.set_color('#FF6B6B')  # Visible
            elif wl < 900:
                bar.set_color('#4ECDC4')  # NIR
            else:
                bar.set_color('#45B7D1')  # SWIR
        
        # Plot 4: Importance distribution histogram
        ax4.hist(avg_importance, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(np.mean(avg_importance), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(avg_importance):.4f}')
        ax4.axvline(np.median(avg_importance), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(avg_importance):.4f}')
        
        ax4.set_xlabel('Importance Score')
        ax4.set_ylabel('Number of Bands')
        ax4.set_title('Distribution of Band Importance Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_spectral_signature_analysis(self, selection_analysis: Dict[str, Any], save_path: str = None) -> str:
        """
        Analyze spectral signatures and clustering of selected bands.
        """
        selected_indices = selection_analysis['selected_band_indices'].numpy()
        selected_wavelengths = selection_analysis['selected_wavelengths']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Wavelength distribution
        ax1.hist(self.wavelengths, bins=50, alpha=0.5, color='lightblue', 
                label='All bands', density=True)
        ax1.hist(selected_wavelengths, bins=20, alpha=0.8, color='red', 
                label='Selected bands', density=True)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Density')
        ax1.set_title('Wavelength Distribution: All vs Selected Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Band spacing analysis
        if len(selected_wavelengths) > 1:
            sorted_wavelengths = np.sort(selected_wavelengths)
            spacings = np.diff(sorted_wavelengths)
            
            ax2.plot(sorted_wavelengths[:-1], spacings, 'bo-', markersize=8, linewidth=2)
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Spacing to Next Band (nm)')
            ax2.set_title('Inter-band Spacing Analysis\n(Diversity in wavelength selection)')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_spacing = np.mean(spacings)
            ax2.axhline(mean_spacing, color='red', linestyle='--', 
                       label=f'Mean spacing: {mean_spacing:.1f}nm')
            ax2.legend()
        
        # Plot 3: Spectral coverage analysis
        total_range = self.wavelengths.max() - self.wavelengths.min()
        selected_range = selected_wavelengths.max() - selected_wavelengths.min()
        coverage_ratio = selected_range / total_range
        
        # Create coverage visualization
        coverage_data = ['Total Spectrum', 'Selected Coverage', 'Gaps']
        coverage_values = [total_range, selected_range, total_range - selected_range]
        colors = ['lightblue', 'green', 'lightcoral']
        
        wedges, texts, autotexts = ax3.pie(coverage_values, labels=coverage_data, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Spectral Coverage Analysis\nCoverage ratio: {coverage_ratio:.2f}')
        
        # Plot 4: Band selection efficiency
        # Calculate how well distributed the selected bands are
        n_regions = 10
        region_edges = np.linspace(self.wavelengths.min(), self.wavelengths.max(), n_regions + 1)
        region_centers = (region_edges[:-1] + region_edges[1:]) / 2
        
        # Count bands in each region
        all_counts, _ = np.histogram(self.wavelengths, bins=region_edges)
        selected_counts, _ = np.histogram(selected_wavelengths, bins=region_edges)
        
        # Calculate selection efficiency (selected/available in each region)
        efficiency = np.divide(selected_counts, all_counts, 
                             out=np.zeros_like(selected_counts, dtype=float), 
                             where=all_counts!=0)
        
        bars = ax4.bar(region_centers, efficiency, width=np.diff(region_edges), 
                      alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Selection Efficiency\n(Selected/Available)')
        ax4.set_title('Band Selection Efficiency Across Spectrum')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Color bars by efficiency
        for bar, eff in zip(bars, efficiency):
            if eff > 0.1:
                bar.set_color('green')
            elif eff > 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_diversity_analysis(self, attention_analysis: Dict[str, Any], 
                              selection_analysis: Dict[str, Any], save_path: str = None) -> str:
        """
        Analyze diversity and redundancy in band selection.
        """
        diversity_scores = attention_analysis['diversity_scores'].numpy()  # (batch, bands, head_dim)
        selected_indices = selection_analysis['selected_band_indices'].numpy()
        
        # Average across batch
        avg_diversity = np.mean(diversity_scores, axis=0)  # (bands, head_dim)
        
        # Get diversity for selected bands
        selected_diversity = avg_diversity[selected_indices]  # (k_bands, head_dim)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Pairwise similarity matrix of selected bands
        # Compute similarity matrix
        selected_normalized = selected_diversity / (np.linalg.norm(selected_diversity, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(selected_normalized, selected_normalized.T)
        
        im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax1.set_title('Pairwise Similarity Matrix\n(Selected Bands)')
        ax1.set_xlabel('Selected Band Index')
        ax1.set_ylabel('Selected Band Index')
        
        # Add wavelength labels
        selected_wavelengths = selection_analysis['selected_wavelengths']
        tick_labels = [f'{wl:.0f}nm' for wl in selected_wavelengths]
        ax1.set_xticks(range(len(selected_indices)))
        ax1.set_yticks(range(len(selected_indices)))
        ax1.set_xticklabels(tick_labels, rotation=45)
        ax1.set_yticklabels(tick_labels)
        
        plt.colorbar(im1, ax=ax1, label='Similarity')
        
        # Plot 2: Diversity score distribution
        diversity_norms = np.linalg.norm(avg_diversity, axis=1)
        selected_diversity_norms = diversity_norms[selected_indices]
        
        ax2.hist(diversity_norms, bins=50, alpha=0.5, color='lightblue', 
                label='All bands', density=True)
        ax2.hist(selected_diversity_norms, bins=20, alpha=0.8, color='red', 
                label='Selected bands', density=True)
        ax2.set_xlabel('Diversity Score Magnitude')
        ax2.set_ylabel('Density')
        ax2.set_title('Diversity Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Redundancy analysis
        # Calculate redundancy as maximum similarity to any other selected band
        redundancy_scores = []
        for i in range(len(selected_indices)):
            similarities = similarity_matrix[i, :]
            # Exclude self-similarity
            other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
            max_similarity = np.max(other_similarities) if len(other_similarities) > 0 else 0
            redundancy_scores.append(max_similarity)
        
        bars = ax3.bar(range(len(selected_indices)), redundancy_scores, alpha=0.8, 
                      edgecolor='black')
        ax3.set_xticks(range(len(selected_indices)))
        ax3.set_xticklabels([f'{wl:.0f}nm' for wl in selected_wavelengths], rotation=45)
        ax3.set_xlabel('Selected Bands')
        ax3.set_ylabel('Max Similarity to Other Selected Bands')
        ax3.set_title('Band Redundancy Analysis\n(Lower is better)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Color bars by redundancy level
        for bar, redundancy in zip(bars, redundancy_scores):
            if redundancy > 0.8:
                bar.set_color('red')  # High redundancy
            elif redundancy > 0.5:
                bar.set_color('orange')  # Medium redundancy
            else:
                bar.set_color('green')  # Low redundancy
        
        # Plot 4: Diversity vs importance trade-off
        if 'band_importance' in attention_analysis:
            band_importance = attention_analysis['band_importance'].numpy()
            avg_importance = np.mean(band_importance, axis=0)
            selected_importance = avg_importance[selected_indices]
            
            scatter = ax4.scatter(selected_importance, selected_diversity_norms, 
                                c=range(len(selected_indices)), cmap='viridis', 
                                s=100, alpha=0.8, edgecolor='black')
            
            # Add labels
            for i, (imp, div, wl) in enumerate(zip(selected_importance, selected_diversity_norms, selected_wavelengths)):
                ax4.annotate(f'{wl:.0f}nm', (imp, div), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax4.set_xlabel('Importance Score')
            ax4.set_ylabel('Diversity Score')
            ax4.set_title('Importance vs Diversity Trade-off\n(Selected Bands)')
            ax4.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax4, label='Band Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_performance_comparison(self, performance_metrics: Dict[str, float], save_path: str = None) -> str:
        """
        Create performance comparison plots.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract metrics
        metrics_names = list(performance_metrics.keys())
        metrics_values = list(performance_metrics.values())
        
        # Plot 1: Overall performance comparison
        bars = ax1.bar(metrics_names, metrics_values, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Overview')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Color bars by performance level (assuming higher is better)
        for bar, value in zip(bars, metrics_values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if they're long
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2-4: Additional analysis based on available metrics
        # This would be customized based on your specific metrics
        # For now, create placeholder plots
        
        ax2.text(0.5, 0.5, 'Feature Selection\nPerformance Analysis\n\n(Customize based on\nspecific metrics)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Performance Analysis Placeholder')
        
        ax3.text(0.5, 0.5, 'Model Comparison\nBefore vs After\nFeature Selection', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax3.set_title('Comparison Analysis Placeholder')
        
        ax4.text(0.5, 0.5, 'Efficiency Metrics\n(Speed, Memory, etc.)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax4.set_title('Efficiency Analysis Placeholder')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def _generate_html_report(
        self, 
        report_dir: str, 
        figure_paths: Dict[str, str],
        selection_analysis: Dict[str, Any],
        attention_analysis: Optional[Dict[str, Any]],
        performance_metrics: Optional[Dict[str, float]],
        experiment_config: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a comprehensive HTML report.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Selection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .figure {{ text-align: center; margin: 20px 0; }}
                .figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .config {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; font-family: monospace; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Feature Selection Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Attention-Based Feature Selection for Hyperspectral Tomato Quality Prediction</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metrics">
                    <h3>Selection Results</h3>
                    <ul>
                        <li><strong>Selected Bands:</strong> {len(selection_analysis['selected_band_indices'])}</li>
                        <li><strong>Wavelength Range:</strong> {selection_analysis['selected_wavelengths'].min():.1f} - {selection_analysis['selected_wavelengths'].max():.1f} nm</li>
                        <li><strong>Spectral Coverage:</strong> {((selection_analysis['selected_wavelengths'].max() - selection_analysis['selected_wavelengths'].min()) / (1000 - 400) * 100):.1f}%</li>
                    </ul>
        """
        
        # Add performance metrics if available
        if performance_metrics:
            html_content += f"""
                    <h3>Performance Metrics</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            for metric, value in performance_metrics.items():
                html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
            html_content += "</table>"
        
        html_content += """
                </div>
            </div>
        """
        
        # Add figures
        for section_name, figure_path in figure_paths.items():
            if figure_path and os.path.exists(figure_path):
                rel_path = os.path.relpath(figure_path, report_dir)
                html_content += f"""
                <div class="section">
                    <h2>{section_name.replace('_', ' ').title()}</h2>
                    <div class="figure">
                        <img src="{rel_path}" alt="{section_name}">
                    </div>
                </div>
                """
        
        # Add detailed data
        html_content += f"""
            <div class="section">
                <h2>Detailed Analysis</h2>
                <h3>Selected Band Details</h3>
                <table>
                    <tr><th>Band Index</th><th>Wavelength (nm)</th><th>Gate Logit</th></tr>
        """
        
        selected_indices = selection_analysis['selected_band_indices'].numpy()
        selected_wavelengths = selection_analysis['selected_wavelengths']
        gate_logits = selection_analysis['gate_logits'].numpy()
        
        for idx, wl in zip(selected_indices, selected_wavelengths):
            html_content += f"<tr><td>{idx}</td><td>{wl:.1f}</td><td>{gate_logits[idx]:.4f}</td></tr>"
        
        html_content += "</table>"
        
        # Add configuration if available
        if experiment_config:
            html_content += f"""
                <h3>Experiment Configuration</h3>
                <div class="config">
                    <pre>{json.dumps(experiment_config, indent=2)}</pre>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(report_dir, 'feature_selection_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _save_data_files(
        self, 
        report_dir: str, 
        selection_analysis: Dict[str, Any],
        attention_analysis: Optional[Dict[str, Any]]
    ):
        """
        Save raw data files for further analysis.
        """
        # Save selection data
        selection_data = {
            'selected_band_indices': selection_analysis['selected_band_indices'].numpy().tolist(),
            'selected_wavelengths': selection_analysis['selected_wavelengths'].tolist(),
            'gate_logits': selection_analysis['gate_logits'].numpy().tolist(),
            'wavelengths_full': self.wavelengths.tolist()
        }
        
        with open(os.path.join(report_dir, 'selection_data.json'), 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame({
            'band_index': selection_analysis['selected_band_indices'].numpy(),
            'wavelength_nm': selection_analysis['selected_wavelengths'],
            'gate_logit': selection_analysis['gate_logits'].numpy()[selection_analysis['selected_band_indices'].numpy()]
        })
        df.to_csv(os.path.join(report_dir, 'selected_bands.csv'), index=False)
        
        # Save attention data if available
        if attention_analysis:
            attention_data = {
                'raw_importance_scores': attention_analysis['raw_importance_scores'].numpy().tolist(),
                'band_importance': attention_analysis.get('band_importance', []).numpy().tolist() if hasattr(attention_analysis.get('band_importance', []), 'numpy') else [],
                'temperature': float(attention_analysis.get('temperature', 0))
            }
            
            with open(os.path.join(report_dir, 'attention_data.json'), 'w') as f:
                json.dump(attention_data, f, indent=2)
        
        print(f"Data files saved in: {report_dir}") 