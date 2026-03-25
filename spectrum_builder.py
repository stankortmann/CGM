"""
Spectrum Builder Module
=======================

A comprehensive module for building mock quasar absorption spectra from cosmological
simulations using the specwizard package. This module provides a SpectrumBuilder class
that handles configuration management, spectrum generation, and output handling.

Configuration:
    Settings are loaded from YAML configuration files located in:
    configurations/specwizard/template/template.yaml

Usage:
    >>> builder = SpectrumBuilder(config_file="configurations/specwizard/template/template.yaml")
    >>> spectra = builder.generate_spectra()
    >>> builder.save_results(output_dir="results/")
"""

import yaml
import os
import numpy as np
from pathlib import Path
import specwizard
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpectrumBuilder:
    """
    Main class for building mock quasar absorption spectra.
    
    This class manages the entire workflow: loading configuration, setting up
    specwizard parameters, generating spectra, and saving results.
    
    Attributes:
        config (dict): Configuration parameters loaded from YAML file
        wizard (dict): Specwizard configuration dictionary
        spectra (dict): Generated spectra output
        optical_depth (dict): Computed optical depth
        projected_los (dict): Projected line-of-sight data
    """
    
    def __init__(self, config_file="configurations/specwizard/template/template.yaml"):
        """
        Initialize the SpectrumBuilder with configuration.
        
        Args:
            config_file (str): Path to the YAML configuration file.
                              Defaults to the template configuration.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        self.config_file = config_file
        self.config = None
        self.wizard = None
        self.spectra = None
        self.optical_depth = None
        self.projected_los = None
        self.data = None
        
        logger.info(f"Initializing SpectrumBuilder with config: {config_file}")
        self._load_config()
        self._setup_wizard()
    
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        The configuration file should contain the following sections:
        - file_type: Simulation type and snapshot file type
        - snapshot_params: Path to simulation snapshot files
        - sightline: Line-of-sight projection parameters
        - ionparams: Ionization parameter settings
        - ODParams: Optical depth computation parameters
        - LongSpectra: Long spectrum wavelength binning parameters
        
        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_file}")
            logger.debug(f"Configuration keys: {self.config.keys()}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def _setup_wizard(self):
        """
        Set up the specwizard configuration dictionary.
        
        This method converts the loaded configuration into the format
        expected by specwizard functions.
        """
        self.wizard = {
            'file_type': self.config.get('file_type', {}),
            'snapshot_params': self.config.get('snapshot_params', {}),
            'sightline': self.config.get('sightline', {}),
            'ionparams': self.config.get('ionparams', {}),
            'ODParams': self.config.get('ODParams', {}),
            'extra_parameters': self.config.get('extra_parameters', {}),
            'LongSpectra': self.config.get('LongSpectra', {}),
        }
        logger.info("Specwizard configuration setup complete")
    
    def generate_spectra(self):
        """
        Generate short spectra using specwizard.
        
        This is the main pipeline that:
        1. Reads particle data from the simulation snapshot
        2. Projects particles along line-of-sight
        3. Computes optical depth along each sightline
        4. Stores all intermediate results
        
        Returns:
            dict: Optical depth data containing computed spectra
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            logger.info("Starting spectrum generation pipeline...")
            
            # Use specwizard's built-in function to generate short spectra
            self.optical_depth, self.projected_los, self.data = \
                specwizard.GenerateShortSpectra(Wizard=self.wizard)
            
            logger.info("Spectrum generation completed successfully")
            logger.info(f"Number of sightlines: {len(self.optical_depth.get('HI', []))}")
            
            return self.optical_depth
            
        except Exception as e:
            logger.error(f"Error generating spectra: {e}")
            raise
    
    def get_optical_depth(self, ion_species='HI'):
        """
        Retrieve optical depth for a specific ion species.
        
        Args:
            ion_species (str): Ion species to retrieve (e.g., 'HI', 'HeII', 'CIV')
                              Default is 'HI' (neutral hydrogen)
        
        Returns:
            np.ndarray: Optical depth array for the specified species
            
        Raises:
            KeyError: If ion species not found in computed optical depth
            ValueError: If spectra not yet generated
        """
        if self.optical_depth is None:
            raise ValueError("Spectra not yet generated. Call generate_spectra() first.")
        
        try:
            return self.optical_depth.get(ion_species)
        except KeyError:
            available_ions = [k for k in self.optical_depth.keys() if k != 'Methods']
            logger.warning(f"Ion species '{ion_species}' not found. Available: {available_ions}")
            raise
    
    def get_wavelength_array(self):
        """
        Generate the wavelength array for the spectra.
        
        Returns:
            np.ndarray: Wavelength array in Angstroms
        """
        long_spectra_params = self.config.get('LongSpectra', {})
        lambda_min = long_spectra_params.get('lambda_min', 300.0)
        lambda_max = long_spectra_params.get('lambda_max', 8000.0)
        dlambda = long_spectra_params.get('dlambda', 0.5)
        
        wavelength = np.arange(lambda_min, lambda_max + dlambda, dlambda)
        logger.info(f"Wavelength array: {lambda_min} - {lambda_max} Å, "
                   f"resolution: {dlambda} Å, N_pixels: {len(wavelength)}")
        
        return wavelength
    
    def compute_transmission(self, ion_species='HI'):
        """
        Compute transmission = exp(-tau) from optical depth.
        
        Args:
            ion_species (str): Ion species to compute transmission for
        
        Returns:
            np.ndarray: Transmission (F = exp(-tau))
        """
        tau = self.get_optical_depth(ion_species)
        if tau is None:
            raise ValueError(f"No optical depth data for {ion_species}")
        
        transmission = np.exp(-np.array(tau))
        logger.info(f"Transmission computed for {ion_species}")
        return transmission
    
    def save_results(self, output_dir="results/", prefix="spectrum"):
        """
        Save the generated spectra and intermediate results to disk.
        
        This creates the output directory if it doesn't exist and saves:
        - Optical depth data
        - Wavelength information
        - Configuration used
        
        Args:
            output_dir (str): Directory to save results. Default: "results/"
            prefix (str): Prefix for output filenames. Default: "spectrum"
        
        Returns:
            str: Path to the output directory
        """
        if self.optical_depth is None:
            raise ValueError("No spectra generated yet. Call generate_spectra() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save optical depth
            np.save(output_path / f"{prefix}_optical_depth.npy", 
                   self.optical_depth, allow_pickle=True)
            logger.info(f"Optical depth saved to {output_path / f'{prefix}_optical_depth.npy'}")
            
            # Save wavelength array
            wavelength = self.get_wavelength_array()
            np.save(output_path / f"{prefix}_wavelength.npy", wavelength)
            logger.info(f"Wavelength array saved to {output_path / f'{prefix}_wavelength.npy'}")
            
            # Save configuration used
            with open(output_path / f"{prefix}_config.yaml", 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {output_path / f'{prefix}_config.yaml'}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def print_config(self):
        """
        Pretty-print the current configuration for verification.
        """
        print("\n" + "="*70)
        print("SPECTRUM BUILDER CONFIGURATION")
        print("="*70)
        
        for section, params in self.config.items():
            print(f"\n{section}:")
            if isinstance(params, dict):
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {params}")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# COMMAND-LINE USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the SpectrumBuilder class.
    
    This demonstrates the typical workflow:
    1. Create a builder instance with configuration
    2. Print configuration for verification
    3. Generate spectra
    4. Extract results
    5. Save to disk
    """
    
    # Initialize the builder
    builder = SpectrumBuilder(
        config_file="configurations/specwizard/template/extra_param.yaml"
    )
    
    # Display the configuration being used
    builder.print_config()
    
    # Generate spectra
    try:
        optical_depth = builder.generate_spectra()
        logger.info("Spectra generation successful!")
        
        # Get wavelength array
        wavelength = builder.get_wavelength_array()
        logger.info(f"Generated wavelength array with {len(wavelength)} pixels")
        
        # Compute transmission for neutral hydrogen
        transmission_hi = builder.compute_transmission('HI')
        logger.info(f"Transmission computed. Shape: {transmission_hi.shape}")
        
        # Save results
        output_dir = builder.save_results(output_dir="results/spectra/")
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in spectrum generation: {e}")
        raise