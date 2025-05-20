import pandas as pd
import numpy as np
from typing import Union, Dict, Any
import json
import os

class DataLoader:
    """Utility class for loading and processing different types of data files."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats (CSV, JSON, Excel).
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {str(e)}")
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and converting data types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        return df_processed
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict[str, Any]: Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Add basic statistics for numeric columns
        if info['numeric_columns']:
            info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
        
        return info
    
    @staticmethod
    def save_analysis_results(results: Dict[str, Any], output_path: str) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Analysis results to save
            output_path (str): Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            raise Exception(f"Error saving results to {output_path}: {str(e)}")
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the dataset for common issues.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict[str, Any]: Dictionary containing validation results
        """
        validation = {
            'has_missing_values': df.isnull().any().any(),
            'missing_value_columns': df.columns[df.isnull().any()].tolist(),
            'duplicate_rows': df.duplicated().sum(),
            'zero_variance_columns': df.columns[df.nunique() == 1].tolist(),
            'high_cardinality_columns': []
        }
        
        # Check for high cardinality in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() > df.shape[0] * 0.5:  # More than 50% unique values
                validation['high_cardinality_columns'].append(col)
        
        return validation 