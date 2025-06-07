#!/usr/bin/env python3
"""
UTI Prediction Model for Nursing Homes
Logistic Regression Implementation

This module implements a predictive analytics system for identifying
high-risk nursing home facilities for urinary tract infections (UTIs).

Author: Healthcare Analytics Team
Date: June 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UTIPredictionModel:
    """
    UTI Prediction Model for Nursing Home Facilities
    
    This class implements a logistic regression model to predict UTI outbreaks
    based on facility characteristics and risk factors.
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            'catheter_rate': 30.0,  # ≥30%
            'staff_ratio': 7.0,     # ≥7:1
            'avg_length_stay': 12.0, # ≥12 days
            'high_risk_population': 40.0, # ≥40%
            'overall_risk_score': 70.0    # ≥70%
        }
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=200):
        """
        Generate sample nursing home data for model training and testing
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pandas.DataFrame: Generated dataset
        """
        np.random.seed(42)
        
        # Generate facility characteristics
        data = {
            'facility_id': [f'F{i:03d}' for i in range(1, n_samples + 1)],
            'catheter_rate': np.random.uniform(15, 50, n_samples),  # 15-50%
            'staff_ratio': np.random.uniform(4.5, 12.0, n_samples),  # 4.5:1 to 12:1
            'avg_length_stay': np.random.uniform(6, 20, n_samples),  # 6-20 days
            'facility_size': np.random.randint(50, 200, n_samples),  # 50-200 residents
            'black_patient_pct': np.random.uniform(10, 70, n_samples),  # 10-70%
        }
        
        df = pd.DataFrame(data)
        
        # Generate UTI outbreak target variable based on risk factors
        # Higher risk = higher probability of outbreak
        risk_score = (
            (df['catheter_rate'] / 50) * 0.4 +
            (df['staff_ratio'] / 12) * 0.3 +
            (df['avg_length_stay'] / 20) * 0.2 +
            (df['black_patient_pct'] / 70) * 0.1
        )
        
        # Add some noise and convert to binary outcome
        outbreak_prob = 1 / (1 + np.exp(-(3 * risk_score - 1.5 + np.random.normal(0, 0.3, n_samples))))
        df['uti_outbreak'] = (outbreak_prob > 0.5).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training/prediction
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            numpy.array: Prepared feature matrix
        """
        feature_columns = ['catheter_rate', 'staff_ratio', 'avg_length_stay', 
                          'facility_size', 'black_patient_pct']
        return df[feature_columns].values
    
    def train_model(self, df):
        """
        Train the logistic regression model
        
        Args:
            df (pandas.DataFrame): Training dataset
        """
        X = self.prepare_features(df)
        y = df['uti_outbreak'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("Model training completed successfully!")
        
    def predict_risk(self, df):
        """
        Predict UTI outbreak risk for facilities
        
        Args:
            df (pandas.DataFrame): Facility data
            
        Returns:
            numpy.array: Risk probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def classify_risk_level(self, risk_prob):
        """
        Classify facilities into risk levels
        
        Args:
            risk_prob (float): Risk probability (0-1)
            
        Returns:
            str: Risk level ('Low', 'Medium', 'High')
        """
        if risk_prob >= 0.7:
            return 'High'
        elif risk_prob >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def evaluate_model(self):
        """
        Evaluate model performance on test data
        
        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'sensitivity': recall_score(self.y_test, y_pred),
            'specificity': precision_score(self.y_test, 1 - y_pred)
        }
        
        return metrics
    
    def get_facility_assessment(self, df):
        """
        Generate comprehensive facility risk assessment
        
        Args:
            df (pandas.DataFrame): Facility data
            
        Returns:
            pandas.DataFrame: Assessment results
        """
        # Get risk predictions
        risk_probs = self.predict_risk(df)
        
        # Create assessment dataframe
        assessment = df.copy()
        assessment['risk_probability'] = risk_probs * 100  # Convert to percentage
        assessment['risk_level'] = [self.classify_risk_level(prob) for prob in risk_probs]
        
        # Check individual risk factors
        assessment['catheter_flag'] = assessment['catheter_rate'] >= self.risk_thresholds['catheter_rate']
        assessment['staff_flag'] = assessment['staff_ratio'] >= self.risk_thresholds['staff_ratio']
        assessment['stay_flag'] = assessment['avg_length_stay'] >= self.risk_thresholds['avg_length_stay']
        assessment['population_flag'] = assessment['black_patient_pct'] >= self.risk_thresholds['high_risk_population']
        
        # Count risk factors
        assessment['risk_factors_count'] = (
            assessment['catheter_flag'].astype(int) +
            assessment['staff_flag'].astype(int) +
            assessment['stay_flag'].astype(int) +
            assessment['population_flag'].astype(int)
        )
        
        return assessment
    
    def generate_alerts(self, assessment_df):
        """
        Generate intervention alerts for high-risk facilities
        
        Args:
            assessment_df (pandas.DataFrame): Facility assessment results
            
        Returns:
            list: Alert messages
        """
        alerts = []
        
        high_risk_facilities = assessment_df[assessment_df['risk_level'] == 'High']
        
        for _, facility in high_risk_facilities.iterrows():
            alert = {
                'facility_id': facility['facility_id'],
                'risk_score': f"{facility['risk_probability']:.1f}%",
                'priority': 'IMMEDIATE',
                'action': 'Schedule catheter cleaning within 24 hours',
                'risk_factors': []
            }
            
            if facility['catheter_flag']:
                alert['risk_factors'].append(f"High catheter rate ({facility['catheter_rate']:.1f}%)")
            if facility['staff_flag']:
                alert['risk_factors'].append(f"High staff ratio ({facility['staff_ratio']:.1f}:1)")
            if facility['stay_flag']:
                alert['risk_factors'].append(f"Extended stays ({facility['avg_length_stay']:.1f} days)")
            if facility['population_flag']:
                alert['risk_factors'].append(f"High-risk population ({facility['black_patient_pct']:.1f}%)")
                
            alerts.append(alert)
        
        return alerts
    
    def print_model_summary(self):
        """Print model summary and performance metrics"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return
            
        print("\n" + "="*60)
        print("UTI PREDICTION MODEL SUMMARY")
        print("="*60)
        
        # Model coefficients
        feature_names = ['Catheter Rate', 'Staff Ratio', 'Avg Length Stay', 
                        'Facility Size', 'Black Patient %']
        coefficients = self.model.coef_[0]
        
        print("\nModel Coefficients:")
        for name, coef in zip(feature_names, coefficients):
            print(f"  {name}: {coef:.3f}")
        print(f"  Intercept: {self.model.intercept_[0]:.3f}")
        
        # Risk thresholds
        print("\nRisk Factor Thresholds:")
        for factor, threshold in self.risk_thresholds.items():
            if factor != 'overall_risk_score':
                print(f"  {factor.replace('_', ' ').title()}: ≥{threshold}")
        
        # Performance metrics
        metrics = self.evaluate_model()
        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)")
        print(f"  Specificity: {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%)")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        
        print("\n" + "="*60)

def simulate_intervention_study():
    """
    Simulate the intervention study results from 2023
    
    Returns:
        pandas.DataFrame: Study results
    """
    # Real facility data from dashboard (2023 study)
    facilities_data = {
        'facility_id': ['Facility 1', 'Facility 2', 'Facility 3', 'Facility 4', 'Facility 5'],
        'catheter_rate': [35, 28, 42, 19, 33],
        'staff_ratio': [8.2, 6.5, 9.1, 5.2, 7.8],
        'avg_length_stay': [14.5, 11.2, 16.8, 8.5, 13.1],
        'facility_size': [120, 95, 150, 80, 110],
        'black_patient_pct': [45, 22, 58, 15, 38],
        'predicted_risk': [82, 55, 89, 23, 71],
        'risk_level': ['High', 'Medium', 'High', 'Low', 'High'],
        'pre_intervention_utis': [18, 8, 24, 3, 15],
        'post_intervention_utis': [13, 6, 17, 2, 9],
        'intervention_date': ['2023-11-15', '2023-11-20', '2023-11-10', None, '2023-11-18']
    }
    
    df = pd.DataFrame(facilities_data)
    
    # Calculate reduction rates
    df['uti_reduction'] = df['pre_intervention_utis'] - df['post_intervention_utis']
    df['reduction_rate'] = (df['uti_reduction'] / df['pre_intervention_utis'] * 100).round(1)
    
    return df

def main():
    """Main execution function"""
    print("UTI Prediction Model for Nursing Homes")
    print("=====================================")
    
    # Initialize model
    model = UTIPredictionModel()
    
    # Generate training data
    print("\n1. Generating sample training data...")
    training_data = model.generate_sample_data(200)
    print(f"Generated {len(training_data)} facility records for training")
    
    # Train model
    print("\n2. Training logistic regression model...")
    model.train_model(training_data)
    
    # Print model summary
    model.print_model_summary()
    
    # Load 2023 study data
    print("\n3. Analyzing 2023 Intervention Study Results...")
    study_data = simulate_intervention_study()
    
    print("\nStudy Results Summary:")
    print("-" * 40)
    for _, facility in study_data.iterrows():
        if facility['intervention_date']:
            print(f"{facility['facility_id']}: {facility['reduction_rate']:.1f}% reduction "
                  f"({facility['pre_intervention_utis']} → {facility['post_intervention_utis']} UTIs)")
    
    # Calculate overall impact
    intervention_facilities = study_data[study_data['intervention_date'].notna()]
    avg_reduction = intervention_facilities['reduction_rate'].mean()
    total_prevented = intervention_facilities['uti_reduction'].sum()
    
    print(f"\nOverall Impact:")
    print(f"  Average UTI Reduction: {avg_reduction:.1f}%")
    print(f"  Total UTIs Prevented: {total_prevented}")
    print(f"  Study Period: July 2023 - February 2024")
    
    # Generate new facility assessments
    print("\n4. Generating Risk Assessment for New Facilities...")
    new_facilities = model.generate_sample_data(5)
    assessment = model.get_facility_assessment(new_facilities)
    
    print("\nNew Facility Risk Assessment:")
    print("-" * 60)
    for _, facility in assessment.iterrows():
        print(f"{facility['facility_id']}: {facility['risk_probability']:.1f}% risk "
              f"({facility['risk_level']} Risk) - {facility['risk_factors_count']} risk factors")
    
    # Generate alerts
    alerts = model.generate_alerts(assessment)
    if alerts:
        print(f"\n5. High-Risk Alerts Generated:")
        print("-" * 40)
        for alert in alerts:
            print(f"🚨 {alert['facility_id']}: {alert['risk_score']} risk")
            print(f"   Action: {alert['action']}")
            print(f"   Risk Factors: {', '.join(alert['risk_factors'])}")
            print()
    
    print("\nModel analysis complete!")
    print("\nTo use this model in production:")
    print("1. Replace sample data with real facility data")
    print("2. Retrain model with historical UTI outbreak data") 
    print("3. Integrate with facility management systems")
    print("4. Set up automated daily risk scoring")
    print("5. Implement intervention protocols for high-risk alerts")

if __name__ == "__main__":
    main()
