"""
Credit Risk Management System
Real-time credit risk scoring with PD/LGD/EAD models, portfolio analysis, stress testing, and regulatory reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import boto3
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskRating(Enum):
    AAA = 1
    AA = 2
    A = 3
    BBB = 4
    BB = 5
    B = 6
    CCC = 7
    CC = 8
    C = 9
    D = 10

@dataclass
class CreditApplication:
    """Credit application data structure"""
    application_id: str
    customer_id: str
    loan_amount: float
    loan_term: int
    annual_income: float
    debt_to_income: float
    credit_score: int
    employment_years: float
    loan_purpose: str
    property_value: Optional[float] = None
    down_payment: Optional[float] = None

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    pd: float  # Probability of Default
    lgd: float  # Loss Given Default
    ead: float  # Exposure at Default
    expected_loss: float
    risk_rating: RiskRating
    confidence_score: float

class AWSServices:
    """AWS Services integration for real-time processing"""
    
    def __init__(self):
        # Initialize AWS clients (mock for demo)
        self.s3_client = None  # boto3.client('s3')
        self.kinesis_client = None  # boto3.client('kinesis')
        self.lambda_client = None  # boto3.client('lambda')
        self.sagemaker_client = None  # boto3.client('sagemaker')
        
    def stream_to_kinesis(self, data: Dict, stream_name: str):
        """Stream data to Kinesis for real-time processing"""
        try:
            # Mock implementation
            logger.info(f"Streaming data to Kinesis stream: {stream_name}")
            return {"success": True, "stream": stream_name}
        except Exception as e:
            logger.error(f"Error streaming to Kinesis: {e}")
            return {"success": False, "error": str(e)}
    
    def invoke_sagemaker_endpoint(self, endpoint_name: str, payload: Dict):
        """Invoke SageMaker endpoint for ML predictions"""
        try:
            # Mock implementation
            logger.info(f"Invoking SageMaker endpoint: {endpoint_name}")
            return {"predictions": [0.05, 0.45, 0.85]}  # Mock PD, LGD, EAD
        except Exception as e:
            logger.error(f"Error invoking SageMaker: {e}")
            return {"error": str(e)}

class PDModel:
    """Probability of Default Model"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, application: CreditApplication) -> np.array:
        """Prepare features for PD model"""
        features = [
            application.credit_score,
            application.annual_income,
            application.debt_to_income,
            application.employment_years,
            application.loan_amount / application.annual_income,  # Loan-to-income ratio
            application.loan_term,
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: pd.DataFrame):
        """Train the PD model"""
        features = training_data[['credit_score', 'annual_income', 'debt_to_income', 
                                'employment_years', 'loan_to_income', 'loan_term']]
        target = training_data['default_flag']
        
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, target)
        self.is_trained = True
        logger.info("PD Model trained successfully")
    
    def predict(self, application: CreditApplication) -> float:
        """Predict probability of default"""
        if not self.is_trained:
            # Use rule-based approach if model not trained
            return self._rule_based_pd(application)
        
        features = self.prepare_features(application)
        features_scaled = self.scaler.transform(features)
        pd_probability = self.model.predict_proba(features_scaled)[0][1]
        return float(pd_probability)
    
    def _rule_based_pd(self, application: CreditApplication) -> float:
        """Rule-based PD calculation"""
        base_pd = 0.05
        
        # Credit score adjustment
        if application.credit_score < 600:
            base_pd *= 3
        elif application.credit_score < 700:
            base_pd *= 1.5
        elif application.credit_score > 800:
            base_pd *= 0.5
        
        # Debt-to-income adjustment
        if application.debt_to_income > 0.4:
            base_pd *= 2
        elif application.debt_to_income < 0.2:
            base_pd *= 0.7
        
        return min(base_pd, 0.95)

class LGDModel:
    """Loss Given Default Model"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def predict(self, application: CreditApplication) -> float:
        """Predict Loss Given Default"""
        # Rule-based LGD calculation
        base_lgd = 0.45  # 45% base LGD
        
        # Secured vs unsecured
        if application.property_value and application.down_payment:
            ltv = (application.loan_amount) / application.property_value
            if ltv < 0.8:
                base_lgd = 0.25  # Lower LGD for secured loans
            else:
                base_lgd = 0.35
        
        # Loan purpose adjustment
        if application.loan_purpose in ['home', 'car']:
            base_lgd *= 0.8  # Lower LGD for asset-backed loans
        elif application.loan_purpose == 'credit_card':
            base_lgd *= 1.3  # Higher LGD for unsecured credit
        
        return min(base_lgd, 0.9)

class EADModel:
    """Exposure at Default Model"""
    
    def predict(self, application: CreditApplication, current_balance: float = None) -> float:
        """Predict Exposure at Default"""
        if current_balance is None:
            current_balance = application.loan_amount
        
        # For term loans, EAD is typically the outstanding balance
        if application.loan_purpose in ['home', 'car', 'personal']:
            ead_factor = 1.0
        else:  # Credit cards and lines of credit
            ead_factor = 0.75  # Typically 75% of credit limit
        
        return current_balance * ead_factor

class CreditRiskEngine:
    """Main credit risk assessment engine"""
    
    def __init__(self):
        self.pd_model = PDModel()
        self.lgd_model = LGDModel()
        self.ead_model = EADModel()
        self.aws_services = AWSServices()
        
    def assess_risk(self, application: CreditApplication) -> RiskMetrics:
        """Comprehensive risk assessment"""
        # Calculate risk components
        pd = self.pd_model.predict(application)
        lgd = self.lgd_model.predict(application)
        ead = self.ead_model.predict(application)
        
        # Calculate expected loss
        expected_loss = pd * lgd * ead
        
        # Determine risk rating
        risk_rating = self._calculate_risk_rating(pd, expected_loss)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(application, pd, lgd)
        
        # Stream to real-time systems
        risk_data = {
            'application_id': application.application_id,
            'pd': pd,
            'lgd': lgd,
            'ead': ead,
            'expected_loss': expected_loss,
            'timestamp': datetime.now().isoformat()
        }
        self.aws_services.stream_to_kinesis(risk_data, 'credit-risk-stream')
        
        return RiskMetrics(
            pd=pd,
            lgd=lgd,
            ead=ead,
            expected_loss=expected_loss,
            risk_rating=risk_rating,
            confidence_score=confidence_score
        )
    
    def _calculate_risk_rating(self, pd: float, expected_loss: float) -> RiskRating:
        """Calculate risk rating based on PD and expected loss"""
        if pd < 0.001 and expected_loss < 0.001:
            return RiskRating.AAA
        elif pd < 0.005 and expected_loss < 0.005:
            return RiskRating.AA
        elif pd < 0.01 and expected_loss < 0.01:
            return RiskRating.A
        elif pd < 0.025 and expected_loss < 0.025:
            return RiskRating.BBB
        elif pd < 0.05 and expected_loss < 0.05:
            return RiskRating.BB
        elif pd < 0.1 and expected_loss < 0.1:
            return RiskRating.B
        elif pd < 0.2:
            return RiskRating.CCC
        elif pd < 0.5:
            return RiskRating.CC
        else:
            return RiskRating.C
    
    def _calculate_confidence(self, application: CreditApplication, pd: float, lgd: float) -> float:
        """Calculate confidence score for the risk assessment"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on data completeness
        if application.property_value and application.down_payment:
            confidence += 0.1
        if application.employment_years > 2:
            confidence += 0.05
        if application.credit_score > 0:
            confidence += 0.05
        
        return min(confidence, 0.99)

class PortfolioAnalyzer:
    """Portfolio-level risk analysis"""
    
    def __init__(self):
        self.risk_engine = CreditRiskEngine()
    
    def analyze_portfolio(self, applications: List[CreditApplication]) -> Dict:
        """Analyze portfolio-level risk metrics"""
        portfolio_metrics = []
        total_exposure = 0
        total_expected_loss = 0
        
        for app in applications:
            risk_metrics = self.risk_engine.assess_risk(app)
            portfolio_metrics.append({
                'application_id': app.application_id,
                'loan_amount': app.loan_amount,
                'pd': risk_metrics.pd,
                'lgd': risk_metrics.lgd,
                'ead': risk_metrics.ead,
                'expected_loss': risk_metrics.expected_loss,
                'risk_rating': risk_metrics.risk_rating.name
            })
            
            total_exposure += risk_metrics.ead
            total_expected_loss += risk_metrics.expected_loss
        
        # Calculate portfolio statistics
        portfolio_df = pd.DataFrame(portfolio_metrics)
        
        return {
            'total_applications': len(applications),
            'total_exposure': total_exposure,
            'total_expected_loss': total_expected_loss,
            'portfolio_loss_rate': total_expected_loss / total_exposure if total_exposure > 0 else 0,
            'avg_pd': portfolio_df['pd'].mean(),
            'avg_lgd': portfolio_df['lgd'].mean(),
            'risk_distribution': portfolio_df['risk_rating'].value_counts().to_dict(),
            'concentration_metrics': self._calculate_concentration(portfolio_df)
        }
    
    def _calculate_concentration(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate concentration risk metrics"""
        return {
            'top_10_exposure_pct': portfolio_df.nlargest(10, 'loan_amount')['loan_amount'].sum() / 
                                 portfolio_df['loan_amount'].sum() * 100,
            'herfindahl_index': ((portfolio_df['loan_amount'] / portfolio_df['loan_amount'].sum()) ** 2).sum()
        }

class StressTesting:
    """Economic stress testing framework"""
    
    def __init__(self):
        self.risk_engine = CreditRiskEngine()
    
    def run_stress_test(self, applications: List[CreditApplication], scenarios: Dict) -> Dict:
        """Run stress testing scenarios"""
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            stressed_results = []
            
            for app in applications:
                # Apply stress scenario
                stressed_app = self._apply_stress_scenario(app, scenario_params)
                risk_metrics = self.risk_engine.assess_risk(stressed_app)
                stressed_results.append(risk_metrics.expected_loss)
            
            baseline_loss = sum(self.risk_engine.assess_risk(app).expected_loss for app in applications)
            stressed_loss = sum(stressed_results)
            
            results[scenario_name] = {
                'baseline_loss': baseline_loss,
                'stressed_loss': stressed_loss,
                'loss_increase': stressed_loss - baseline_loss,
                'loss_increase_pct': (stressed_loss - baseline_loss) / baseline_loss * 100 if baseline_loss > 0 else 0
            }
        
        return results
    
    def _apply_stress_scenario(self, application: CreditApplication, scenario: Dict) -> CreditApplication:
        """Apply stress scenario to application"""
        stressed_app = CreditApplication(
            application_id=application.application_id,
            customer_id=application.customer_id,
            loan_amount=application.loan_amount,
            loan_term=application.loan_term,
            annual_income=application.annual_income * scenario.get('income_shock', 1.0),
            debt_to_income=application.debt_to_income * scenario.get('debt_shock', 1.0),
            credit_score=int(application.credit_score + scenario.get('credit_score_change', 0)),
            employment_years=application.employment_years,
            loan_purpose=application.loan_purpose,
            property_value=application.property_value * scenario.get('property_value_shock', 1.0) 
                          if application.property_value else None,
            down_payment=application.down_payment
        )
        return stressed_app

class RegulatoryReporting:
    """Generate regulatory reports"""
    
    def __init__(self):
        self.portfolio_analyzer = PortfolioAnalyzer()
    
    def generate_basel_report(self, applications: List[CreditApplication]) -> Dict:
        """Generate Basel III compliance report"""
        portfolio_analysis = self.portfolio_analyzer.analyze_portfolio(applications)
        
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'total_risk_weighted_assets': portfolio_analysis['total_exposure'] * 1.0,  # Simplified
            'capital_adequacy_ratio': 0.12,  # Mock - should calculate actual
            'tier1_capital_ratio': 0.10,
            'leverage_ratio': 0.05,
            'expected_credit_losses': portfolio_analysis['total_expected_loss'],
            'provisions_coverage': portfolio_analysis['total_expected_loss'] * 1.1,
            'risk_distribution': portfolio_analysis['risk_distribution']
        }
    
    def generate_ifrs9_report(self, applications: List[CreditApplication]) -> Dict:
        """Generate IFRS 9 ECL report"""
        portfolio_analysis = self.portfolio_analyzer.analyze_portfolio(applications)
        
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'stage1_assets': 0,  # 12-month ECL
            'stage2_assets': 0,  # Lifetime ECL - not credit impaired
            'stage3_assets': 0,  # Lifetime ECL - credit impaired
            'total_ecl_provisions': portfolio_analysis['total_expected_loss'],
            'ecl_coverage_ratio': portfolio_analysis['portfolio_loss_rate'] * 100
        }

# Example usage and testing
def create_sample_applications() -> List[CreditApplication]:
    """Create sample credit applications for testing"""
    applications = []
    
    for i in range(100):
        app = CreditApplication(
            application_id=f"APP{i:06d}",
            customer_id=f"CUST{i:06d}",
            loan_amount=np.random.uniform(10000, 500000),
            loan_term=np.random.choice([12, 24, 36, 60, 84]),
            annual_income=np.random.uniform(30000, 150000),
            debt_to_income=np.random.uniform(0.1, 0.6),
            credit_score=np.random.randint(500, 850),
            employment_years=np.random.uniform(0.5, 20),
            loan_purpose=np.random.choice(['home', 'car', 'personal', 'credit_card']),
            property_value=np.random.uniform(50000, 800000) if np.random.random() > 0.5 else None,
            down_payment=np.random.uniform(5000, 100000) if np.random.random() > 0.5 else None
        )
        applications.append(app)
    
    return applications

def main():
    """Main execution function"""
    print("=== Credit Risk Management System Demo ===\n")
    
    # Create sample data
    applications = create_sample_applications()
    
    # Initialize components
    risk_engine = CreditRiskEngine()
    portfolio_analyzer = PortfolioAnalyzer()
    stress_tester = StressTesting()
    regulatory_reporter = RegulatoryReporting()
    
    # Single application assessment
    print("1. Individual Risk Assessment:")
    sample_app = applications[0]
    risk_metrics = risk_engine.assess_risk(sample_app)
    
    print(f"Application ID: {sample_app.application_id}")
    print(f"Loan Amount: ${sample_app.loan_amount:,.2f}")
    print(f"PD: {risk_metrics.pd:.3f} ({risk_metrics.pd*100:.1f}%)")
    print(f"LGD: {risk_metrics.lgd:.3f} ({risk_metrics.lgd*100:.1f}%)")
    print(f"EAD: ${risk_metrics.ead:,.2f}")
    print(f"Expected Loss: ${risk_metrics.expected_loss:,.2f}")
    print(f"Risk Rating: {risk_metrics.risk_rating.name}")
    print(f"Confidence: {risk_metrics.confidence_score:.2f}")
    
    # Portfolio analysis
    print("\n2. Portfolio Analysis:")
    portfolio_results = portfolio_analyzer.analyze_portfolio(applications)
    print(f"Total Applications: {portfolio_results['total_applications']}")
    print(f"Total Exposure: ${portfolio_results['total_exposure']:,.2f}")
    print(f"Total Expected Loss: ${portfolio_results['total_expected_loss']:,.2f}")
    print(f"Portfolio Loss Rate: {portfolio_results['portfolio_loss_rate']:.3f} ({portfolio_results['portfolio_loss_rate']*100:.1f}%)")
    print(f"Average PD: {portfolio_results['avg_pd']:.3f}")
    print(f"Risk Distribution: {portfolio_results['risk_distribution']}")
    
    # Stress testing
    print("\n3. Stress Testing:")
    stress_scenarios = {
        'mild_recession': {
            'income_shock': 0.95,
            'credit_score_change': -20,
            'property_value_shock': 0.9
        },
        'severe_recession': {
            'income_shock': 0.8,
            'credit_score_change': -50,
            'debt_shock': 1.2,
            'property_value_shock': 0.7
        }
    }
    
    stress_results = stress_tester.run_stress_test(applications[:20], stress_scenarios)  # Sample for demo
    for scenario, results in stress_results.items():
        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  Baseline Loss: ${results['baseline_loss']:,.2f}")
        print(f"  Stressed Loss: ${results['stressed_loss']:,.2f}")
        print(f"  Loss Increase: ${results['loss_increase']:,.2f} ({results['loss_increase_pct']:.1f}%)")
    
    # Regulatory reporting
    print("\n4. Regulatory Reporting:")
    basel_report = regulatory_reporter.generate_basel_report(applications)
    print(f"Basel III Report ({basel_report['report_date']}):")
    print(f"  Risk Weighted Assets: ${basel_report['total_risk_weighted_assets']:,.2f}")
    print(f"  Capital Adequacy Ratio: {basel_report['capital_adequacy_ratio']:.1%}")
    print(f"  Expected Credit Losses: ${basel_report['expected_credit_losses']:,.2f}")
    
    ifrs9_report = regulatory_reporter.generate_ifrs9_report(applications)
    print(f"\nIFRS 9 Report ({ifrs9_report['report_date']}):")
    print(f"  Total ECL Provisions: ${ifrs9_report['total_ecl_provisions']:,.2f}")
    print(f"  ECL Coverage Ratio: {ifrs9_report['ecl_coverage_ratio']:.2f}%")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
