# Algorithmic Trading Project Report

**Author:** Everaldo Gomes  
**Date:** August 9, 2025  
**Project:**: ALGORITHMIC TRADING PRACTICE PROJECT 01 - Brazilian Cryptocurrency Algorithmic Trading System  

## Current Understanding of Central Topics and Tools

Through this project, I developed comprehensive understanding of:

**Machine Learning for Finance:** Implemented BaggingRegressor ensemble combining DecisionTree (60%) + MLP (40%) with 73 features including SMA analysis, log returns, momentum, volatility, and 6-period lags. Results showed -3.21% return vs +9.56% buy & hold, highlighting the complexity of financial prediction.

**Python Scientific Stack:** Gained hands-on proficiency with numpy, pandas, scikit-learn through practical application. Key learning included NaN handling with forward/backward fill, feature engineering pipelines, and model ensemble techniques.

**API Integration:** Developed complete mercado_bitcoin_python library (v0.2.7) with authorization token authentication, extending from basic endpoints to full trading capabilities. Achieved 79 tests coverage through systematic debugging.

**Production Systems:** Implemented comprehensive logging architecture, timeframe interpolation (5m candles), anti-loss protection considering trading fees (0.75% minimum profit), and Telegram notifications.

```python
# Key ML implementation insight
def train_once(self, historical_data: pd.DataFrame):
    # Single training without retraining (book approach)
    X, y = self.prepare_ml_data(historical_data)
    split = int(len(X) * 0.7)  # 70/30 sequential split
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    
    # Handle NaN values - critical for financial data
    nan_mask = pd.isna(X_train).any(axis=1) | pd.isna(y_train)
    X_train, y_train = X_train[~nan_mask], y_train[~nan_mask]
```

## Strengths and Weaknesses Identified

**Strengths:**
- **System Engineering:** Successfully built production-ready infrastructure with proper error handling, logging, and testing
- **Problem-Solving:** Overcame Brazilian regulatory restrictions by developing custom API library: since OANDA is not available in Brazil, I used Mercado Bitcoin for the experiments
- **AI Collaboration:** Effectively leveraged Claude Code for rapid prototyping while maintaining code quality

**Weaknesses:**
- **Feature Engineering:** Mixed different paradigms (AdaBoost book features + traditional technical indicators) which may have hindered ML performance
- **Market Intuition:** Limited experience with cryptocurrency market dynamics led to oversimplified prediction targets
- **Model Validation:** Insufficient backtesting period and inadequate warmup (55 candles) for complex feature stabilization

## Experiences Working Independently 

**Technical Background:** Computer Science (UFPR 2007), Master's (2010), 2 months experience with the Backtrader library (Nov/Dec 2024), but limited numpy/pandas hands-on experience before this project.

**Independent Development:** Successfully managed complete project lifecycle from conception to deployment. Live trading session (6 hours, SOL-BRL) generated 100% sell signals during Friday evening weak market conditions, demonstrating system's conservative behavior.

**Learning Approach:** Combined book methodology ("Python for Algorithmic Trading" pages 298-301) with practical system requirements, though this hybrid approach may have compromised ML effectiveness.

## Suggestions for Improving Existing Materials and Resources

**Enhanced Learning Path:**
1. **Timeframe Considerations:** Materials should emphasize the critical importance of timeframe selection and warmup periods for feature stability
2. **Feature Engineering Guidance:** Clearer distinction between different ML paradigms (AdaBoost vs ensemble methods) and when to combine them
3. **Production Readiness:** More emphasis on error handling, logging, and testing in financial applications

**Practical Improvements:**
- Include Brazilian market examples and regulatory considerations
- Provide template for API wrapper development when standard brokers unavailable  
- Add section on timeframe interpolation techniques for missing candle periods

## Additional Comments and Observations

**Key Insight:** Simple technical strategies often outperform complex ML models in volatile crypto markets. The project's main value was in building robust infrastructure rather than profitable predictions.

**Claude Code Experience:** AI-assisted development proved exceptionally effective for:
- Automatic best practices implementation
- Real-time learning of scientific computing patterns  
- Systematic debugging of complex integration issues

**Market Reality:** Friday evening trading conditions with SOL at 20-day momentum low validated system's conservative approach through predominant sell signals.

**Future Focus:** Asset momentum screening across multiple pairs would be more valuable than fixed-asset complex feature engineering.

---

*This independent work demonstrates modern collaborative AI development methodologies while highlighting both possibilities and limitations of ML in cryptocurrency trading.*