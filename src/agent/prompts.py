"""Custom prompts for the data analysis agent."""

DATA_ANALYSIS_PROMPT = """You are an expert data analyst assistant. Your role is to help users understand and analyze their data.
When analyzing data, follow these guidelines:

1. Always start with basic statistical summaries
2. Look for patterns and correlations
3. Identify potential outliers or anomalies
4. Provide clear, concise explanations
5. Use appropriate visualizations when needed
6. Consider the context of the data

Current conversation:
{chat_history}

User question: {question}

Think step by step:
1. What type of analysis is needed?
2. What tools should I use?
3. How should I present the findings?

Your analysis:"""

SUMMARY_PROMPT = """Please provide a comprehensive summary of the dataset. Include:
1. Basic statistics (mean, median, standard deviation)
2. Data distribution characteristics
3. Key insights and patterns
4. Potential areas for further analysis

Dataset context: {context}
Data summary: {summary}

Your analysis:"""

CORRELATION_PROMPT = """Analyze the correlations in the dataset. Consider:
1. Strong positive and negative correlations
2. Potential causal relationships
3. Interesting patterns or anomalies
4. Recommendations for further analysis

Correlation matrix: {correlations}

Your analysis:"""

TREND_ANALYSIS_PROMPT = """Analyze the trends in the dataset. Focus on:
1. Temporal patterns (if time series data)
2. Growth or decline trends
3. Seasonal patterns
4. Significant changes or breakpoints

Data context: {context}
Trend information: {trends}

Your analysis:"""

OUTLIER_DETECTION_PROMPT = """Identify and analyze outliers in the dataset. Consider:
1. Statistical methods for outlier detection
2. Potential causes of outliers
3. Impact on analysis
4. Recommendations for handling outliers

Data context: {context}
Outlier information: {outliers}

Your analysis:"""

VISUALIZATION_PROMPT = """Suggest appropriate visualizations for the data. Consider:
1. Data type and distribution
2. Key relationships to highlight
3. Best visualization types
4. Important features to include

Data context: {context}
Visualization requirements: {requirements}

Your recommendations:""" 