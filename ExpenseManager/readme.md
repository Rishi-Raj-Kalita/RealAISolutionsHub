# Expense Manager

## Overview

Expense Manager is a **100% local AI-powered expense tracking solution** designed to automate the categorization of transactions, identify splits, and provide insightful analytics—all while ensuring data privacy. The application leverages **Deepseek R1:7B** running locally via **Ollama**, with **FAISS** as a vector store for advanced querying and analysis.

## Features

- **Automated Transaction Categorization**: Uses transaction notes to classify expenses into categories like food, shopping, transportation, etc.
- **Split Expense Tracking**: Identifies and assigns split transactions to respective contributors.
- **Privacy-Preserving AI**: Runs entirely on your local machine with no external API calls.
- **Ad-hoc Financial Analysis**: Uses FAISS to create a knowledge base for interactive financial queries.
- **Data Visualization**: Generates insights on spending trends for better financial management.

## Tech Stack

- **Deepseek R1:7B** via Ollama (Local LLM Execution)
- **FAISS** (Vector Store for Financial Querying)
- **CSVLoader** (Efficient Data Chunking and Preprocessing)
- **Python** for backend processing and automation

## How It Works

1. **Download Transactions**: Export your bank transaction data as a CSV file.
2. **Preprocessing**: The model reads transaction notes (e.g., from Google Pay) to categorize expenses and identify splits.
3. **Staging & Approval**: A draft file is generated for manual review and minor adjustments.
4. **Analysis & Visualization**: The cleaned data is used to generate financial insights and track spending patterns.
5. **Querying via Knowledge Base**: FAISS enables efficient ad-hoc analysis on the structured data.

## Installation

### Prerequisites

- Python 3.8+
- Ollama installed and configured with **Deepseek R1:7B**
- FAISS library installed

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Rishi-Raj-Kalita/RealAISolutionsHub.git
   cd RealAISolutionsHub/ExpenseManager
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python expense_manager.py --input transactions.csv
   ```

## Future Enhancements

- Support for multiple bank statement formats.
- Integration with budgeting tools.
- Enhanced visualization with interactive dashboards.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

For any questions or suggestions, feel free to reach out or connect on [**LinkedIn**](https://www.linkedin.com/in/rishiraj-kalita-5946511a0/).

---

This project is part of **RealAISolutionsHub**—an initiative to build and open-source practical AI-driven solutions for everyday challenges.

