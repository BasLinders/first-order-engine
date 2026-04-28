# Setting Up the First Order Engine (FOE)

This document provides a step-by-step guide to setting up your local development environment and preparing the **First Order Engine** for production deployment.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10+**: The engine is optimized for Python 3.10 to match the Google Cloud Functions runtime.
* **Git**: To clone the repository and manage version control.
* **Virtualenv/Conda**: Highly recommended for isolating project dependencies.
* **Google Cloud SDK**: Required if you plan to deploy manually from your terminal.

---

## 2. Local Installation

Follow these steps to get the engine running on your machine:

### Clone the Repository
```bash
git clone [https://github.com/your-org/first-order-engine.git](https://github.com/your-org/first-order-engine.git)
cd first-order-engine
```

### Create and Activate a Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

### Install the FOE Package
Install the package in editable mode with development dependencies. This allows you to modify the source code in foe/ and see changes immediately without re-installing the package.
```bash
pip install -e .[dev]
```

## 3. Running the Test Suite
We use pytest for mathematical and structural validation. Always run the tests before pushing to main to ensure the "First Order" logic remains intact.

# Run all unit tests with coverage reporting
```bash
pytest tests/ --cov=foe --cov-report=term-missing
```
Important: Statistical failures are often due to floating-point precision issues; if you are writing new tests, use pytest.approx() for assertions.

## 4. Local GCP Simulation
To test the Cloud Function entry points without deploying to Google Cloud, use the functions-framework.

```bash
# Install the framework:
pip install functions-framework
```
## 5. Start the local server
Point the framework to your specific module handler. For example, to test the Frequentist module:

```bash
# From the root of the project
export PYTHONPATH=$PYTHONPATH: functions-framework --target=frequentist_handler --source=gcp/functions/frequentist/main.py
```

## 6. Deployment Configuration
**GitHub Actions Secrets**
To enable the automated deployment pipeline (.github/workflows/deploy.yml), you must configure the following Secrets in your GitHub repository settings:
* Secret Name
* Descriptio
* nGCP_WIP_PROVIDER
* The full path to your Workload Identity Provider (e.g., projects/123/locations/global/workloadIdentityPools/...).
* GCP_SERVICE_ACCOUNT
* The email of the Service Account with Cloud Functions Developer and Service Account User permissions.

**GCP Project Setup** 
Ensure the following APIs are enabled in your Google Cloud Project:
* Cloud Functions API
* Cloud Build API
* Artifact Registry API6.

**Project Structure Overview**
* foe/: The core statistical library containing the mathematical engines.
* gcp/functions/: The adapter layer that routes JSON requests into the engine.
* tests/: The automated validation suite for ensuring mathematical truth.
* pyproject.toml: The central source of truth for all metadata and dependencies.

For technical support or mathematical inquiries regarding predicate logic integration, please contact the engineering team.
