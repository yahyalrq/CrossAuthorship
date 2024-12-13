**CrossAuthorship Documentation**

**Table of Contents**

1. [Introduction](#introduction)
2. [Purpose](#purpose)
3. [Overview](#overview)
4. [Architecture](#architecture)
5. [Core Components](#core-components)
    - [Commit Retrieval and Validation](#commit-retrieval-and-validation)
    - [Diff Analysis](#diff-analysis)
    - [Authorship Weight Calculation](#authorship-weight-calculation)
    - [Semantic Analysis](#semantic-analysis)
    - [Abstract Syntax Tree (AST) Analysis](#abstract-syntax-tree-ast-analysis)
    - [Code Similarity Checks](#code-similarity-checks)
6. [Rationale for Fairness](#rationale-for-fairness)
7. [Fairness and Bias Mitigation](#fairness-and-bias-mitigation)
8. [Capabilities](#capabilities)
9. [Usage Instructions](#usage-instructions)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Running the Script](#running-the-script)
10. [Integration](#integration)
11. [Limitations](#limitations)
12. [Conclusion](#conclusion)
13. [Appendix](#appendix)
    - [Description of the Authorship Assessment Code](#description-of-the-authorship-assessmen)
        - [Repository Processing](#repository-processing)
        - [Diff Analysis](#diff-analysis-1)
        - [Authorship Weight Calculation](#authorship-weight-calculation-1)
        - [AST Factor](#ast-factor)
        - [Semantic Factor](#semantic-factor)
        - [Similarity Factor](#similarity-factor)
        - [Score Aggregation](#score-aggregation)
        - [Output Generation](#output-generation)
    - [Source Code](#source-code)
    - [Configuration Details](#configuration-details)
    - [AST Parser Script (parse_code.js)](#ast-parser-script-parse_codejs)
14. [Contact and Support](#contact-and-support)

**Introduction**

**CrossAuthorship** is a specialized code analysis algorithm developed to provide a fair and precise evaluation of authorship within software projects. Designed to assess the contributions of individual developers, CrossAuthorship integrates multiple analytical methodologies to ensure a comprehensive and unbiased assessment of code changes.

**Purpose**

The primary goal of CrossAuthorship is to accurately attribute code contributions to individual developers by analyzing various aspects of code changes. It aims to move beyond simplistic metrics such as lines of code added or the number of commits, offering a nuanced evaluation that considers the quality, significance, and originality of contributions.

**Overview**

CrossAuthorship operates by systematically processing commits from specified GitHub repositories, analyzing the nature and impact of code changes, and assigning weighted scores to each developer based on their contributions. The algorithm employs a combination of quantitative and qualitative analyses, including semantic classification, Abstract Syntax Tree (AST) evaluations, and code similarity checks, to ensure a balanced and fair assessment.

**Architecture**

CrossAuthorship is structured into several interconnected components that work together to analyze and evaluate code contributions. The architecture can be visualized as follows:

sql

Copy code

+----------------------+ +---------------------+

| Commit Retrieval & | | Semantic Analysis |

| Validation | | |

+----------+-----------+ +----------+----------+

| |

| |

+----------v-----------+ +----------v----------+

| Diff Analysis | | AST Analysis |

| | | |

+----------+-----------+ +----------+----------+

| |

| |

+----------v-----------+ +----------v----------+

| Authorship Weight | | Code Similarity |

| Calculation | | Checks |

+----------+-----------+ +----------+----------+

| |

+---------------+---------------+

|

v

+----------+----------+

| Score Aggregation |

| & Output Generation |

+---------------------+

Each component plays a vital role in ensuring the accuracy and fairness of the authorship assessment.

**Core Components**

**Commit Retrieval and Validation**

**Functionality:**

- **Repository Access:** Connects to specified GitHub repositories using the GitPython library.
- **Commit Extraction:** Retrieves all commits from the repositories, iterating through each branch and sorting commits chronologically.
- **Validation:** Ensures the integrity of each commit by checking its validity and the overall repository health using Git's fsck command.

**Key Features:**

- Handles multiple repositories and branches.
- Detects and processes first commits separately to account for new file additions.
- Logs detailed information about each commit for transparency.

**Diff Analysis**

**Functionality:**

- **Diff Computation:** Calculates the differences between consecutive commits to identify changes in the codebase.
- **Manual and Automatic Diffs:** Handles diffs within the same repository automatically and computes manual diffs for cross-repository changes.
- **File Filtering:** Focuses solely on JavaScript (.js) files, excluding non-relevant or peripheral files based on predefined criteria.

**Key Features:**

- Efficiently identifies added, removed, and modified lines.
- Stores diffs for later analysis, facilitating semantic and AST evaluations.
- Maintains a history of recent commits to contextualize current changes.

**Authorship Weight Calculation**

**Functionality:**

- **Weighted Scoring:** Assigns weights to contributions based on line change ratios, semantic classifications, and AST factors.
- **Composite Weighting:** Combines multiple factors to derive a comprehensive contribution score for each commit.

**Key Features:**

- Balances quantitative measures (line changes) with qualitative insights (semantic and AST analyses).
- Ensures that significant changes receive appropriate weight, regardless of their size.
- Logs detailed computations for each weight factor, enhancing transparency.

**Semantic Analysis**

**Functionality:**

- **Classification of Changes:** Utilizes OpenAI's language models to categorize code changes into predefined classifications such as Feature Addition, Bug Fix, Refactoring, etc.
- **AI-Driven Insights:** Leverages natural language understanding to discern the intent and impact of code modifications.
- **Retries and Fallbacks:** Implements retry mechanisms to handle API rate limits, ensuring robustness.

**Key Features:**

- Provides nuanced classification beyond simple metrics.
- Assigns different weights based on the nature of the change, influencing the overall authorship score.
- Integrates seamlessly with the OpenAI API for advanced semantic understanding.

**Abstract Syntax Tree (AST) Analysis**

**Functionality:**

- **Structural Evaluation:** Analyzes the structural changes in the code using AST metrics to assess the depth and impact of modifications.
- **Function and Signature Changes:** Counts the number of function changes, additions, removals, and signature alterations to gauge the complexity of changes.
- **Weighted Scoring:** Assigns factors based on the extent of structural changes, influencing the overall contribution weight.

**Key Features:**

- Differentiates between superficial and substantial code changes.
- Caps factors to prevent disproportionate weighting from extensive changes.
- Enhances the algorithm's ability to recognize meaningful contributions.

**Code Similarity Checks**

**Functionality:**

- **Similarity Metrics:** Computes Jaccard similarity and SequenceMatcher ratios to evaluate the originality of contributions.
- **Duplication Detection:** Identifies duplicate or highly similar code changes, reducing authorship weight accordingly.
- **Contextual Similarity:** Ensures that similarity checks are relevant by comparing current changes with recent changes from other authors.

**Key Features:**

- Prevents inflated scores from repetitive or non-unique code modifications.
- Balances the assessment by considering both overlap and sequence-based similarities.
- Adjusts authorship weights dynamically based on similarity factors.

**Rationale for Fairness**

The fairness of the authorship assessment code stems from its multifaceted approach, which mitigates common biases associated with traditional metrics:

1. **Contextual Line Counting:** Recognizes that raw lines of code added can be misleading, especially in projects where certain platforms require more verbose configurations or native modules. By focusing on JavaScript files and adjusting for platform-specific discrepancies, the algorithm ensures a more accurate reflection of contributions.
2. **Quality Over Quantity:** By incorporating semantic and AST analyses, the algorithm prioritizes the significance and impact of changes rather than mere quantity. This ensures that critical enhancements or complex bug fixes are appropriately weighted, regardless of the number of lines involved.
3. **Duplication and Refactoring Handling:** The algorithm identifies duplicate changes and refactoring efforts, preventing inflation of authorship scores due to repetitive or non-functional modifications.
4. **Cross-Repository Consistency:** In scenarios where code is copied across repositories (e.g., separating iOS and Android versions in React Native projects), the algorithm treats such duplications appropriately, ensuring that authorship attribution remains fair and proportional.
5. **Use of Advanced AI Models:** Leveraging OpenAI's language models for semantic analysis introduces a layer of intelligence that understands the context and purpose of code changes, leading to more nuanced and accurate classification of contributions.
6. **Transparency and Logging:** Comprehensive logging throughout the assessment process ensures transparency, allowing for auditing and verification of the authorship scores generated.
7. **Adaptive Factors:** Incorporates dynamic factors that can be tuned based on project requirements, ensuring flexibility and adaptability to different project dynamics.

**Fairness and Bias Mitigation**

CrossAuthorship is meticulously designed to ensure fairness and mitigate common biases associated with traditional authorship metrics. Here's how it achieves fairness:

1. **Quality Over Quantity:**
    - **Balanced Metrics:** Combines quantitative measures (e.g., lines of code) with qualitative analyses (e.g., semantic classification) to prioritize the significance of changes over their size.
    - **Semantic and AST Factors:** Ensures that meaningful contributions like feature additions or complex bug fixes are appropriately weighted, irrespective of the number of lines involved.
2. **Contextual Awareness:**
    - **Platform-Specific Adjustments:** Accounts for platform-specific discrepancies in codebase size and complexity, ensuring that contributions to one platform do not disproportionately influence the overall score.
    - **Repository-Specific Analysis:** Analyzes each repository individually where applicable, avoiding cross-repository biases.
3. **Duplication and Refactoring Handling:**
    - **Similarity Checks:** Detects duplicate or refactored code changes, reducing authorship weight to prevent score inflation.
    - **Refactoring Identification:** Differentiates between refactoring efforts and feature implementations, ensuring accurate attribution.
4. **Automated and Transparent Processes:**
    - **Comprehensive Logging:** Maintains detailed logs of all processes, computations, and decisions, facilitating auditing and verification.
    - **Algorithmic Transparency:** Clearly documents all factors and weighting mechanisms, ensuring that the assessment criteria are understandable and justifiable.
5. **Advanced AI Integration:**
    - **Semantic Understanding:** Utilizes AI-driven semantic analysis to grasp the intent behind code changes, promoting a deeper and fairer evaluation of contributions.
    - **Adaptive Learning:** The use of machine learning models allows the algorithm to continuously improve its classification accuracy over time.

**Capabilities**

CrossAuthorship boasts a range of capabilities that make it a robust tool for authorship assessment:

- **Multi-Faceted Analysis:** Integrates line change ratios, semantic classifications, AST evaluations, and code similarity checks to provide a holistic assessment.
- **Scalability:** Capable of handling large codebases and multiple repositories efficiently through automation.
- **Customization:** Features configurable parameters and classification factors, allowing adaptation to different project requirements.
- **Robustness:** Implements retry mechanisms and fallbacks to handle API rate limits and other potential disruptions.
- **Transparency:** Offers detailed logging and clear computation pathways, ensuring that assessments are transparent and verifiable.

**Usage Instructions**

**Prerequisites**

Before using CrossAuthorship, ensure that the following prerequisites are met:

1. **Python Environment:**
    - Python 3.7 or higher installed on your system.
    - Necessary Python libraries installed (gitpython, openai, etc.).
2. **Git Repositories:**
    - Local clones of the GitHub repositories (patt.club, prod_back_patt, patt_admin, PATT_Access, PATT_App) to be analyzed.
3. **OpenAI API Key:**
    - A valid OpenAI API key with access to the required language models.
4. **Node.js Environment:**
    - Node.js installed on your system.
    - A parse_code.js script capable of performing AST analysis based on provided input.

**Setup**

1. **Clone Repositories:**
    - Ensure that the repositories you intend to analyze are cloned locally.
    - Update the repository paths in the script (REPO_ANDROID_PATH, REPO_IOS_PATH, etc.) to point to the correct local directories.
2. **Install Dependencies:**
    - Use pip to install necessary Python libraries:

bash

Copy code

pip install gitpython openai

1. **Configure Environment Variables:**
    - Set the OPENAI_API_KEY environment variable with your OpenAI API key.
        - **Windows:**

bash

Copy code

set OPENAI_API_KEY=your_openai_api_key

- - - **Unix/Linux/macOS:**

bash

Copy code

export OPENAI_API_KEY=your_openai_api_key

1. **Prepare AST Parser:**
    - Ensure that the parse_code.js script is present in the designated directory and is executable.
    - This script should accept JSON input with old and new code snippets and output AST analysis metrics in JSON format.

**Running the Script**

1. **Execute the Script:**
    - Navigate to the directory containing the CrossAuthorship script.
    - Run the script using Python:

bash

Copy code

python cross_authorship.py

1. **Monitor Progress:**
    - The script logs detailed information about its progress, including commit processing, diff analysis, semantic classification, and weight computations.
    - Logs are output to the console and can be redirected to a file if needed.
2. **Output Generation:**
    - Upon completion, the script generates an authorship_scores.txt file containing:
        - **File Authorship Scores:** Detailed scores per file for each author.
        - **Project Authorship Scores:** Aggregate scores per author across the entire project.

**Integration**

CrossAuthorship is designed to integrate seamlessly with existing development workflows and tools:

- **Continuous Integration (CI) Pipelines:**
  - Incorporate CrossAuthorship into CI pipelines to automate authorship assessments with each commit or release.
- **Reporting Tools:**
  - Export authorship scores to reporting tools or dashboards for visualization and further analysis.
- **Developer Portals:**
  - Integrate authorship metrics into internal developer portals to recognize and reward contributions.

**Limitations**

While CrossAuthorship offers a comprehensive authorship assessment, it has certain limitations:

1. **Dependency on External Scripts:**
    - Relies on the parse_code.js script for AST analysis, requiring proper configuration and maintenance.
2. **API Rate Limits:**
    - Dependent on the OpenAI API for semantic analysis, which may be subject to rate limits and associated costs.
3. **Language and File Type Constraints:**
    - Currently focused on JavaScript (.js) files, limiting its applicability to projects using other programming languages or file types.
4. **Cross-Repository Analysis:**
    - Limited in handling complex cross-repository code coupling and dependencies, necessitating potential enhancements for large-scale projects.
5. **Manual Intervention:**
    - Certain scenarios, such as complex refactoring or unconventional code changes, may require manual review to ensure accurate classification.

**Conclusion**

CrossAuthorship stands as a robust and fair authorship assessment tool, meticulously designed to evaluate individual contributions within software projects. By integrating multiple analytical methodologies and prioritizing quality over quantity, it ensures that developers are accurately recognized for their impactful contributions. While it offers extensive capabilities, ongoing enhancements and adaptations can further extend its applicability and effectiveness across diverse project landscapes.

**Appendix**

**Description of the Authorship Assessment Code**

The provided Python script automates the process of assessing authorship across different PATT products. Here's a breakdown of its core components:

**Repository Processing**

- **Commit Retrieval:**
  - Gathers all commits from specified repositories, sorting them chronologically.
  - Iterates through each branch within the repositories to ensure comprehensive commit coverage.
- **Commit Validation:**
  - Ensures each commit is valid and the repository integrity is intact before processing.
  - Uses Git's fsck command to verify repository health, detecting any corrupted or missing objects.

**Diff Analysis**

- **Manual Diff Computation:**
  - For cross-repository differences, manually computes file-level diffs to capture accurate changes.
  - Utilizes Python's difflib library to generate unified diffs between commits from different repositories.
- **Automatic Diff Processing:**
  - Within the same repository, leverages Git's diff capabilities to identify changes between commits.
  - Processes only JavaScript (.js) files, excluding files specified in the EXCLUDED_FILES set.

**Authorship Weight Calculation**

- **Line Change Ratio:**
  - Determines the extent of changes in each commit, adjusting for platform-specific codebase sizes.
  - Calculates the ratio of added and removed lines to assess the magnitude of changes.
- **AST Factor:**
  - Analyzes the structural impact of changes using AST metrics, assigning higher weights to significant modifications.
  - Uses the parse_code.js script to extract AST-related metrics such as function additions, removals, and signature changes.
- **Semantic Factor:**
  - Classifies changes into predefined categories (e.g., Feature Addition, Bug Fix) using OpenAI's language models.
  - Assigns weights based on the nature and significance of the contribution.
- **Similarity Factor:**
  - Evaluates the originality of contributions by comparing current changes with recent changes from other authors.
  - Reduces weight for similar or duplicated code to prevent score inflation.

**AST Factor**

- **Structural Analysis:**
  - Parses both the old and new code using the parse_code.js script to generate ASTs.
  - Identifies function additions, removals, and signature changes to determine the complexity and significance of changes.
- **Weighted Scoring:**
  - Assigns weights based on the number and type of structural changes.
  - Caps factors to prevent disproportionate weighting from extensive changes.

**Semantic Factor**

- **Classification Process:**
  - Constructs prompts for OpenAI's language models to classify the nature of code changes.
  - Utilizes a primary model (gpt-4o-mini) with a fallback to gpt-3.5-turbo in case of rate limits or failures.
- **Weight Assignment:**
  - Maps classification results to predefined factors (CLASSIFICATION_FACTORS) to influence the overall authorship score.

**Similarity Factor**

- **Similarity Metrics:**
  - Computes Jaccard similarity and SequenceMatcher ratios to evaluate the overlap and sequence-based similarities between code changes.
- **Weight Adjustment:**
  - Converts similarity scores to reduction factors, decreasing the contribution weight for higher similarity to recent changes by other authors.

**Score Aggregation**

- **File-Level Scores:**
  - Aggregates authorship scores per file, reflecting individual contributions.
  - Maintains a mapping of file paths to authors and their respective scores.
- **Project-Level Scores:**
  - Summarizes overall contributions per author across the entire project.
  - Provides a holistic view of each author's impact on the project.

**Output Generation**

- **Authorship Scores:**
  - Outputs detailed authorship scores both at the file and project levels.
  - Generates an authorship_scores.txt file containing:
    - **File Authorship Scores:** JSON-formatted scores per file for each author.
    - **Project Authorship Scores:** Aggregate scores per author across the entire project.

**Configuration Details**

- **Repository Paths:**
  - Update the REPO_ANDROID_PATH and REPO_IOS_PATH variables to point to the local directories of your GitHub repositories.
- **Excluded Files:**
  - Modify the EXCLUDED_FILES set to include any additional files or patterns you wish to exclude from analysis.
- **Classification Factors:**
  - Adjust the CLASSIFICATION_FACTORS dictionary to change the weight assigned to different types of code changes.
- **OpenAI Models:**
  - Ensure that the MODEL_PRIMARY and MODEL_FALLBACK variables are set to valid OpenAI language models accessible with your API key.
- **AST Parser:**
  - The script relies on an external parse_code.js Node.js script for AST analysis. Ensure that this script is correctly implemented and accessible.
- **Repository Paths:**
  - Update the REPO_ANDROID_PATH and REPO_IOS_PATH variables to point to the local directories of your GitHub repositories.
- **Excluded Files:**
  - Modify the EXCLUDED_FILES set to include any additional files or patterns you wish to exclude from analysis.
- **Classification Factors:**
  - Adjust the CLASSIFICATION_FACTORS dictionary to change the weight assigned to different types of code changes.
- **OpenAI Models:**
  - Ensure that the MODEL_PRIMARY and MODEL_FALLBACK variables are set to valid OpenAI language models accessible with your API key.
- **AST Parser:**
  - The script relies on an external parse_code.js Node.js script for AST analysis. Ensure that this script is correctly implemented and accessible.