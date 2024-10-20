AI Code Debugging Chatbot
This project is a code debugging chatbot that uses an interface built with HTML, CSS, and JavaScript, with a Flask-based backend. The chatbot utilizes free open-source large language models (LLMs) from Hugging Face (Microsoft Codert base) for code analysis and Salesforce LLM for providing fixes and recommendations.

Features
Interactive Chatbot Interface: Users can input code snippets through a web interface and receive feedback about code vulnerabilities or errors.
AI-Powered Code Debugging: The chatbot uses LLMs to detect code issues, vulnerabilities, and best practices.
Fix Suggestions: Based on detected issues, the chatbot provides fixes using Salesforce LLM.
Open Source LLM Integration: Integrates open-source models from Hugging Face for intelligent code analysis.
Tech Stack
Frontend
HTML: Structure for the chatbot interface.
CSS: Styling for a responsive and user-friendly interface.
JavaScript: Client-side interaction, handling user inputs, and managing chatbot responses.
Backend
Flask (Python): The backend framework used to handle API requests, process data, and interact with the LLMs.
AI Models
Hugging Face Transformers: Open-source large language models (LLMs) used for code analysis and debugging. The specific model used is based on Microsoft's Codert.
Salesforce LLM: Provides detailed fixes and suggestions for improving or securing the code.
Installation and Setup
Prerequisites
Python 3.8+
Flask and other Python dependencies (see requirements below)
Node.js (optional, if you plan to extend the frontend with additional JavaScript libraries)
Steps to Install
Clone this repository:

bash

git clone https://github.com/yourusername/ai-code-debugger-chatbot.git
cd ai-code-debugger-chatbot
Install Python dependencies:

bash

pip install -r requirements.txt
Install the necessary transformers for the LLMs:

bash

pip install transformers
Run the Flask server:

bash

python app.py
Open the project in your browser:

arduino

http://localhost:5000
Example Requirements (requirements.txt):
txt
Copy code
Flask==2.0.1
transformers==4.15.0
torch==1.10.0
How to Use
Start the Flask server and access the web interface.
Input a code snippet into the chatbot interface.
The chatbot processes the code using LLMs, identifying any vulnerabilities or issues.
The chatbot will respond with:
Detected issues (e.g., syntax errors, security vulnerabilities)
Explanations for the issues
Fix suggestions provided by the Salesforce LLM
Example
If you input a code snippet with an SQL injection vulnerability, the chatbot might respond with:

bash

Detected Vulnerability: SQL Injection
Explanation: This code concatenates user input directly into the SQL query, making it vulnerable to SQL injection attacks.
Suggested Fix: Use parameterized queries instead of directly concatenating user input.


Architecture
Frontend: The chatbot interface (HTML, CSS, JavaScript) collects user input.
Backend (Flask): Sends the code snippet to the AI models for analysis.
LLMs: Hugging Face model (Microsoft Codert base) detects code issues, and Salesforce LLM provides fixes.
Response: The chatbot displays detailed analysis, explanations, and suggested fixes back to the user.
Future Improvements
Expanded LLM Support: Explore additional models for handling various programming languages and deeper code analysis.
User Authentication: Secure the chatbot for multi-user environments with user sessions.
Improved UI: Add more interactive features and better error highlighting in the code snippets.
License
This project is licensed under the MIT License. See the LICENSE file for details.

