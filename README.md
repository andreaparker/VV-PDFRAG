README.md

# VisionVault-PDFRAG

VisionVault-PDFRAG is an advanced, vision-based Retrieval-Augmented Generation (RAG) system for document analysis and question-answering.

## Key Features

- **OCR-Free Document Processing**: Uses visual embeddings to understand documents without text extraction.
- **Multi-Format Support**: Handles PDFs and images seamlessly.
- **Intelligent Retrieval**: Leverages ColPali or ColQwen for efficient document page retrieval.
- **Flexible Response Generation**: Utilizes various Vision Language Models (VLMs) for comprehensive answers.
- **User-Friendly Interface**: Offers a chat-based interface for easy interaction.
- **Session Management**: Enables creation, renaming, and switching between chat sessions.
- **Persistent Storage**: Saves indexes and sessions for future use.

## How It Works

1. **Document Indexing**:
   - Converts uploaded documents to visual embeddings.
   - Stores these embeddings for quick retrieval.

2. **Query Processing**:
   - Matches user queries against stored visual embeddings.
   - Retrieves the most relevant document pages.

3. **Response Generation**:
   - Passes retrieved pages to a chosen VLM.
   - Generates responses based on visual and textual content.

## Supported Models

- **Retrieval**: ColPali, ColQwen
- **Response Generation**: Qwen2-VL-7B-Instruct, OpenAI GPT-4o,

## Getting Started

### Prerequisites

- Anaconda or Miniconda
- Python 3.10+
- Git
- LLM API keys if you want to run non-local workflows

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/andreaparker/VV-PDFRAG.git
   cd VV-PDFRAG
   ```

2. Set up the environment:
   ```bash
   conda create -n visionvault-pdfrag-local python=3.10
   conda activate visionvault-pdfrag-local
   pip install -r requirements.txt
   pip install git+https://github.com/huggingface/transformers
   ```

3. Configure API keys:

### 3.1. For Local Development

If you're running the project locally, you can set the API key as an environment variable:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Replace 'your_openai_api_key_here' with your actual OpenAI API key.

For persistence across terminal sessions, you can add this line to your `~/.bashrc` or `~/.zshrc` file.

### 3.2. For Terraform Deployment

When deploying with Terraform, you'll need to provide the API key as a variable during the `terraform apply` command:

```bash
terraform apply -var="OPENAI_API_KEY=your_openai_api_key_here"
```

We ignore the lower-case naming convention for this variable name.

Alternatively, you can create a `terraform.tfvars` file in your project directory with the following content:

```hcl
openai_api_key = "your_openai_api_key_here"
```

⚠️ **Important Security Note**: 
- Never commit your API key to version control. 
- If using a `terraform.tfvars` file, make sure it's listed in your `.gitignore`.
- For CI/CD pipelines, use secure environment variables or secrets management provided by your CI/CD tool.

### 3.3 Verifying the API Key

To verify that your API key is correctly set, you can run the following command:

For local development:
```bash
echo $OPENAI_API_KEY
```

For Terraform (after apply):
```bash
terraform output openai_api_key
```

If configured correctly, these commands should display your API key (or a masked version of it in the case of Terraform).

4. Launch the application:
   ```bash
   python app.py
   ```

4.1 After launching the app

- a `chat_messages.html` file will be created: this file creates a visual chat history where you can see who said what, and any images that were part of the conversation


5. Access the web interface at `http://localhost:5050/`

## Usage Guide

1. **Start a New Chat**:
   - Click "New Chat" to begin a session.

2. **Upload Documents**:
   - Choose PDF or image files.
   - Click "Upload and Index" to process documents.

3. **Ask Questions**:
   - Type your query in the chat box.
   - Click "Send" to get responses.

4. **Manage Sessions**:
   - Rename, switch, or delete sessions as needed.

5. **Adjust Settings**:
   - Select preferred language models and image dimensions.

## Project Structure

```
VisionVault-PDFRAG/
├── app.py                 # Main Flask application
├── models/                # Core functionality modules
├── templates/             # HTML templates
├── static/                # CSS, JavaScript, and images
├── uploaded_documents/    # Storage for uploaded files
├── byaldi_indices/        # Storage for document indexes
└── sessions/              # Session data storage
```

## Adding your favorite models

To add in new models or domain-specific models you can create a branch, work on that and then submit your feature branch back to us:

1. Fork the repository.
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request.

## Learn More

For a detailed understanding of the system's workflow and architecture, please refer to the full documentation which will be written in early Q4 2024.
