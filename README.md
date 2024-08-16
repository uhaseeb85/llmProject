# PDF Question Answering Application

This application allows users to load PDF documents, process them, and ask questions about their content using a local language model.

## Prerequisites

- Python 3.8 or higher
- LM Studio (for running a local language model)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-qa-app.git
   cd pdf-qa-app
   ```


2. Create a virtual environment:
   ```
   python -m venv venv
   ```


3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```


4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


5. Set up LM Studio:
   - Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Launch LM Studio and load a language model of your choice
   - Start the local server in LM Studio (usually runs on http://localhost:1234)

## Usage

1. Run the application:
   ```
   python pdf_qa_gui.py
   ```


2. Use the GUI to:
   - Load PDF documents
   - Process the documents
   - Ask questions about the content

## Building an Executable

To create a standalone executable:

1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```


2. Create the executable:
   ```
   pyinstaller --onefile --windowed pdf_qa_gui.py
   ```


3. Find the executable in the `dist` directory

Note: Users will still need LM Studio installed and running to use the application.

## Project Structure

- `pdf_qa_gui.py`: Main application file with GUI
- `pdf_qa_app.py`: Core functionality for PDF processing and question answering
- `requirements.txt`: List of Python dependencies
- `knowledge_base/`: Directory for storing processed document embeddings

## Troubleshooting

- Ensure LM Studio is running and serving a model at http://localhost:1234
- If you encounter any "module not found" errors, make sure all dependencies are installed:
  ```
  pip install -r requirements.txt
  ```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.