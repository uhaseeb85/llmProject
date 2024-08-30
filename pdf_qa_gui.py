import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QFileDialog, QLabel, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pdf_qa_app import PDFKnowledgeBase, PDFQuestionAnswering

class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, directory, knowledge_base):
        super().__init__()
        self.directory = directory
        self.knowledge_base = knowledge_base

    def run(self):
        pdf_files = [f for f in os.listdir(self.directory) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        for i, pdf_file in enumerate(pdf_files):
            file_path = os.path.join(self.directory, pdf_file)
            self.knowledge_base.add_pdf(file_path)
            progress = int((i + 1) / total_files * 100)
            self.progress_update.emit(progress)
        
        self.finished.emit(self.knowledge_base)

class PDFQuestionAnsweringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.knowledge_base = PDFKnowledgeBase()
        self.qa = PDFQuestionAnswering(self.knowledge_base)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PDF Question Answering App')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # PDF loading section
        pdf_layout = QHBoxLayout()
        self.pdf_label = QLabel('No PDFs loaded')
        pdf_layout.addWidget(self.pdf_label)
        
        load_button = QPushButton('Load PDFs')
        load_button.clicked.connect(self.load_pdfs)
        pdf_layout.addWidget(load_button)

        refresh_button = QPushButton('Refresh Data')
        refresh_button.clicked.connect(self.refresh_data)
        pdf_layout.addWidget(refresh_button)

        layout.addLayout(pdf_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Question input
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Enter your question here...")
        layout.addWidget(self.question_input)

        # Answer button
        answer_button = QPushButton('Get Answer')
        answer_button.clicked.connect(self.get_answer)
        layout.addWidget(answer_button)

        # Answer display
        self.answer_display = QTextEdit()
        self.answer_display.setReadOnly(True)
        layout.addWidget(self.answer_display)

        central_widget.setLayout(layout)

    def load_pdfs(self):
        directory = QFileDialog.getExistingDirectory(self, "Select PDF Directory")
        if directory:
            self.current_directory = directory
            self.pdf_label.setText(f"Processing PDFs from: {directory}")
            self.process_pdfs()

    def refresh_data(self):
        if hasattr(self, 'current_directory'):
            self.pdf_label.setText(f"Refreshing PDFs from: {self.current_directory}")
            self.process_pdfs()
        else:
            self.pdf_label.setText("No directory selected. Please load PDFs first.")

    def process_pdfs(self):
        self.progress_bar.setVisible(True)
        self.processing_thread = ProcessingThread(self.current_directory, self.knowledge_base)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_processing_finished(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.qa = PDFQuestionAnswering(self.knowledge_base)
        self.pdf_label.setText("PDFs processed and ready for questions")
        self.progress_bar.setVisible(False)

    def get_answer(self):
        question = self.question_input.toPlainText()  # Changed from text() to toPlainText()
        self.answer_display.clear()  # Changed from answer_output to answer_display
        answer = self.qa.answer_question(question, self.update_answer)
        self.answer_display.append(answer)

    def update_answer(self, token: str):
        self.answer_display.insertPlainText(token)  # Changed from answer_output to answer_display
        self.answer_display.ensureCursorVisible()
        QApplication.processEvents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PDFQuestionAnsweringApp()
    ex.show()
    sys.exit(app.exec())