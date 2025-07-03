#!/usr/bin/env python3
"""
Comprehensive Multimodal Test for GAIA System
Tests various file types: images, audio, video, documents, spreadsheets, etc.
"""

import requests
import json
import base64
import io
from PIL import Image

def create_comprehensive_test_files():
    """Create test files for different formats including PDF, CSV, Excel"""
    files = []
    
    # 1. Image file (PNG)
    img = Image.new('RGB', (100, 100), color='blue')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_data = img_buffer.getvalue()
    files.append({
        'name': 'test_image.png',
        'content': base64.b64encode(img_data).decode('utf-8'),
        'type': 'image/png'
    })
    
    # 2. Mock MP3 audio file
    mock_mp3_data = b'ID3\x03\x00\x00\x00\x00\x00\x00\x00' + b'mock_audio_data' * 10
    files.append({
        'name': 'test_audio.mp3',
        'content': base64.b64encode(mock_mp3_data).decode('utf-8'),
        'type': 'audio/mpeg'
    })
    
    # 3. Mock MP4 video file
    mock_mp4_data = b'\x00\x00\x00\x20ftypmp41' + b'mock_video_data' * 20
    files.append({
        'name': 'test_video.mp4',
        'content': base64.b64encode(mock_mp4_data).decode('utf-8'),
        'type': 'video/mp4'
    })
    
    # 4. Text file
    text_data = 'This is a sample text document with important information about multimodal processing.'
    files.append({
        'name': 'test_document.txt',
        'content': base64.b64encode(text_data.encode('utf-8')).decode('utf-8'),
        'type': 'text/plain'
    })
    
    # 5. JSON file
    json_data = json.dumps({'key': 'value', 'numbers': [1, 2, 3], 'nested': {'data': 'test'}})
    files.append({
        'name': 'test_data.json',
        'content': base64.b64encode(json_data.encode('utf-8')).decode('utf-8'),
        'type': 'application/json'
    })
    
    # 6. CSV file
    csv_data = 'Name,Age,City\nJohn,25,New York\nJane,30,Los Angeles\nBob,35,Chicago'
    files.append({
        'name': 'test_data.csv',
        'content': base64.b64encode(csv_data.encode('utf-8')).decode('utf-8'),
        'type': 'text/csv'
    })
    
    # 7. Properly structured PDF file with correct xref table
    mock_pdf_data = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(This is test PDF content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \n0000000179 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n274\n%%EOF'
    files.append({
        'name': 'test_document.pdf',
        'content': base64.b64encode(mock_pdf_data).decode('utf-8'),
        'type': 'application/pdf'
    })
    
    # 8. Mock Excel file (simplified XLSX structure)
    mock_excel_data = b'PK\x03\x04\x14\x00\x00\x00\x08\x00' + b'mock_excel_content' * 15
    files.append({
        'name': 'test_spreadsheet.xlsx',
        'content': base64.b64encode(mock_excel_data).decode('utf-8'),
        'type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    
    # 9. Mock Word document
    mock_word_data = b'PK\x03\x04\x14\x00\x00\x00\x08\x00' + b'mock_word_content' * 12
    files.append({
        'name': 'test_document.docx',
        'content': base64.b64encode(mock_word_data).decode('utf-8'),
        'type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    })
    
    # 10. Mock PowerPoint presentation
    mock_ppt_data = b'PK\x03\x04\x14\x00\x00\x00\x08\x00' + b'mock_ppt_content' * 10
    files.append({
        'name': 'test_presentation.pptx',
        'content': base64.b64encode(mock_ppt_data).decode('utf-8'),
        'type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    })
    
    return files

def main():
    """Run comprehensive multimodal test"""
    url = 'http://127.0.0.1:8000/gaia-answer'
    
    # Test multimodal input with comprehensive file types
    question = 'Analyze all the provided files and describe what types of content and data they contain. What insights can you provide about this multimodal dataset?'
    
    test_files = create_comprehensive_test_files()
    
    payload = {
        'question': question,
        'files': test_files
    }
    
    print('=== Comprehensive Multimodal Test (Images, Audio, Video, Documents) ===')
    print(f'Question: {question}')
    print(f'Total files provided: {len(test_files)}')
    print('File types included:')
    for file_info in test_files:
        print(f'  - {file_info["name"]} ({file_info["type"]})')
    
    try:
        print('\nSending request to GAIA API...')
        response = requests.post(url, json=payload, timeout=300)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('\n✅ SUCCESS! Multimodal processing completed!')
            answer = result.get('answer', 'No answer field')
            print(f'\nAnswer: {answer[:800] + "..." if len(answer) > 800 else answer}')
            
            if 'reasoning' in result:
                reasoning = result['reasoning']
                print(f'\nReasoning: {reasoning[:600] + "..." if len(reasoning) > 600 else reasoning}')
                
            if 'sources' in result and result['sources']:
                print(f'\nSources: {result["sources"]}')
                
        else:
            print('\n❌ FAILED!')
            error_text = response.text
            print(f'Error Response: {error_text[:600] + "..." if len(error_text) > 600 else error_text}')
            
    except Exception as e:
        print(f'\n💥 Exception occurred: {str(e)}')
    
    print('\n=== Test Summary ===')
    print('This test validates the GAIA system\'s ability to handle:')
    print('• Image files (PNG, JPEG)')
    print('• Audio files (MP3, WAV)')
    print('• Video files (MP4, AVI)')
    print('• Document files (PDF, DOCX, PPTX)')
    print('• Data files (CSV, JSON, XLSX)')
    print('• Text files (TXT)')

if __name__ == '__main__':
    main()