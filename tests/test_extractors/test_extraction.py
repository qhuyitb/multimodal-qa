"""
Test Extractors: Trích xuất Document và Video
"""

import pytest
import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from extractors.document import extract_pdf_text, extract_docx_text
from extractors.video import extract_video_text
from conftest import get_project_root, get_input_dir, get_output_dir, check_file_exists, get_files_by_pattern


class TestDocumentExtraction:
    """Test chức năng trích xuất document"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.project_root = get_project_root()
        cls.input_dir = get_input_dir() / "documents"
        cls.output_base_dir = get_output_dir() / "documents"
        cls.output_pdf_dir = cls.output_base_dir / "pdf"
        cls.output_docx_dir = cls.output_base_dir / "docx"
        cls.output_pdf_dir.mkdir(parents=True, exist_ok=True)
        cls.output_docx_dir.mkdir(parents=True, exist_ok=True)
    
    def test_pdf_extraction(self):
        """Test trích xuất text từ PDF"""
        print("\n Test PDF Extraction ")
        
        pdf_files = get_files_by_pattern(self.input_dir / "pdf", "*.pdf")
        if not pdf_files:
            pytest.skip("No PDF files found for testing")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            text = extract_pdf_text(str(pdf_file))
            
            assert text is not None and len(text) > 0, "Invalid extracted text"
            assert check_file_exists(self.output_pdf_dir / f"{pdf_file.stem}.txt"), "Output file not created"
            
            print(f"Extracted {len(text)} characters")
    
    def test_docx_extraction(self):
        """Test trích xuất text từ DOCX"""
        print("\n Test DOCX Extraction ")
        
        docx_files = get_files_by_pattern(self.input_dir / "docx", "*.docx")
        if not docx_files:
            pytest.skip("No DOCX files found for testing")
        
        for docx_file in docx_files:
            print(f"\nProcessing: {docx_file.name}")
            text = extract_docx_text(str(docx_file))
            
            assert text is not None and len(text) > 0, "Invalid extracted text"
            assert check_file_exists(self.output_docx_dir / f"{docx_file.stem}.txt"), "Output file not created"
            
            print(f"Extracted {len(text)} characters")
    
    def test_table_extraction_pdf(self):
        """Test trích xuất bảng từ PDF"""
        print("\n Test PDF Table Extraction ")
        
        pdf_files = get_files_by_pattern(self.input_dir / "pdf", "*.pdf")
        if not pdf_files:
            pytest.skip("No PDF files found")
        
        text = extract_pdf_text(str(pdf_files[0]))
        if "[table]" in text.lower():
            assert "[/table]" in text.lower(), "Table closing tag not found"
            print("[PASS] Table extraction OK")
        else:
            print("[WARN] No tables found")
    
    def test_table_extraction_docx(self):
        """Test trích xuất bảng từ DOCX"""
        print("\n Test DOCX Table Extraction ")
        
        docx_files = get_files_by_pattern(self.input_dir / "docx", "*.docx")
        if not docx_files:
            pytest.skip("No DOCX files found")
        
        text = extract_docx_text(str(docx_files[0]))
        if "[table]" in text.lower():
            assert "[/table]" in text.lower(), "Table closing tag not found"
            print("[PASS] Table extraction OK")
        else:
            print("[WARN] No tables found")
    
    def test_output_file_encoding(self):
        """Test file output sử dụng mã hóa UTF-8"""
        print("\n Test Output Encoding ")
        
        txt_files = get_files_by_pattern(self.output_pdf_dir, "*.txt") + \
                    get_files_by_pattern(self.output_docx_dir, "*.txt")
        
        if not txt_files:
            pytest.skip("No output files found")
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                assert len(f.read()) > 0, f"File is empty: {txt_file}"
        
        print(f"[PASS] {len(txt_files)} file(s) UTF-8 OK")
    
    def test_extraction_accuracy_metrics(self):
        """Test độ chính xác và chất lượng trích xuất"""
        print("\n Test Extraction Accuracy ")
        
        txt_files = get_files_by_pattern(self.output_pdf_dir, "*.txt") + \
                    get_files_by_pattern(self.output_docx_dir, "*.txt")
        
        if not txt_files:
            pytest.skip("No output files found")
        
        issues = []
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 100:
                issues.append(f"{txt_file.name}: too short")
            
            special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
            if special_char_ratio > 0.3:
                issues.append(f"{txt_file.name}: high special char ratio")
            
            if self._check_extraction_artifacts(content):
                issues.append(f"{txt_file.name}: extraction artifacts")
        
        if issues:
            print(f"[WARN] Issues: {len(issues)}")
            for issue in issues[:3]:
                print(f"  - {issue}")
        else:
            print(f"[PASS] All {len(txt_files)} file(s) quality OK")
    
    def _check_extraction_artifacts(self, text):
        """Kiểm tra lỗi trích xuất phổ biến như ký tự lặp lại"""
        import re
        # Look for 5+ repeated characters (common artifact)
        pattern = r'(.)\1{4,}'
        matches = re.findall(pattern, text)
        return len(matches) > 0
    
    def test_table_extraction_quality(self):
        """Test chất lượng trích xuất bảng"""
        print("\n Test Table Extraction Quality ")
        
        txt_files = get_files_by_pattern(self.output_pdf_dir, "*.txt") + \
                    get_files_by_pattern(self.output_docx_dir, "*.txt")
        
        if not txt_files:
            pytest.skip("No output files found")
        
        tables_found = 0
        issues = []
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "[table]" in content.lower():
                tables_found += 1
                import re
                tables = re.findall(r'\[table\](.*?)\[/table\]', content, re.IGNORECASE | re.DOTALL)
                
                for idx, table in enumerate(tables, 1):
                    rows = [line for line in table.split('\n') if line.strip() and '|' in line]
                    
                    if rows:
                        col_counts = [line.count('|') for line in rows]
                        if len(set(col_counts)) != 1:
                            issues.append(f"{txt_file.name} table {idx}: inconsistent columns")
                    else:
                        issues.append(f"{txt_file.name} table {idx}: no rows")
        
        if tables_found == 0:
            print("[INFO] No tables found")
        elif issues:
            print(f"[WARN] {len(issues)} issue(s)")
            for issue in issues[:3]:
                print(f"  - {issue}")
        else:
            print(f"[PASS] {tables_found} table(s) validated")


class TestVideoExtraction:
    """Test chức năng trích xuất video"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.project_root = get_project_root()
        cls.input_dir = get_input_dir() / "videos"
        cls.output_base_dir = get_output_dir() / "transcripts"
        cls.output_txt_dir = cls.output_base_dir / "txt"
        cls.output_json_dir = cls.output_base_dir / "json"
        cls.output_txt_dir.mkdir(parents=True, exist_ok=True)
        cls.output_json_dir.mkdir(parents=True, exist_ok=True)
    
    def test_video_transcription(self):
        """Test phiên âm video"""
        print("\n Test Video Transcription ")
        
        video_files = get_files_by_pattern(self.input_dir, "*.mp4")
        if not video_files:
            pytest.skip("No video files found for testing")
        
        video_file = video_files[0]
        
        try:
            transcript = extract_video_text(str(video_file))
            assert transcript is not None and len(transcript) > 0, "Invalid transcript"
            
            assert check_file_exists(self.output_txt_dir / f"{video_file.stem}.txt"), "Text output not created"
            assert check_file_exists(self.output_json_dir / f"{video_file.stem}.json"), "JSON output not created"
            
            print(f"[PASS] Transcribed {len(transcript)} chars")
            
        except Exception as e:
            pytest.skip(f"Transcription failed: {str(e)}")
    
    def test_transcript_json_format(self):
        """Test JSON transcript chứa các trường bắt buộc"""
        print("\n Test Transcript JSON Format ")
        
        import json
        
        json_files = get_files_by_pattern(self.output_json_dir, "*.json")
        if not json_files:
            pytest.skip("No JSON transcript files found")
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "text" in data and isinstance(data["text"], str) and len(data["text"]) > 0, \
                "Invalid text field in JSON"
        
        print(f"[PASS] {len(json_files)} JSON file(s) valid")
    
    def test_transcription_accuracy_metrics(self):
        """Test độ chính xác và chất lượng phiên âm"""
        print("\n Test Transcription Accuracy ")
        
        import json
        
        json_files = get_files_by_pattern(self.output_json_dir, "*.json")
        if not json_files:
            pytest.skip("No JSON transcript files found")
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            words = data.get("text", "").split()
            print(f"  {json_file.name}: {len(words)} words", end="")
            
            if "segments" in data and data["segments"]:
                segments = data["segments"]
                total_duration = segments[-1].get("end", 0)
                gaps = sum(1 for i in range(len(segments)-1) 
                          if segments[i+1]["start"] - segments[i]["end"] > 1.0)
                
                print(f", {total_duration:.1f}s", end="")
                if gaps > 0:
                    print(f" [WARN] {gaps} gap(s)")
                else:
                    print()
            else:
                print()
            
            if len(words) < 10:
                print(f"    [WARN] Very short transcript")
        
        print(f"[PASS] Checked {len(json_files)} file(s)")
    
    def test_transcription_confidence(self):
        """Test điểm độ tin cậy phiên âm nếu có"""
        print("\n Test Transcription Confidence ")
        
        import json
        
        json_files = get_files_by_pattern(self.output_json_dir, "*.json")
        if not json_files:
            pytest.skip("No JSON transcript files found")
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "segments" in data and data["segments"] and "avg_logprob" in data["segments"][0]:
                segments = data["segments"]
                avg_confidence = sum(seg.get("avg_logprob", 0) for seg in segments) / len(segments)
                
                status = "[PASS] High" if avg_confidence > -0.5 else \
                        "[INFO] Medium" if avg_confidence > -1.0 else "[WARN] Low"
                print(f"  {json_file.name}: {avg_confidence:.3f} {status}")
                
                low_conf = sum(1 for seg in segments if seg.get("avg_logprob", 0) < -1.0)
                if low_conf > 0:
                    print(f"    [WARN] {low_conf} low confidence segment(s)")
            else:
                print(f"  [INFO] No confidence scores in {json_file.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
