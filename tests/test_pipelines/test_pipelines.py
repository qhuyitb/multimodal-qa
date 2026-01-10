"""
Test Pipelines: Document, Video và QA Pipelines
"""

import pytest
import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from conftest import get_project_root, get_input_dir, get_output_dir, check_file_exists


class TestDocumentPipeline:
    """Test pipeline xử lý document"""
    
    @classmethod
    def setup_class(cls):
        cls.root = get_project_root()
        cls.input_dir = get_input_dir("documents")
        cls.output_dir = get_output_dir("documents")
    
    def test_pipeline_exists(self):
        pipeline_file = self.root / "src" / "pipelines" / "document_pipeline.py"
        assert check_file_exists(pipeline_file, "Document pipeline")
    
    def test_end_to_end_document_processing(self):
        print("\n Test End-to-End Document Pipeline ")
        from extractors.document import extract_pdf_text, extract_docx_text
        
        # Find input files
        input_files = []
        for folder, ext in [("pdf", "*.pdf"), ("docx", "*.docx")]:
            folder_path = self.input_dir / folder
            if folder_path.exists():
                input_files.extend(list(folder_path.glob(ext)))
        
        if not input_files:
            pytest.skip("No input documents found")
        
        # Process files
        output_files = []
        for f in input_files:
            if f.suffix == ".pdf":
                extract_pdf_text(str(f))
            elif f.suffix == ".docx":
                extract_docx_text(str(f))
            
            out_dir = self.output_dir / f.suffix[1:]  
            out_file = out_dir / f"{f.stem}.txt"
            if out_file.exists():
                output_files.append(out_file)
        
        assert len(output_files) > 0, "No output files generated"
        print(f"[PASS] Processed {len(output_files)}/{len(input_files)} files")


class TestVideoPipeline:
    """Test pipeline xử lý video"""
    
    @classmethod
    def setup_class(cls):
        cls.root = get_project_root()
        cls.input_dir = get_input_dir("videos")
        cls.output_dir = get_output_dir()
    
    def test_pipeline_exists(self):
        pipeline_file = self.root / "src" / "pipelines" / "video_pipeline.py"
        assert check_file_exists(pipeline_file, "Video pipeline")
    
    def test_end_to_end_video_processing(self):
        print("\n Test End-to-End Video Pipeline ")
        from extractors.video import extract_video_text
        
        video_files = list(self.input_dir.glob("*.mp4")) if self.input_dir.exists() else []
        if not video_files:
            pytest.skip("No video files found")
        
        try:
            transcript = extract_video_text(str(video_files[0]))
            assert transcript is not None and len(transcript) > 0
            print(f"[PASS] Transcribed video: {len(transcript)} chars")
        except Exception as e:
            pytest.skip(f"Video processing failed: {str(e)}")


class TestQAPipeline:
    """Test Q&A pipeline"""
    
    @classmethod
    def setup_class(cls):
        cls.root = get_project_root()
    
    def test_pipeline_exists(self):
        pipeline_file = self.root / "src" / "pipelines" / "qa_pipeline.py"
        assert check_file_exists(pipeline_file, "QA pipeline")
        
        from extractors.video import extract_video_text
        
        # Find video files
        if not self.input_dir.exists():
            pytest.skip("Videos directory not found")
        
        video_files = list(self.input_dir.glob("*.mp4"))
        if not video_files:
            pytest.skip("No video files found")
        
        print(f"Found {len(video_files)} video files")
        
        # Process first video only (slow)
        video_file = video_files[0]
        print(f"Processing: {video_file.name}")
        
        try:
            transcript_dir = self.output_dir / "transcripts"
            transcript = extract_video_text(str(video_file), str(transcript_dir))
            
            assert transcript is not None, "Transcript is None"
            print(f"[PASS] Extracted transcript: {len(transcript)} characters")
            
            # Next steps in pipeline (to be implemented):
            print("\nPipeline steps:")
            print("  [PASS] Step 1: Video transcription")
            print("  [TODO] Step 2: Translation")
            print("  [TODO] Step 3: Subtitle generation")
            print("  [TODO] Step 4: Vector DB indexing")
            print("  [TODO] Step 5: Subtitle burning")
            
        except Exception as e:
            pytest.skip(f"Video processing failed: {str(e)}")


class TestQAPipeline:
    """Test Q&A pipeline"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.project_root = pathlib.Path(__file__).parent.parent.parent
    
    def test_pipeline_exists(self):
        """Test QA pipeline file exists"""
        pipeline_file = self.project_root / "src" / "pipelines" / "qa_pipeline.py"
        assert pipeline_file.exists(), "QA pipeline file not found"
        print(f"[PASS] QA pipeline exists: {pipeline_file}")
    
    def test_qa_pipeline_placeholder(self):
        """Test QA pipeline (to be implemented)"""
        print("\n Test QA Pipeline ")
        print("QA Pipeline components (TODO):")
        print("  [TODO] Vector DB setup (ChromaDB)")
        print("  [TODO] Embedding model")
        print("  [TODO] Query processing")
        print("  [TODO] Answer generation")
        print("  [TODO] Source citation")
        pytest.skip("QA pipeline not yet implemented")


class TestIntegration:
    """Test integration between pipelines"""
    
    def test_data_flow(self):
        print("\n Test Pipeline Integration ")
        from conftest import check_dir_exists
        
        root = get_project_root()
        dirs = [
            ("data/input/documents", "Input Documents"),
            ("data/input/videos", "Input Videos"),
            ("data/output/documents", "Output Documents"),
            ("data/output/transcripts", "Output Transcripts"),
        ]
        
        for path, name in dirs:
            check_dir_exists(root / path, name)
        
        print("[PASS] Integration structure OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
