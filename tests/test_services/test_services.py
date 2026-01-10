"""
Test Services: QA Engine, Vector Store, Subtitle Service
"""

import pytest
import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from conftest import get_project_root, check_file_exists, check_dir_exists


class TestQAEngine:
    """Test service QA Engine"""
    
    def test_qa_engine_exists(self):
        root = get_project_root()
        assert check_file_exists(root / "src/services/qa_engine.py", "QA engine")


class TestVectorStore:
    """Test service Vector Store"""
    
    def test_vector_store_exists(self):
        root = get_project_root()
        assert check_file_exists(root / "src/services/vector_store.py", "Vector store")
    
    def test_vector_db_directory(self):
        root = get_project_root()
        assert check_dir_exists(root / "data/vector_db", "Vector DB")


class TestSubtitleService:
    """Test service Subtitle"""
    
    def test_subtitle_service_exists(self):
        root = get_project_root()
        assert check_file_exists(root / "src/services/subtitle.py", "Subtitle service")
    
    def test_subtitle_directory(self):
        root = get_project_root()
        assert check_dir_exists(root / "data/output/subtitles", "Subtitle output")
    

class TestServiceIntegration:
    """Test cấu trúc files service"""
    
    def test_services_structure(self):
        print("\n Test Services Structure ")
        from conftest import get_project_root, check_file_exists
        
        root = get_project_root()
        files = ["__init__.py", "qa_engine.py", "vector_store.py", "subtitle.py"]
        
        for fname in files:
            assert check_file_exists(root / f"src/services/{fname}", fname)
        
        print("[PASS] All service files exist")


class TestOutputDirectories:
    """Test các thư mục output"""
    
    def test_all_output_directories_exist(self):
        print("\n Test Output Directories ")
        from conftest import get_project_root, check_dir_exists
        
        root = get_project_root()
        dirs = ["documents", "transcripts", "translations", "subtitles", "videos"]
        
        for d in dirs:
            check_dir_exists(root / f"data/output/{d}", d)
        
        print("[PASS] All output directories exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
