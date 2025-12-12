# tests/test_ingestion_service.py
"""Unit tests for ingestion service."""
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from app.services.ingestion_service import (
    save_uploaded_files,
    cleanup_temp_files
)


class TestIngestionService(unittest.TestCase):
    """Test ingestion service."""
    
    def test_save_uploaded_files_valid_extensions(self):
        """Test saving files with valid extensions."""
        import asyncio
        
        # Create mock file objects with async read
        mock_file1 = MagicMock()
        mock_file1.filename = "test.txt"
        async def read1():
            return b"test content 1"
        mock_file1.read = read1
        
        mock_file2 = MagicMock()
        mock_file2.filename = "test.pdf"
        async def read2():
            return b"test content 2"
        mock_file2.read = read2
        
        files = [mock_file1, mock_file2]
        
        async def async_test():
            file_paths, temp_dir = await save_uploaded_files(files)
            self.assertIsNotNone(temp_dir)
            self.assertEqual(len(file_paths), 2)
            # Verify files exist
            for file_path in file_paths:
                self.assertTrue(os.path.exists(file_path))
            # Cleanup
            cleanup_temp_files(file_paths, temp_dir)
        
        asyncio.run(async_test())
    
    def test_save_uploaded_files_invalid_extension(self):
        """Test that invalid file extensions raise ValueError."""
        import asyncio
        
        mock_file = MagicMock()
        mock_file.filename = "test.exe"
        async def read():
            return b"content"
        mock_file.read = read
        
        async def async_test():
            with self.assertRaises(ValueError) as context:
                await save_uploaded_files([mock_file])
            self.assertIn("Unsupported file type", str(context.exception))
        
        asyncio.run(async_test())
    
    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files."""
        # Create a temp file
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        file_paths = [test_file]
        
        # Verify file exists
        self.assertTrue(os.path.exists(test_file))
        
        # Cleanup
        cleanup_temp_files(file_paths, temp_dir)
        
        # Verify file and directory are gone
        self.assertFalse(os.path.exists(test_file))
        self.assertFalse(os.path.exists(temp_dir))


if __name__ == '__main__':
    unittest.main()
