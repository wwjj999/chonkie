"""Test the TextChef class."""

import pytest

from chonkie.chef.text import TextChef


@pytest.fixture
def text_chef():
    """Create a default TextChef instance."""
    return TextChef()


@pytest.fixture
def custom_text_chef():
    """Create a TextChef with custom settings."""
    return TextChef(
        whitespace=True,
        newlines=True,
        abbreviations=True,
        ellipsis=True,
        mid_sentence_newlines=True,
        sentence_endings=".!?",
    )


class TestWhitespaceNormalization:

    """Test the whitespace normalization feature."""

    def test_multiple_spaces(self, text_chef):
        """Test normalizing multiple spaces."""
        text = "This    has    multiple    spaces"
        result = text_chef.clean(text)
        assert result == "This has multiple spaces"

    def test_empty_text(self, text_chef):
        """Test handling empty text."""
        assert text_chef.clean("") == ""
        assert text_chef.clean(None) == None


class TestNewlineNormalization:

    """Test the newline normalization feature."""

    def test_mixed_newlines(self, text_chef):
        """Test normalizing different types of newlines."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = text_chef.clean(text)
        assert result == "Line 1\nLine 2\nLine 3\nLine 4"

    def test_multiple_newlines(self, text_chef):
        """Test normalizing multiple newlines."""
        text = "Paragraph 1\n\n\n\nParagraph 2"
        result = text_chef.clean(text)
        assert result == "Paragraph 1\n\nParagraph 2"


class TestMidSentenceNewlines:

    """Test the mid sentence newlines feature."""

    def test_basic_line_wrapping(self, custom_text_chef):
        """Test basic line wrapping behavior."""
        text = "This is a\nwrapped line."
        result = custom_text_chef.clean(text)
        assert result == "This is a wrapped line."

    def test_preserve_paragraph_breaks(self, custom_text_chef):
        """Test preserving paragraph breaks."""
        text = "Paragraph 1.\n\nParagraph 2."
        result = custom_text_chef.clean(text)
        assert result == "Paragraph 1.\n\nParagraph 2."

    def test_sentence_boundaries(self, custom_text_chef):
        """Test handling sentence boundaries."""
        text = "First sentence.\nSecond sentence"
        result = custom_text_chef.clean(text)
        assert result == "First sentence.\nSecond sentence"

    def test_dialog_markers(self, custom_text_chef):
        """Test handling dialog markers."""
        text = 'He said,\n"Hello there!"'
        result = custom_text_chef.clean(text)
        assert result == 'He said,\n"Hello there!"'


class TestEllipsis:

    """Test the ellipsis feature."""

    def test_replace_dots(self, text_chef):
        """Test replacing multiple dots with ellipsis character."""
        text = "And then..."
        result = text_chef.clean(text)
        assert result == "And then…"

    def test_longer_dots(self, text_chef):
        """Test replacing longer sequences of dots."""
        text = "And then......."
        result = text_chef.clean(text)
        assert result == "And then…"


class TestAbbreviations:

    """Test the abbreviations feature."""

    def test_common_abbreviations(self, text_chef):
        """Test handling common abbreviations."""
        text = "Mr. Smith and Dr. Jones"
        result = text_chef.clean(text)
        assert "Mr․" in result  # Using dot leader
        assert "Dr․" in result

    def test_multiple_abbreviations(self, text_chef):
        """Test handling multiple abbreviations in text."""
        text = "Prof. Smith, Ph.D., M.D."
        result = text_chef.clean(text)
        # Note: as we handling the ending as non-word character the last abbreviation at the end won't get replaced
        # which is intended. 
        assert all(c not in result for c in ["Prof.", "Ph.D."])
        assert all(c in result for c in ["Prof․", "Ph․D․"])
        
    def test_abbreviation_with_word_boundaries(self):
        """Test handling abbreviation only with a word boundary beginning and a non-word character ending."""
        text_chef_no_email = TextChef(email=False)
        text = "My email is someone@domain.org. My height is 72 in. and weight is 65 kg."
        result = text_chef_no_email.clean(text)
        assert "someone@domain.org" in result  # dot leader shouldn't present here
        assert "in․" in result  # dot replaced with dot leader
        
    def test_abbreviation_with_prefixes(self, text_chef):
        """Test handling abbreviations with prefixes existing among themselves."""
        text = "The time now is 7 a.m. I will increase 1 m. after 1 hour."
        result = text_chef.clean(text)
        assert "a․m․" in result          # a.m. should be replaced individually
        assert "1 m․" in result         # m. should be replaced individually 
        

class TestEmail:
    
    """Test the emails feature."""
    
    def test_email(self, text_chef):
        """Test handling emails."""
        text = "someone@domain.org"
        result = text_chef.clean(text)
        assert "someone@domain․org" in result


class TestURL:
    
    """Test the URLs feature."""
    
    def test_url(self, text_chef):
        """Test handling URLs."""
        text = "https://cloud.chonkie.ai/"
        result = text_chef.clean(text)
        assert "https://cloud․chonkie․ai/" in result


class TestDecimalPoints:
    
    """Test the decimal points in between digits."""
    
    def test_decimals(self, text_chef):
        """Test handling decimal points."""
        text = "The average temperature in July was 25.5 degrees Celsius."
        result = text_chef.clean(text)
        assert "25․5" in result


class TestFeatureToggling:

    """Test the feature toggling feature."""

    def test_disable_whitespace(self):
        """Test with whitespace normalization disabled."""
        chef = TextChef(whitespace=False)
        text = "Multiple    spaces"
        result = chef.clean(text)
        assert result == text

    def test_disable_newlines(self):
        """Test with newline normalization disabled."""
        chef = TextChef(newlines=False)
        text = "Line 1\n\n\nLine 2"
        result = chef.clean(text)
        assert result == text

    def test_disable_abbreviations(self):
        """Test with abbreviation handling disabled."""
        chef = TextChef(abbreviations=False)
        text = "Mr. Smith"
        result = chef.clean(text)
        assert result == text

    def test_disable_ellipsis(self):
        """Test with ellipsis handling disabled."""
        chef = TextChef(ellipsis=False)
        text = "And then..."
        result = chef.clean(text)
        assert result == text
        
    def test_disable_emails(self):
        """Test with emails handling disabled."""
        chef = TextChef(email=False)
        text = "someone@domain.org"
        result = chef.clean(text)
        assert result == text
        
    def test_disable_urls(self):
        """Test with urls handling disabled."""
        chef = TextChef(url=False)
        text = "https://cloud․chonkie․ai/"
        result = chef.clean(text)
        assert result == text
        
    def test_disable_decimals(self):
        """Test with decimals handling disabled."""
        chef = TextChef(decimals=False)
        text = "25.5"
        result = chef.clean(text)
        assert result == text


class TestFileHandling:

    """Test the file handling feature."""

    def test_supported_extensions(self, text_chef, tmp_path):
        """Test handling supported file extensions."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test   content")
        result = text_chef.process(files=str(file_path))
        assert result == "Test content"

    def test_unsupported_extension(self, text_chef, tmp_path):
        """Test handling unsupported file extension."""
        file_path = tmp_path / "test.pdf"
        file_path.write_text("Test content")
        with pytest.raises(ValueError, match="not a supported extension"):
            text_chef.process(files=str(file_path))

    def test_directory_processing(self, text_chef, tmp_path):
        """Test processing multiple files in directory."""
        # Create test files
        (tmp_path / "test1.txt").write_text("File   1")
        (tmp_path / "test2.md").write_text("File   2")
        (tmp_path / "test3.pdf").write_text("File 3")  # Should be ignored

        result = text_chef.process(directory=str(tmp_path))
        assert len(result) == 2
        assert "File 1" in result
        assert "File 2" in result


class TestRevert:
    
    """Test the cleaned text reverted back to the original text."""
    
    def test_reverted_text(self, text_chef):
        """Test whether the cleaned text is reverted back to the original."""
        text = "My email is john.doe@example.com. Visit https://cloud.chonkie.ai/ for details. Pi is 3.14. Meetings at 10.30 a.m. and 5.45 p.m. Dr. Smith is a good doctor."
        cleaned_text = text_chef.clean(text)
        reverted_text = text_chef.revert(cleaned_text)
        assert reverted_text == text