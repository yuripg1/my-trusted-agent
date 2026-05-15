from pathlib import Path
from unittest.mock import MagicMock, patch

from reportlab.pdfgen import canvas as pdf_canvas

from tool.read_pdf_document import read_pdf_document


def _create_test_pdf(
    target: Path,
    page_contents: list[str] | None = None,
) -> None:
    """Create a PDF file with the given page contents for testing."""
    if page_contents is None:
        page_contents = ["First page content"]
    c: pdf_canvas.Canvas = pdf_canvas.Canvas(str(target))
    for page_content in page_contents:
        c.drawString(100, 700, page_content)
        c.showPage()
    c.save()


class TestReadPdfDocument:
    """Tests for the `read_pdf_document` tool"""

    def test_read_local_pdf_successfully(self, tmp_path: Path) -> None:
        """Read a local PDF document successfully"""
        target: Path = tmp_path.joinpath("doc.pdf")
        _create_test_pdf(target, ["First page content", "Second page content"])
        location_type: str = "local"
        location: str = str(target)
        result: str = read_pdf_document(location_type, location)
        expected_result: str = f'<pdf_document_read location_type="{location_type}" location="{location}">\n<pages>\n<page number="1">\nFirst page content\n</page>\n<page number="2">\nSecond page content\n</page>\n</pages>\n</pdf_document_read>'
        assert result == expected_result

    def test_read_local_pdf_not_found(self, tmp_path: Path) -> None:
        """Do not read a local PDF that does not exist"""
        target: Path = tmp_path.joinpath("nonexistent.pdf")
        location_type: str = "local"
        location: str = str(target)
        result: str = read_pdf_document(location_type, location)
        expected_result: str = f'<pdf_document_read location_type="{location_type}" location="{location}">\n<error>Could not fetch the PDF document</error>\n</pdf_document_read>'
        assert result == expected_result

    def test_invalid_location_type(self) -> None:
        """Use an invalid location type"""
        location_type: str = "invalid"
        location: str = "/path/to/doc.pdf"
        result: str = read_pdf_document(location_type, location)
        expected_result: str = f'<pdf_document_read location_type="{location_type}" location="{location}">\n<error>Invalid location_type "{location_type}"</error>\n</pdf_document_read>'
        assert result == expected_result

    def test_reading_denied_by_user(self, tmp_path: Path) -> None:
        """Do not read a PDF due to being denied by the user"""
        target: Path = tmp_path.joinpath("doc.pdf")
        _create_test_pdf(target)
        location_type: str = "local"
        location: str = str(target)
        result: str = read_pdf_document(location_type, location, tool_call_permission=False)
        expected_result: str = f'<pdf_document_read location_type="{location_type}" location="{location}">\n<error>PDF document reading manually denied by the user</error>\n</pdf_document_read>'
        assert result == expected_result

    def test_note_included(self, tmp_path: Path) -> None:
        """Include a note in the output"""
        target: Path = tmp_path.joinpath("doc.pdf")
        _create_test_pdf(target)
        location_type: str = "local"
        location: str = str(target)
        note: str = "Test note"
        result: str = read_pdf_document(location_type, location, note=note)
        expected_result: str = f'<pdf_document_read location_type="{location_type}" location="{location}">\n<note>{note}</note>\n<pages>\n<page number="1">\nFirst page content\n</page>\n</pages>\n</pdf_document_read>'
        assert result == expected_result

    def test_web_pdf_successfully(self, tmp_path: Path) -> None:
        """Read a PDF document from the web successfully"""
        url: str = "https://example.com/doc.pdf"
        status_code: int = 200
        content_type: str = "application/pdf"
        pdf_file: Path = tmp_path.joinpath("doc.pdf")
        _create_test_pdf(pdf_file, ["Web page 1 content", "Web page 2 content"])
        pdf_bytes: bytes = pdf_file.read_bytes()
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = pdf_bytes
        mock_response.headers = {"content-type": content_type}
        mock_client_instance: MagicMock = MagicMock()
        mock_client_instance.get.return_value = mock_response
        with patch("tool.read_pdf_document.Client", return_value=mock_client_instance):
            result: str = read_pdf_document("web", url)
        expected_result: str = f'<pdf_document_read location_type="web" location="{url}">\n<pages>\n<page number="1">\nWeb page 1 content\n</page>\n<page number="2">\nWeb page 2 content\n</page>\n</pages>\n</pdf_document_read>'
        assert result == expected_result

    def test_web_pdf_invalid_header(self) -> None:
        """Fetch content from the web whose header is not valid PDF"""
        url: str = "https://example.com/doc.pdf"
        status_code: int = 200
        content_type: str = "application/pdf"
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = b"not a pdf at all"
        mock_response.headers = {"content-type": content_type}
        mock_client_instance: MagicMock = MagicMock()
        mock_client_instance.get.return_value = mock_response
        with patch("tool.read_pdf_document.Client", return_value=mock_client_instance):
            result: str = read_pdf_document("web", url)
        expected_result: str = f'<pdf_document_read location_type="web" location="{url}">\n<error>The fetched file does not seem to be a valid PDF document</error>\n<status_code>{status_code}</status_code>\n<content_type>{content_type}</content_type>\n</pdf_document_read>'
        assert result == expected_result

    def test_web_pdf_corrupted_content(self) -> None:
        """Fetch content from the web whose header is valid but content is corrupted"""
        url: str = "https://example.com/doc.pdf"
        status_code: int = 200
        content_type: str = "application/pdf"
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = b"%PDF-1.4 fake content"
        mock_response.headers = {"content-type": content_type}
        mock_client_instance: MagicMock = MagicMock()
        mock_client_instance.get.return_value = mock_response
        with patch("tool.read_pdf_document.Client", return_value=mock_client_instance):
            result: str = read_pdf_document("web", url)
        expected_result: str = f'<pdf_document_read location_type="web" location="{url}">\n<error>Could not read the PDF document</error>\n<status_code>{status_code}</status_code>\n<content_type>{content_type}</content_type>\n</pdf_document_read>'
        assert result == expected_result

    def test_web_pdf_empty_pages(self) -> None:
        """Read a PDF from the web whose pages have no extractable text"""
        url: str = "https://example.com/doc.pdf"
        status_code: int = 200
        content_type: str = "application/pdf"
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = b"%PDF-1.4 valid header"
        mock_response.headers = {"content-type": content_type}
        mock_client_instance: MagicMock = MagicMock()
        mock_client_instance.get.return_value = mock_response
        with (
            patch("tool.read_pdf_document.Client", return_value=mock_client_instance),
            patch("tool.read_pdf_document.PdfReader") as mock_reader_class,
        ):
            mock_page: MagicMock = MagicMock()
            mock_page.extract_text.return_value = ""
            mock_reader_instance: MagicMock = MagicMock()
            mock_reader_instance.pages = [mock_page]
            mock_reader_class.return_value = mock_reader_instance
            result: str = read_pdf_document("web", url)
        expected_result: str = f'<pdf_document_read location_type="web" location="{url}">\n<error>Could not read the PDF document</error>\n<status_code>{status_code}</status_code>\n<content_type>{content_type}</content_type>\n</pdf_document_read>'
        assert result == expected_result

    def test_web_pdf_non_200_status(self) -> None:
        """Fetch a PDF from the web with a non-200 status code"""
        url: str = "https://example.com/doc.pdf"
        status_code: int = 500
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = b""
        mock_response.headers = {"content-type": "text/html"}
        mock_client_instance: MagicMock = MagicMock()
        mock_client_instance.get.return_value = mock_response
        with patch("tool.read_pdf_document.Client", return_value=mock_client_instance):
            result: str = read_pdf_document("web", url)
        expected_result: str = f'<pdf_document_read location_type="web" location="{url}">\n<error>Could not fetch the PDF document</error>\n<status_code>{status_code}</status_code>\n</pdf_document_read>'
        assert result == expected_result
