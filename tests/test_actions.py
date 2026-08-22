"""Tests for core/actions.py."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from core.actions import get_date, get_sysinfo, get_time, open_app, web_search
from core.exceptions import ActionError


class TestGetTime:
    """Test get_time function."""

    def test_returns_formatted_time(self):
        """get_time should return HH:MM AM/PM format."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
            result = get_time()
            assert result == "02:30 PM"

    def test_returns_string(self):
        """get_time should return a string."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
            result = get_time()
            assert isinstance(result, str)

    def test_handles_midnight(self):
        """get_time should handle midnight (12:00 AM)."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 0, 0, 0)
            result = get_time()
            assert result == "12:00 AM"

    def test_handles_noon(self):
        """get_time should handle noon (12:00 PM)."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
            result = get_time()
            assert result == "12:00 PM"


class TestGetDate:
    """Test get_date function."""

    def test_returns_formatted_date(self):
        """get_date should return Weekday, Month DD, YYYY format."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
            result = get_date()
            assert result == "Monday, January 15, 2024"

    def test_returns_string(self):
        """get_date should return a string."""
        with patch("core.actions.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
            result = get_date()
            assert isinstance(result, str)


class TestGetSysinfo:
    """Test get_sysinfo function."""

    @patch("core.actions.psutil")
    def test_returns_dict_with_cpu_memory_disk(self, mock_psutil):
        """get_sysinfo should return dict with cpu, memory, disk percentages."""
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        mock_psutil.disk_usage.return_value = Mock(percent=45.0)

        result = get_sysinfo()

        assert isinstance(result, dict)
        assert "cpu_percent" in result
        assert "memory_percent" in result
        assert "disk_percent" in result
        assert result["cpu_percent"] == 25.5
        assert result["memory_percent"] == 60.0
        assert result["disk_percent"] == 45.0

    @patch("core.actions.psutil")
    def test_raises_action_error_on_failure(self, mock_psutil):
        """get_sysinfo should raise ActionError on psutil failure."""
        mock_psutil.cpu_percent.side_effect = OSError("Permission denied")

        with pytest.raises(ActionError) as exc_info:
            get_sysinfo()
        assert "Failed to get system info" in str(exc_info.value)


class TestOpenApp:
    """Test open_app function."""

    @patch("core.actions.shutil.which")
    @patch("core.actions.subprocess.Popen")
    def test_launches_allowed_app(self, mock_popen, mock_which):
        """open_app should launch app via Popen with argv list."""
        mock_which.return_value = "/usr/bin/firefox"
        mock_process = Mock()
        mock_popen.return_value = mock_process

        result = open_app("firefox")

        mock_which.assert_called_once_with("firefox")
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert args[0] == ["/usr/bin/firefox"]
        assert (
            kwargs.get("shell") is not True
        )  # shell defaults to False, ensure not explicitly True
        assert "Successfully launched" in result

    @patch("core.actions.shutil.which")
    def test_rejects_unknown_app(self, mock_which):
        """open_app should reject apps not in PATH."""
        mock_which.return_value = None

        with pytest.raises(ActionError) as exc_info:
            open_app("nonexistent_app")
        assert "not found in PATH" in str(exc_info.value)

    @patch("core.actions.shutil.which")
    @patch("core.actions.subprocess.Popen")
    def test_raises_action_error_on_popen_failure(self, mock_popen, mock_which):
        """open_app should raise ActionError if Popen fails."""
        mock_which.return_value = "/usr/bin/firefox"
        mock_popen.side_effect = OSError("Permission denied")

        with pytest.raises(ActionError) as exc_info:
            open_app("firefox")
        assert "Failed to launch" in str(exc_info.value)

    def test_no_shell_true(self):
        """open_app should never use shell=True."""
        with (
            patch("core.actions.shutil.which", return_value="/usr/bin/firefox"),
            patch("core.actions.subprocess.Popen") as mock_popen,
        ):
            open_app("firefox")
            args, kwargs = mock_popen.call_args
            assert kwargs.get("shell") is not True


class TestWebSearch:
    """Test web_search function."""

    @patch("core.actions.webbrowser.open_new_tab")
    @patch("core.actions.urllib.parse.quote_plus")
    def test_opens_browser_with_encoded_query(self, mock_quote, mock_open):
        """web_search should URL-encode query and open browser."""
        mock_quote.return_value = "hello+world"

        result = web_search("hello world")

        mock_quote.assert_called_once_with("hello world")
        mock_open.assert_called_once()
        args, _ = mock_open.call_args
        assert "hello+world" in args[0]
        assert "Successfully searched" in result

    @patch("core.actions.webbrowser.open_new_tab")
    def test_raises_action_error_on_failure(self, mock_open):
        """web_search should raise ActionError if browser fails."""
        mock_open.side_effect = OSError("No browser")

        with pytest.raises(ActionError) as exc_info:
            web_search("test")
        assert "Failed to search" in str(exc_info.value)

    def test_handles_special_characters(self):
        """web_search should handle special characters in query."""
        with (
            patch("core.actions.urllib.parse.quote_plus") as mock_quote,
            patch("core.actions.webbrowser.open_new_tab"),
        ):
            mock_quote.return_value = "test%2Bquery"
            result = web_search("test+query")
            mock_quote.assert_called_once_with("test+query")
            assert "Successfully searched" in result


class TestSecurity:
    """Security-focused tests."""

    def test_no_shell_true_anywhere(self):
        """Ensure no shell=True in actions module (actual code, not comments)."""
        import inspect

        import core.actions as actions_module

        source = inspect.getsource(actions_module)
        # Remove comments and docstrings, then check
        lines = []
        for line in source.split("\n"):
            stripped = line.strip()
            if (
                not stripped.startswith("#")
                and not stripped.startswith('"""')
                and not stripped.startswith("'''")
            ):
                lines.append(line)
        code = "\n".join(lines)
        assert "shell=True" not in code

    def test_no_os_system(self):
        """Ensure no os.system calls."""
        import inspect

        import core.actions as actions_module

        source = inspect.getsource(actions_module)
        assert "os.system" not in source

    def test_no_eval_exec(self):
        """Ensure no eval/exec calls."""
        import inspect

        import core.actions as actions_module

        source = inspect.getsource(actions_module)
        assert "eval(" not in source
        assert "exec(" not in source
